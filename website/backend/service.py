"""Versioned model loading, selective recovery, fault reporting, and explanations."""

import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
import pandas as pd
import torch
from grace_rul.artifacts import load_bundle
from grace_rul.config import DEFAULT_CONFIG, FORMAT_VERSION, SENSOR_NAMES, SENSORS
from grace_rul.data import make_window
from grace_rul.explain import explain_window
from grace_rul.training import interval_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InputError(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def spans(cycles, flagged):
    result, start, end = [], None, None
    for cycle, flag in zip(cycles, flagged):
        if flag:
            if start is None:
                start = int(cycle)
            elif int(cycle) != end + 1:
                result.append({"start": start, "end": end}); start = int(cycle)
            end = int(cycle)
        elif start is not None:
            result.append({"start": start, "end": end}); start = end = None
    if start is not None:
        result.append({"start": start, "end": end})
    return result


class InferenceService:
    def __init__(self):
        self.lock = threading.RLock()
        self.bundle, self.loaded = None, {}
        self.path = self.signature = self.load_error = None
        self.config = dict(DEFAULT_CONFIG)
        self.device = torch.device(os.getenv("GRACE_DEVICE", "cpu"))
        torch.set_num_threads(max(1, int(os.getenv("GRACE_CPU_THREADS", "2"))))

    def candidate(self):
        explicit = os.getenv("GRACE_MODEL_PATH")
        if explicit:
            path = Path(explicit).expanduser().resolve()
            return path if path.is_file() else None
        root = Path(os.getenv("GRACE_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
        candidates = []
        for path in root.glob("*/grace_rul_ALL.pt"):
            try:
                meta = json.loads((path.parent / "run_manifest.json").read_text())
                if meta.get("format_version") == FORMAT_VERSION and meta.get("quick_run") is False:
                    candidates.append(path)
            except (OSError, ValueError):
                continue
        return max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None

    def refresh(self):
        with self.lock:
            path = self.candidate()
            if path is None:
                self.bundle, self.loaded, self.path, self.signature = None, {}, None, None
                self.config = dict(DEFAULT_CONFIG)
                self.load_error = "The selected model file does not exist." if os.getenv("GRACE_MODEL_PATH") else None
                return
            signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            if signature == self.signature:
                return
            try:
                bundle, loaded = load_bundle(path, self.device)
                self.bundle, self.loaded, self.config, self.path = bundle, loaded, bundle["config"], path
                self.load_error = None
            except Exception as exc:
                self.bundle, self.loaded, self.path, self.load_error = None, {}, path, str(exc)
                self.config = dict(DEFAULT_CONFIG)
            self.signature = signature

    def status(self):
        self.refresh()
        with self.lock:
            ready = self.bundle is not None
            quick = ready and self.config.get("quick_run", False)
            parameters = {name: sum(p.numel() for entry in self.loaded.values() for p in getattr(entry["model"], name).parameters())
                          for name in ["detector", "generator", "predictor", "corrector"]} if ready else None
            return {"ready": ready, "state": "error" if self.load_error else "ready" if ready else "awaiting_model",
                "message": self.load_error or ("Quick validation model connected; results are execution checks." if quick else "Trained model connected." if ready else "Your model slot is ready. Complete full notebook training to connect the model."),
                "model_name": "GRACE-RUL", "format_version": FORMAT_VERSION, "bundle_name": self.path.parent.name if self.path else None,
                "mode": "quick" if quick else "trained" if ready else "pending", "device": str(self.device),
                "config": self.config, "supported_subsets": list(self.loaded) if ready else self.config["subsets"], "parameters": parameters,
                "interval_confidence": 100 * (1 - self.config["interval_alpha"]), "interval_method": "empirical calibration residual quantiles",
                "explanations_available": bool(importlib.util.find_spec("shap") and importlib.util.find_spec("lime")),
                "placeholder": {"notebook": "GRACE_RUL_all_subsets.ipynb", "artifact": "artifacts/<run>/grace_rul_ALL.pt", "manifest": "models/model_manifest.json"}}

    def model_report(self):
        status = self.status()
        with self.lock:
            root = self.path.parent if self.bundle else None
            tables = {}
            for name in ["test_metrics", "training_history", "correction_history", "robustness_summary", "ablation_metrics", "seed_summary", "cross_validation_metrics", "cross_validation_summary"]:
                try:
                    path = root / f"{name}.csv" if root else None
                    tables[name] = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).to_dict("records") if path and path.is_file() else []
                except (ValueError, OSError):
                    tables[name] = []
            return {**status, "tables": tables, "selected_seeds": {s: e["entry"]["selected_seed"] for s, e in self.loaded.items()}}

    def require_model(self):
        if not self.bundle:
            raise ModelUnavailable(self.load_error or "No trained model is connected. Run the notebook with QUICK_RUN = False, then refresh model status.")

    def predict(self, frame, filename, mc_dropout=False):
        from .inference import clean_json, preview, regression_metrics
        self.refresh()
        with self.lock, torch.inference_mode():
            self.require_model()
            unknown = set(frame["subset"]) - set(self.loaded)
            if unknown:
                raise InputError("The connected model does not support: " + ", ".join(sorted(unknown)))
            started, engines = time.perf_counter(), []
            length = self.config["sequence_length"]
            groups = list(frame.groupby(["subset", "unit"], sort=True))
            samples = min(8, max(1, 2048 // len(groups)))
            for (subset, unit), group in groups:
                item = self.loaded[subset]
                model, scaler, entry = item["model"], item["scaler"], item["entry"]
                values, regimes = scaler.transform(group)
                ends = np.unique(np.linspace(min(19, len(group) - 1), len(group) - 1, min(samples, len(group)), dtype=int))
                pairs = [make_window(values, int(end), length) for end in ends]
                x = torch.tensor(np.stack([w for w, _ in pairs]), device=self.device)
                valid = torch.tensor(np.stack([v for _, v in pairs]), device=self.device)
                out = model(x, valid)
                prediction = out["prediction"].cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ModelUnavailable("The model produced a non-finite prediction. Check the input and artifact.")
                lo, hi = interval_bounds(prediction, entry["calibration"])
                final_valid = valid[-1]
                repaired = out["repaired"][-1, final_valid].cpu().numpy()
                observed = group.tail(len(repaired))
                finite = np.isfinite(observed[SENSORS].to_numpy(float))
                scores = out["fault_scores"][-1, final_valid].cpu().numpy()
                thresholds = np.asarray(entry["thresholds"])
                flags = (scores >= thresholds[None]) & finite
                cycles = observed.cycle.to_numpy()
                physical = repaired * np.asarray(scaler.state["sensor_std"])[regimes[-len(repaired):]] + np.asarray(scaler.state["sensor_mean"])[regimes[-len(repaired):]]
                original = observed[SENSORS].to_numpy(float)
                sensor_reports = []
                for j, name in enumerate(SENSORS):
                    available_scores = scores[finite[:, j], j]
                    missing, suspected = int((~finite[:, j]).sum()), int(flags[:, j].sum())
                    delta = np.abs(repaired[:, j] - np.nan_to_num(values[-len(repaired):, 3 + j]))
                    sensor_reports.append({"sensor": name, "name": SENSOR_NAMES[j], "availability": 100 * float(finite[:, j].mean()),
                        "missing": missing, "suspected_cycles": suspected, "status": "missing + suspected" if missing and suspected else "missing" if missing else "suspected" if suspected else "no flag",
                        "fault_score": float(available_scores.max()) if len(available_scores) else None, "threshold": float(thresholds[j]),
                        "repair_magnitude": float(delta.mean()), "fault_spans": spans(cycles, flags[:, j]), "missing_spans": spans(cycles, ~finite[:, j])})
                target = group.iloc[-1].get("true_rul", np.nan)
                truth = float(target) if pd.notna(target) else None
                mc_std = None
                if mc_dropout:
                    for module in model.predictor.modules():
                        if isinstance(module, torch.nn.Dropout):
                            module.train()
                    try:
                        draws = torch.stack([model(x[-1:], valid[-1:])["prediction"] for _ in range(self.config["mc_passes"])])
                        mc_std = float(draws.std(unbiased=False))
                    finally:
                        model.eval()
                engines.append({"id": f"{subset}:{unit}", "subset": subset, "unit": int(unit), "last_cycle": int(group.iloc[-1].cycle), "observations": len(group),
                    "predicted_rul": float(prediction[-1]), "base_rul": float(out["base"][-1]), "correction_cycles": float(out["delta"][-1]),
                    "interval_lower": float(lo[-1]), "interval_upper": float(hi[-1]), "interval_width": float(hi[-1] - lo[-1]),
                    "interval_scope": entry["calibration"]["scope"], "mc_std_cycles": mc_std,
                    "estimated_failure_cycle": int(group.iloc[-1].cycle) + float(prediction[-1]), "true_rul": truth, "error": float(prediction[-1]) - truth if truth is not None else None,
                    "availability": 100 * float(finite.mean()), "missing_readings": int((~finite).sum()), "padded_cycles": max(0, length - len(group)),
                    "operating_regime": int(regimes[-1]), "local_gate": float(out["gates"][-1, 0]), "global_gate": float(out["gates"][-1, 1]),
                    "suspected_sensors": sum(s["suspected_cycles"] > 0 for s in sensor_reports), "sensors": sensor_reports,
                    "trajectory": [{"cycle": int(group.iloc[int(end)].cycle), "predicted_rul": float(p), "interval_lower": float(l), "interval_upper": float(h)} for end, p, l, h in zip(ends, prediction, lo, hi)],
                    "sensor_history": [{"cycle": int(cycle), "observed": original[j].tolist(), "repaired": physical[j].tolist()} for j, cycle in enumerate(cycles)]})
            predictions = np.asarray([e["predicted_rul"] for e in engines])
            return clean_json({"filename": filename, "demo": False, "model": {"name": "GRACE-RUL", "bundle": self.path.parent.name, "mode": "quick" if self.config["quick_run"] else "trained", "device": str(self.device)},
                "input": preview(frame, filename), "engines": engines, "summary": {"engines": len(engines), "rows": len(frame), "mean_rul": float(predictions.mean()), "min_rul": float(predictions.min()), "max_rul": float(predictions.max()),
                    "availability": preview(frame, filename)["availability"], "mean_interval_width": float(np.mean([e["interval_width"] for e in engines])), "near_failure": int((predictions <= 25).sum()),
                    "suspected_sensors": sum(e["suspected_sensors"] for e in engines), "interval_confidence": 100 * (1 - self.config["interval_alpha"]), "elapsed_seconds": time.perf_counter() - started},
                "metrics": regression_metrics(engines), "metrics_by_subset": {s: regression_metrics([e for e in engines if e["subset"] == s]) for s in frame["subset"].unique()}})

    def explain(self, frame, subset, unit, sensor=None):
        from .inference import clean_json
        self.refresh()
        with self.lock:
            self.require_model()
            if subset not in self.loaded:
                raise InputError("The connected model does not support this subset.")
            group = frame[(frame["subset"] == subset) & (frame["unit"] == unit)]
            if group.empty:
                raise InputError("The selected engine is not in the uploaded file.")
            if sensor is not None and not 1 <= sensor <= 21:
                raise InputError("Sensor number must be between 1 and 21.")
            item = self.loaded[subset]
            normalized, _ = item["scaler"].transform(group)
            x, valid = make_window(normalized, len(group) - 1, self.config["sequence_length"])
            if sensor is None:
                with torch.no_grad():
                    out = item["model"](torch.tensor(x[None], device=self.device), torch.tensor(valid[None], device=self.device))
                finite = np.isfinite(x[:, 3:]) & valid[:, None]
                available = finite.sum(0)
                scores = (out["fault_scores"][0].cpu().numpy() * finite).sum(0) / np.maximum(1, available)
                scores[available == 0] = -1
                target = int(scores.argmax()) if available.any() else None
            else:
                target = sensor - 1
                if not np.isfinite(x[valid, 3 + target]).any():
                    raise InputError("This sensor is entirely missing. Its outage is directly observable; choose an available sensor for finite-fault explanations.")
            result = explain_window(item["model"], x, valid, item["entry"]["background"], item["entry"]["background_valid"],
                                    target_sensor=target, samples=self.config["explanation_samples"], seed=self.config["seed"])
            return clean_json({**result, "engine": f"{subset}:{unit}", "model_bundle": self.path.parent.name,
                               "note": "SHAP and LIME explain estimated fault scores and RUL. Sensor importance alone does not confirm a fault."})
