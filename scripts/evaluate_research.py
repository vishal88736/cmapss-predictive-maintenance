"""Evaluate a frozen format-7/8 bundle without training, selection or recalibration.

The evaluation plan is saved before opening official data. Each prediction and
its target-specific interval is retained. Previously inspected benchmark tests
remain exploratory; test errors must not choose model settings or thresholds.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from grace_rul.artifacts import load_bundle
from grace_rul.data import FAULTS, WindowDataset, load_subset, source_hashes
from grace_rul.training import device_for, evaluate, exposure_metrics, interval_bounds, metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protocol_rows(records, report, calibration, config):
    """Recompute every metric from saved-case predictions, for each target law."""
    records = records.copy()
    if not np.isfinite(records[["prediction", "target", "raw_target"]].to_numpy()).all():
        raise FloatingPointError("Nonfinite predictions or targets; evaluation is incomplete.")
    rows = []
    capped = f"capped_{config['train_rul_cap']:g}"
    for protocol, column, prefix in [(capped, "target", "capped"), ("official_uncapped", "raw_target", "raw")]:
        lo, hi = interval_bounds(records.prediction, calibration, protocol=protocol)
        records[f"{prefix}_interval_lower"] = lo
        records[f"{prefix}_interval_upper"] = hi
        contract = calibration["protocols"][protocol]
        rows.append({**report, **metrics(records.prediction, records[column]),
            **exposure_metrics(records, records[column]), "protocol": protocol,
            "Interval_Coverage": float(((records[column] >= lo) & (records[column] <= hi)).mean()),
            "Interval_Method": calibration["method"], "Interval_Policy": calibration["endpoint_policy"],
            "Interval_Unbounded": bool(contract["unbounded"]),
            "Interval_Radius": contract["radius"]})
    return rows, records


def evaluate_bundle(bundle_path, output_dir=None, *, project_root=ROOT, device="auto", subsets=None,
                    clean_only=False, cpu_threads=2):
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer.")
    source = Path(bundle_path).resolve()
    root = Path(project_root).resolve()
    selected_device = device_for({"device": device})
    torch.set_num_threads(cpu_threads)
    before = sha256(source)
    bundle, loaded = load_bundle(source, device=str(selected_device))
    if bundle["format_version"] < 7:
        raise ValueError("Use a format-7/8 bundle with separate capped/raw calibration contracts.")
    requested = list(loaded) if subsets is None else list(subsets)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(loaded):
        raise ValueError("Requested evaluation subsets must be unique and present in the bundle.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (Path(output_dir) if output_dir else root / "artifacts" / f"frozen_evaluation_{stamp}").resolve()
    if output == source.parent or source.parent in output.parents:
        raise ValueError("Evaluation output must be outside the preserved source run directory.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    cases = [("clean", 0., 0)]
    if not clean_only:
        cases.extend((family, severity, replicate) for family in [*FAULTS, "mixed"]
                     for severity in [.1, .2, .3, .4, .5] for replicate in [0, 1, 2])
    states, configurations = {}, {}
    for subset in requested:
        item = loaded[subset]
        item["model"].eval().requires_grad_(False)
        states[subset] = model_sha256(item["model"])
        # Runtime dataset limits must never silently omit official endpoints.
        configurations[subset] = {**item["model"].config, "device": str(selected_device),
            "maximum_batches": None, "maximum_windows": None, "maximum_engines": None}
    plan = {"created_at_utc": stamp, "source_bundle": str(source), "source_bundle_sha256": before,
        "format_version": bundle["format_version"], "architecture": bundle["architecture"],
        "corruption_protocol": bundle["corruption_protocol"], "device": str(selected_device),
        "torch_version": str(torch.__version__), "cpu_threads": cpu_threads, "subsets": requested,
        "run_type": "frozen_model_evaluation", "weights_retrained": False,
        "selection_performed": False, "recalibration_performed": False,
        "evaluation_status": "exploratory; these official benchmark tests were inspected in prior development",
        "scope": "Saved validation-selected subset models; all official final observed endpoints, not all training seeds.",
        "source_quick_run": bool(bundle["config"]["quick_run"]), "case_count_per_subset": len(cases),
        "cases": [{"family": f, "severity": s, "replicate": k} for f, s, k in cases],
        "model_state_sha256": states, "evaluation_configurations": configurations,
        "repair_selected": {s: bool(loaded[s]["model"].repair_enabled) for s in requested},
        "selected_seeds": {s: int(loaded[s]["entry"]["selected_seed"]) for s in requested}}
    # Persist the complete frozen plan before source_hashes/load_subset open test files.
    write_json(output / "evaluation_plan.json", plan)
    manifest = {**plan, "status": "running", "input_sha256": {}}
    write_json(output / "run_manifest.json", manifest)
    print(f"Evaluation device: {selected_device} | Source format: {bundle['format_version']} | "
          f"{len(cases)} cases/subset | Output: {output}", flush=True)
    clean, robustness, diagnostics = [], [], []
    try:
        with torch.no_grad():
            for subset in requested:
                item, config = loaded[subset], configurations[subset]
                actual_hashes = source_hashes(root / "data", subset)
                expected_hashes = item["entry"]["source_hashes"]
                if actual_hashes != expected_hashes:
                    raise ValueError(f"{subset}: source files differ from the bundle's saved data provenance.")
                manifest["input_sha256"][subset] = actual_hashes
                write_json(output / "run_manifest.json", manifest)
                _, test, truths = load_subset(root / "data", subset)
                seed = int(item["entry"]["selected_seed"])
                clean_rmse = {}
                for family, severity, replicate in cases:
                    scenario_seed = seed + 3 if family == "clean" else config["validation_seed"] + 100000 + replicate
                    dataset = WindowDataset(test, item["scaler"], config, mode="test", family=family,
                        severity=severity, seed=scenario_seed, truths=truths)
                    report, records = evaluate(item["model"], dataset, config,
                        fault_thresholds=item["entry"]["thresholds"], capture_recovery_tails=True)
                    if len(records) != test.unit.nunique() or records.unit.duplicated().any():
                        raise ValueError(f"{subset}: evaluation did not cover exactly one endpoint per test engine.")
                    rows, records = protocol_rows(records, report, item["entry"]["calibration"], config)
                    meta = {"subset": subset, "seed": seed, "family": family, "severity": severity, "replicate": replicate}
                    records.assign(**meta).to_csv(output / "predictions" /
                        f"{subset}_{family}_s{severity:g}_r{replicate}.csv.gz", index=False, compression="gzip")
                    diagnostics.append({**meta, **report})
                    for row in rows:
                        if family == "clean":
                            clean_rmse[row["protocol"]] = row["RMSE"]
                            clean.append({"subset": subset, "seed": seed, **row})
                        else:
                            robustness.append({**meta, **row, "relative_RMSE_increase":
                                row["RMSE"] / max(clean_rmse[row["protocol"]], 1e-8) - 1})
                    # Keep completed cases inspectable after an interruption.
                    pd.DataFrame(clean).to_csv(output / "test_metrics.csv", index=False)
                    pd.DataFrame(diagnostics).to_csv(output / "recovery_diagnostics.csv", index=False)
                    if robustness:
                        pd.DataFrame(robustness).to_csv(output / "robustness_summary.csv", index=False)
                    print(f"{subset}: {family} severity={severity:g} replicate={replicate} complete", flush=True)
                if model_sha256(item["model"]) != states[subset]:
                    raise ValueError(f"{subset}: model parameters/buffers changed during evaluation.")
                if source_hashes(root / "data", subset) != actual_hashes:
                    raise ValueError(f"{subset}: source data changed during evaluation.")
        if sha256(source) != before:
            raise ValueError("Source bundle changed during evaluation.")
        manifest.update(status="complete", weights_and_buffers_unchanged=True, source_bundle_unchanged=True,
            clean_metric_rows=len(clean), robustness_metric_rows=len(robustness),
            evaluated_contexts=len(diagnostics), completed_at_utc=datetime.now(timezone.utc).isoformat(),
            output_sha256={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                           if p.is_file() and p.name != "run_manifest.json"})
        write_json(output / "run_manifest.json", manifest)
    except Exception as error:
        manifest.update(status="error", error=f"{type(error).__name__}: {error}")
        write_json(output / "run_manifest.json", manifest)
        raise
    print(f"Frozen evaluation saved: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, help="New directory, outside the preserved source run.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--subsets", nargs="+", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--clean-only", action="store_true", help="Clean endpoints only; omit the 8-family/5-severity/3-replicate sweep.")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    evaluate_bundle(args.bundle, args.output_dir, device=args.device, subsets=args.subsets,
                    clean_only=args.clean_only, cpu_threads=args.cpu_threads)


if __name__ == "__main__":
    main()
