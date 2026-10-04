"""Thin artifact loader and adapter around the project's existing inference functions."""
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import joblib
import numpy as np
import torch

from .project import CANONICAL_14_SENSORS, SUBSETS
from .validation import ValidatedData
from src.preprocessing import RegimeAwarePreprocessor
from src.dataset import _build_windows
from src.models.predictor import SupervisedPredictor
from src.td3.agent import TD3Agent
from src.lqr import LQRSafetyFilter
from src.safety_gate import SafetyGate
from src.evaluation import run_full_inference
from src.metrics import calculate_metrics

ARTIFACT_ROOT = Path(os.environ.get("CMAPSS_MODEL_DIR", Path(__file__).parent / "artifacts")).resolve()
FILES = ("manifest.json", "predictor.pt", "td3_actor.pt", "td3_critic.pt", "preprocessor.joblib", "safety_gate.json")


class ModelUnavailable(RuntimeError):
    pass


def artifact_status(subset):
    directory = ARTIFACT_ROOT / subset
    missing = [name for name in FILES if not (directory / name).is_file()]
    return {"subset": subset, "ready": not missing, "reason": "Trained artifacts found; loading is checked during inference." if not missing else "Trained model could not be loaded. Add the trained artifact bundle to the configured model directory.", "missing": missing}


@dataclass
class Bundle:
    predictor: object
    agent: object
    lqr: object
    gate: object
    preprocessor: object
    manifest: dict
    lock: object


def load_bundle(subset):
    if subset not in SUBSETS:
        raise ModelUnavailable("Unsupported C-MAPSS subset.")
    directory = ARTIFACT_ROOT / subset
    status = artifact_status(subset)
    if not status["ready"]:
        raise ModelUnavailable(status["reason"])
    fingerprint = tuple((name, (directory / name).stat().st_mtime_ns, (directory / name).stat().st_size) for name in FILES)
    return _load_bundle(subset, fingerprint)


@lru_cache(maxsize=4)
def _load_bundle(subset, fingerprint):
    directory = ARTIFACT_ROOT / subset
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["schemaVersion"] != 1 or manifest["subset"] != subset:
            raise ValueError("Artifact manifest does not match the selected subset.")
        if manifest["features"] != CANONICAL_14_SENSORS:
            raise ValueError("Artifact features do not match the existing pipeline.")
        if manifest["windowSelection"] not in ("all_windows", "last_window"):
            raise ValueError("The trained run's window selection must be recorded.")
        if not isinstance(manifest["windowLength"], int) or manifest["windowLength"] < 1:
            raise ValueError("Invalid recorded window length.")
        preprocessor = joblib.load(directory / "preprocessor.joblib")
        if not isinstance(preprocessor, RegimeAwarePreprocessor) or preprocessor.kmeans is None:
            raise ValueError("The preprocessor must be fitted on the original training data.")
        for vector in (preprocessor.global_mean, preprocessor.global_std):
            if vector is None or np.asarray(vector).shape != (len(CANONICAL_14_SENSORS),) or not np.isfinite(vector).all():
                raise ValueError("Invalid fitted normalization statistics.")
        if (preprocessor.global_std <= 0).any():
            raise ValueError("Normalization standard deviations must be positive.")
        device = torch.device("cpu")
        predictor = SupervisedPredictor(**manifest["predictorParameters"]).to(device)
        predictor.load_state_dict(torch.load(directory / "predictor.pt", map_location=device, weights_only=True), strict=True)
        predictor.eval()
        agent = TD3Agent(**manifest["td3Parameters"], device=device)
        agent.load(str(directory / "td3_actor.pt"), str(directory / "td3_critic.pt"))
        lqr = LQRSafetyFilter(**manifest["lqrParameters"])
        gate = SafetyGate()
        gate.load(str(directory / "safety_gate.json"))
        return Bundle(predictor, agent, lqr, gate, preprocessor, manifest, threading.Lock())
    except Exception as error:
        raise ModelUnavailable(f"Trained model could not be loaded. {error}") from error


def predict_uploaded(data: ValidatedData, subset: str, truth_text: str | None, truth: dict | None, progress):
    bundle = load_bundle(subset)
    manifest = bundle.manifest
    # The model and fitted preprocessor are reused; no training or fitting occurs here.
    with bundle.lock, TemporaryDirectory(prefix="cmapss-inference-") as temporary:
        test_path = Path(temporary) / "test.txt"
        test_path.write_text(data.text)
        rul_path = None
        if truth_text is not None:
            rul_path = Path(temporary) / "RUL.txt"
            rul_path.write_text(truth_text)
        progress("preprocessing", "running")
        features, labels, _, unit_ids = bundle.preprocessor.transform(str(test_path), test_rul_path=str(rul_path) if rul_path else None)
        if truth is None:
            # run_full_inference requires a label-shaped argument but never uses it
            # to predict. Unknown labels remain NaN and are never returned as truth.
            labels = np.full(len(features), np.nan, dtype=np.float32)
        progress("preprocessing", "completed")
        progress("features", "running")
        windows, labels, engine_ids, cycles_norm = _build_windows(features, labels, unit_ids, manifest["windowLength"])
        unique_ids = np.unique(engine_ids)
        last = np.asarray([np.flatnonzero(engine_ids == unit)[-1] for unit in unique_ids])
        if manifest["windowSelection"] == "last_window":
            windows, labels, engine_ids, cycles_norm = windows[last], labels[last], engine_ids[last], cycles_norm[last]
            last = np.arange(len(unique_ids))
        progress("features", "completed")
        progress("inference", "running")
        result = run_full_inference(
            bundle.predictor, bundle.agent, bundle.lqr, bundle.gate,
            torch.from_numpy(windows), torch.from_numpy(labels), torch.from_numpy(cycles_norm),
            torch.from_numpy(engine_ids.astype(np.int64)), torch.device("cpu"),
            corruption_mode="clean", **manifest["inferenceParameters"],
        )
        output = np.asarray(result["y_final"])[last]
        if len(output) != len(unique_ids) or not np.isfinite(output).all() or (output < 0).any():
            raise RuntimeError("The existing pipeline returned invalid predictions.")
        progress("inference", "completed")
        progress("prediction", "running")
        predictions = []
        for unit, predicted in zip(unique_ids, output):
            row = {"engineId": int(unit), "cycle": int(data.rows[data.rows[:, 0] == unit, 1][-1]), "predictedRUL": float(predicted)}
            if truth is not None:
                row["actualRUL"] = truth[int(unit)]
                row["absoluteError"] = abs(row["predictedRUL"] - row["actualRUL"])
            predictions.append(row)
        evaluation = None
        if truth is not None:
            actual = np.asarray([truth[int(unit)] for unit in unique_ids])
            metrics = calculate_metrics(output, actual)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError("Evaluation produced non-finite metrics. Check the supplied ground-truth RUL values.")
            evaluation = {"metrics": metrics, "predictions": predictions, "source": "Existing evaluation pipeline · uploaded test file · final observation per engine", "subset": subset, "scope": "Last observation per engine"}
        progress("prediction", "completed")
        return {"predictions": predictions, "evaluation": evaluation, "subset": subset, "modelName": manifest["modelName"], "windowSelection": manifest["windowSelection"]}
