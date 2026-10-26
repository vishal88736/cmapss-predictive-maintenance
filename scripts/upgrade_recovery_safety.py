#!/usr/bin/env python3
"""Validate a frozen format-7 model with fitting-support recovery, then export v8.

This command performs no training or official-test evaluation. Existing neural
weights, detector thresholds, and the selected corrector remain unchanged.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from grace_rul.artifacts import load_bundle
from grace_rul.calibration import calibrate_endpoint_intervals
from grace_rul.checkpoints import matches_pipeline_identity, save_checkpoint
from grace_rul.config import ARCHITECTURE, CORRUPTION_PROTOCOL, FEATURE_COLUMNS, FORMAT_VERSION
from grace_rul.data import OperatingRegimeScaler, WindowDataset, load_training_subset
from grace_rul.model import GraceRUL
from grace_rul.recovery_safety import fit_recovery_envelope
from grace_rul.recovery_validation import RecoveryValidationMatrix, audit_recovery_candidate
from grace_rul.training import checkpoint_guards, evaluate


SUPPORT_BUFFERS = {"recovery_centers", "recovery_lower", "recovery_upper", "recovery_regime_count"}


class SafetyUpgradeRejected(RuntimeError):
    """Validation rejected a candidate; the source remains the usable artifact."""


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint(frame):
    values = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(values + repr(list(frame.columns)).encode()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def _reference_values(rmse, nasa):
    rmse, nasa = float(rmse), float(nasa)
    if not np.isfinite([rmse, nasa]).all() or min(rmse, nasa) < 0:
        raise ValueError("Original predictor clean RMSE/NASA references must be finite and nonnegative.")
    return rmse, nasa


def original_clean_references(source_directory, subset, entry):
    """Recover the selected original predictor's references, never test metrics."""
    seed = int(entry["selected_seed"])
    path = Path(source_directory) / "checkpoints" / f"pipeline_{subset}_{seed}.pt"
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved = checkpoint.get("identity", {})
        expected = entry["pipeline_identity"]
        if not matches_pipeline_identity(saved, expected):
            raise ValueError("Original-reference checkpoint does not match the complete source training identity/configuration.")
        upstream = checkpoint.get("model")
        keys = [key for key in entry["state_dict"] if not key.startswith("corrector.") and key != "correction_bound"]
        if (not isinstance(upstream, dict) or any(key not in upstream or not isinstance(upstream[key], torch.Tensor) or
                not torch.equal(upstream[key].cpu(), entry["state_dict"][key].cpu()) for key in keys)):
            raise ValueError("Original-reference checkpoint upstream weights/state do not match the frozen source model.")
        rmse, nasa = _reference_values(checkpoint["clean_reference"], checkpoint["clean_nasa_reference"])
        return {"RMSE": rmse, "NASA_Score_Mean": nasa, "source": str(path), "sha256": _sha256(path),
                "selection": "original selected predictor references saved before recovery/correction",
                "complete_training_identity_verified": True, "upstream_state_verified": True}
    path = Path(source_directory) / "training_history.csv"
    if not path.exists():
        raise ValueError("Missing original clean references: retain the source pipeline checkpoint or training_history.csv.")
    history = pd.read_csv(path)
    required = {"subset", "seed", "stage", "checkpoint_accepted", "validation_macro_RMSE",
                "validation_clean_RMSE", "validation_clean_NASA_Score_Mean"}
    if not required.issubset(history.columns):
        raise ValueError("Source training history cannot establish the original clean RMSE/NASA references.")
    rows = history[(history.subset == subset) & (history.seed == seed) & (history.stage == "predictor") &
                   history.checkpoint_accepted.astype(str).str.lower().eq("true")]
    if rows.empty or not np.isfinite(rows[["validation_macro_RMSE", "validation_clean_RMSE", "validation_clean_NASA_Score_Mean"]].to_numpy()).all():
        raise ValueError("No finite accepted original predictor reference is available.")
    row = rows.loc[rows.validation_macro_RMSE.idxmin()]
    rmse, nasa = _reference_values(row.validation_clean_RMSE, row.validation_clean_NASA_Score_Mean)
    return {"RMSE": rmse, "NASA_Score_Mean": nasa, "source": str(path), "sha256": _sha256(path),
            "selection": "accepted predictor row with minimum validation_macro_RMSE",
            "epoch": int(row.epoch) if "epoch" in row else None}


def _training_partitions(project_root, subset, entry):
    train_path = Path(project_root) / "data" / f"train_{subset}.txt"
    expected_hash = entry.get("source_hashes", {}).get(train_path.name)
    if expected_hash is None or _sha256(train_path) != expected_hash:
        raise ValueError(f"{subset}: training source hash does not match the frozen artifact.")
    frame = load_training_subset(Path(project_root) / "data", subset)
    groups = entry["partitions"]
    if set(groups) != {"fit", "validation", "calibration"}:
        raise ValueError("Source must declare fitting, validation, and independent calibration engines.")
    for units in groups.values():
        if not units or len(units) != len(set(units)) or any(isinstance(unit, bool) or int(unit) != unit or unit < 1 for unit in units):
            raise ValueError("Source engine partitions must contain unique positive integer IDs.")
    fit_ids, val_ids, cal_ids = [set(groups[key]) for key in ["fit", "validation", "calibration"]]
    if fit_ids & val_ids or fit_ids & cal_ids or val_ids & cal_ids:
        raise ValueError("Source fitting, validation, and calibration engines overlap.")
    if not (fit_ids | val_ids | cal_ids) <= set(frame.unit):
        raise ValueError("Source engine partitions reference missing training trajectories.")
    if set(entry["scaler"]["fit_units"]) != fit_ids:
        raise ValueError("Source scaler provenance disagrees with fitting-engine partition.")
    fit, val, cal = [frame[frame.unit.isin(units)].copy() for units in [fit_ids, val_ids, cal_ids]]
    identity = entry["pipeline_identity"]
    if (identity.get("subset") != subset or identity.get("seed") != int(entry["selected_seed"]) or
            identity.get("fit") != _fingerprint(fit) or identity.get("validation") != _fingerprint(val)):
        raise ValueError("Source fitting/validation fingerprints disagree with the training data.")
    return fit, val, cal, train_path, expected_hash


def upgrade_config(weight_config, cpu_threads):
    """Change inference safety only; preserve the separately saved training law."""
    return {**deepcopy(weight_config), "device": "cpu", "cpu_threads": cpu_threads,
            "recovery_safety": "fitted_support", "recovery_validation": True,
            "recovery_validation_severities": [.1, .2, .3, .4, .5],
            "recovery_validation_max_channel_intact_rmse": weight_config.get("recovery_validation_max_channel_intact_rmse"),
            "recovery_validation_max_window_intact_rmse": weight_config.get("recovery_validation_max_window_intact_rmse"),
            "recovery_validation_max_reconstruction_rmse": weight_config.get("recovery_validation_max_reconstruction_rmse"),
            "run_extended_evaluation": False, "run_matched_benchmarks": False, "tune_before_training": False}


def migrate_frozen_model(entry, config):
    model = GraceRUL(config)
    result = model.load_state_dict(entry["state_dict"], strict=False)
    if set(result.missing_keys) != SUPPORT_BUFFERS or result.unexpected_keys:
        raise ValueError("Migration may add only the four fitting-support buffers; source weights/state are incompatible.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    assert_frozen_state(model, entry["state_dict"])
    return model


def assert_frozen_state(model, source_state):
    state = model.state_dict()
    if set(state) != set(source_state) | SUPPORT_BUFFERS:
        raise ValueError("Unexpected state keys appeared during frozen-weight migration.")
    if any(not torch.equal(state[key].cpu(), value.cpu()) for key, value in source_state.items()):
        raise ValueError("A source neural weight, detector threshold, corrector, or existing buffer changed.")


def upgrade_recovery_safety(source_path, output_dir=None, *, project_root=ROOT, cpu_threads=2):
    """Export only after complete held-validation acceptance; never weaken guards."""
    source_path = Path(source_path).resolve()
    project_root = Path(project_root).resolve()
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir).resolve() if output_dir else project_root / "artifacts" / f"safety_upgrade_v8_{stamp}"
    if output == source_path.parent or source_path.parent in output.parents:
        raise ValueError("The upgrade output must be outside the immutable source run directory.")
    if output.exists():
        raise FileExistsError("Safety upgrade requires a new output directory; existing evidence will not be overwritten.")
    before = _sha256(source_path)
    bundle, loaded = load_bundle(source_path, device="cpu")
    if bundle.get("format_version") != 7:
        raise ValueError("Frozen safety upgrade accepts only a format-7 source bundle.")
    if FORMAT_VERSION != 8:
        raise ValueError("This migration requires the format-8 fitting-support implementation.")
    del loaded
    output.mkdir(parents=True)
    torch.set_num_threads(cpu_threads)
    report = {"status": "validating", "run_type": "post_training_safety_upgrade", "weights_retrained": False,
              "source_bundle": str(source_path), "source_bundle_sha256": before, "created_at": stamp,
              "selection_data": "source fitting and held validation training engines only; separate calibration engines",
              "official_test_evaluated": False, "corrector_policy": "retain original selected corrector exactly",
              "subsets": [], "scope": "Declared simulated held-validation contexts only; no unseen/fleet performance guarantee."}
    path = output / "safety_upgrade_report.json"
    pending = []
    try:
        for subset, entry in bundle["models"].items():
            print(f"{subset}: validating frozen weights with fitting-support safety on CPU", flush=True)
            config = upgrade_config(entry.get("training_config", bundle["config"]), cpu_threads)
            fit, val, cal, train_path, train_hash = _training_partitions(project_root, subset, entry)
            reference = original_clean_references(source_path.parent, subset, entry)
            scaler = OperatingRegimeScaler(entry["scaler"])
            model = migrate_frozen_model(entry, config)
            profile = fit_recovery_envelope(fit, scaler)
            model.set_recovery_envelope(profile)
            matrix = RecoveryValidationMatrix(val, scaler, config, seed=config["validation_seed"])
            with torch.inference_mode():
                matrix_audit = audit_recovery_candidate(model, matrix, config, use_repair=True)
                clean_ds = WindowDataset(val, scaler, config, mode="validation", seed=config["validation_seed"], family="clean")
                mixed_ds = WindowDataset(val, scaler, config, mode="validation", seed=config["validation_seed"], family="mixed", severity=.2)
                clean, _ = evaluate(model, clean_ds, config, fault_thresholds=entry["thresholds"])
                mixed, _ = evaluate(model, mixed_ds, config, fault_thresholds=entry["thresholds"])
            guards = checkpoint_guards(clean, mixed, reference["RMSE"], config,
                                       clean_nasa_reference=reference["NASA_Score_Mean"])
            guards.update(matrix_complete=bool(matrix_audit["complete"]), matrix_accepted=bool(matrix_audit["accepted"]))
            accepted = all(guards.values())
            assert_frozen_state(model, entry["state_dict"])
            item = {"subset": subset, "seed": int(entry["selected_seed"]), "accepted": accepted,
                    "guards": guards, "original_clean_references": reference,
                    "clean_validation_retained_corrector": clean, "mixed_validation_retained_corrector": mixed,
                    "recovery_validation": matrix_audit, "recovery_support": profile,
                    "partition_counts": {key: len(value) for key, value in entry["partitions"].items()},
                    "training_source": str(train_path), "training_source_sha256": train_hash,
                    "source_state_unchanged": True}
            report["subsets"].append(item)
            pending.append((subset, entry, model, scaler, config, cal, item))
            print(f"{subset}: {'accepted' if accepted else 'rejected'}; guards={guards}", flush=True)
        if not all(item["accepted"] for item in report["subsets"]) or not report["subsets"]:
            report["status"] = "rejected"
            _write_json(path, report)
            raise SafetyUpgradeRejected(f"Frozen safety upgrade rejected by held-validation guards. No model exported. See {path}; retain the source artifact or train a fresh format-8 run.")
        models = {}
        for subset, entry, model, scaler, config, cal, item in pending:
            with torch.inference_mode():
                calibration, records = calibrate_endpoint_intervals(model, cal, scaler, config, config["validation_seed"] + 2)
            assert_frozen_state(model, entry["state_dict"])
            item["calibration"] = calibration
            new_entry = {key: deepcopy(value) for key, value in entry.items()
                         if key not in {"state_dict", "training_config", "calibration", "pipeline_identity", "test_metrics", "cross_validation_metrics"}}
            new_entry.update(state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                training_config=config, weight_training_config=deepcopy(entry.get("training_config", bundle["config"])),
                weight_training_pipeline_identity=deepcopy(entry["pipeline_identity"]), calibration=calibration,
                recovery_support=item["recovery_support"], recovery_validation_audit=item["recovery_validation"],
                run_type="post_training_safety_upgrade", weights_retrained=False,
                source_provenance={"bundle": str(source_path), "sha256": before, "format_version": 7,
                                   "selected_seed": int(entry["selected_seed"]), "original_clean_references": item["original_clean_references"]},
                pipeline_identity={"version": FORMAT_VERSION, "architecture": ARCHITECTURE, "subset": subset,
                                   "seed": int(entry["selected_seed"]), "run_type": "post_training_safety_upgrade",
                                   "fit": entry["pipeline_identity"]["fit"], "validation": entry["pipeline_identity"]["validation"],
                                   "config": config, "source_bundle_sha256": before})
            models[subset] = new_entry
            records.assign(subset=subset, seed=int(entry["selected_seed"])).to_csv(output / f"calibration_{subset}_{entry['selected_seed']}.csv", index=False)
        if _sha256(source_path) != before or any(_sha256(item["training_source"]) != item["training_source_sha256"] or
                _sha256(item["original_clean_references"]["source"]) != item["original_clean_references"]["sha256"] for item in report["subsets"]):
            raise ValueError("A source input changed during validation; refusing to export.")
        new_bundle = {"format_version": FORMAT_VERSION, "architecture": ARCHITECTURE,
                      "corruption_protocol": CORRUPTION_PROTOCOL, "feature_columns": FEATURE_COLUMNS,
                      "config": upgrade_config(bundle["config"], cpu_threads), "models": models,
                      "created_at": stamp, "torch_version": str(torch.__version__),
                      "run_type": "post_training_safety_upgrade", "weights_retrained": False,
                      "source_provenance": {"bundle": str(source_path), "sha256": before, "format_version": 7}}
        candidate = output / ".grace_rul_ALL.candidate.pt"
        save_checkpoint(candidate, new_bundle)
        # Validate the complete serialized v8 contract before giving this run an
        # accepted manifest or a final artifact name. No inference is needed.
        _, verified_models = load_bundle(candidate, device="cpu")
        del verified_models
        for subset, entry in models.items():
            save_checkpoint(output / "seed_models" / f"{subset}_{entry['selected_seed']}.pt", entry)
        candidate.replace(output / "grace_rul_ALL.pt")
        _write_json(output / "run_manifest.json", {"format_version": FORMAT_VERSION, "architecture": ARCHITECTURE,
            "corruption_protocol": CORRUPTION_PROTOCOL, "run_type": "post_training_safety_upgrade", "weights_retrained": False,
            "created_at": stamp, "subsets": list(models), "artifact": "grace_rul_ALL.pt",
            "quick_run": bool(bundle["config"].get("quick_run", False)),
            "source_bundle": str(source_path), "source_bundle_sha256": before,
            "selection_data": report["selection_data"], "official_test_evaluated": False,
            "evaluation_status": "post-training held-validation safety check; exploratory; official test performance not evaluated for this artifact"})
        report.update(status="accepted", source_bundle_unchanged=True, artifact="grace_rul_ALL.pt",
                      serialized_bundle_strictly_loaded=True)
        _write_json(path, report)
        print(f"Saved validated frozen-weight safety upgrade: {output / 'grace_rul_ALL.pt'}", flush=True)
        return output / "grace_rul_ALL.pt", report
    except SafetyUpgradeRejected:
        raise
    except Exception as error:
        candidate = output / ".grace_rul_ALL.candidate.pt"
        if candidate.exists():
            candidate.unlink()
        report.update(status="error", error=f"{type(error).__name__}: {error}", artifact_exported=(output / "grace_rul_ALL.pt").exists())
        _write_json(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", "--source", dest="source", required=True, type=Path,
                        help="Existing format-7 grace_rul_ALL.pt; it remains unchanged.")
    parser.add_argument("--output-dir", type=Path, default=None, help="New directory; defaults to a timestamped artifacts/safety_upgrade_v8_*.")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    try:
        upgrade_recovery_safety(args.source, args.output_dir, cpu_threads=args.cpu_threads)
    except (ValueError, FileExistsError, SafetyUpgradeRejected) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
