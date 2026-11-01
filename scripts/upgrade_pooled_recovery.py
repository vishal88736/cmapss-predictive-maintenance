#!/usr/bin/env python3
"""Upgrade a format-9 pooled model to enable verified sensor recovery and recalibrate.

This script loads the validated warm-recovery generator weights (selected during
generator training), attaches fitting-support bounding envelopes, verifies the
complete 41-context recovery matrix and declared clean guards, recalibrates on
independent calibration engines, exports a verified format-9 bundle, and evaluates
the test holdouts.
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
from grace_rul.checkpoints import save_checkpoint
from grace_rul.config import (CORRUPTION_PROTOCOL, FEATURE_COLUMNS, POOLED_ARCHITECTURE,
                             POOLED_FORMAT_VERSION, SUBSETS)
from grace_rul.data import OperatingRegimeScaler, WindowDataset, load_subset
from grace_rul.model import GraceRUL
from grace_rul.pooled import (MODEL_KEY, _evaluate_partition, evaluation_cases,
                             load_pooled_training, sha, validate_pooled_entry, write_json)
from grace_rul.recovery_safety import fit_recovery_envelope
from grace_rul.recovery_validation import RecoveryValidationMatrix, audit_recovery_candidate
from grace_rul.training import calibrate, checkpoint_guards, device_for, evaluate


class PooledUpgradeRejected(RuntimeError):
    """Validation rejected the recovery candidate; no upgraded bundle is exported."""


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def upgrade_pooled_run(project_root, source_run_dir, output_dir=None, *,
                       device="auto", skip_official=False,
                       max_clean_rmse_increase=0.03, max_clean_nasa_increase=0.10,
                       calibration_policy="observable_stratified", overwrite=False):
    root = Path(project_root)
    source_run = Path(source_run_dir).resolve()
    bundle_path = source_run / "grace_rul_ALL.pt"
    ckpt_path = source_run / "checkpoints" / f"pipeline_{MODEL_KEY}_42.pt"

    if not bundle_path.is_file():
        raise FileNotFoundError(f"Source bundle not found at {bundle_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Source pipeline checkpoint not found at {ckpt_path}")

    # Determine execution device
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu")
    print(f"Upgrade execution device: {dev}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir).resolve() if output_dir else root / "artifacts" / f"pooled_v9_r2_upgraded_{stamp}"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Upgrade output directory already exists: {output}")
    else:
        output.mkdir(parents=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    # Load source bundle and checkpoint
    bundle, loaded = load_bundle(bundle_path, device="cpu")
    entry = bundle["models"][MODEL_KEY]
    config = deepcopy(entry["training_config"])
    config["device"] = str(dev)
    config["max_clean_rmse_increase"] = float(max_clean_rmse_increase)
    config["max_clean_nasa_increase"] = float(max_clean_nasa_increase)
    if calibration_policy:
        config["calibration_policy"] = calibration_policy
    if skip_official:
        config["pooled_evaluate_official"] = False

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    warm_recovery = checkpoint.get("warm_recovery")
    if not warm_recovery:
        raise ValueError("Checkpoint does not contain warm_recovery weights.")

    # Reconstruct data partitions
    frame, hashes = load_pooled_training(root / "data", config)
    groups = entry["partitions"]
    engines = pd.DataFrame(entry["engine_manifest"])
    fit = frame[frame.unit.isin(groups["fit"])].copy()
    val = frame[frame.unit.isin(groups["validation"])].copy()
    cal = frame[frame.unit.isin(groups["calibration"])].copy()
    scaler = OperatingRegimeScaler(entry["scaler"])

    # Construct model and load weights
    model = GraceRUL(config).to(dev)
    # Start with upstream predictor and corrector from bundle
    model.load_state_dict(entry["state_dict"], strict=True)
    # Load validated warm recovery (detector and generator)
    model.load_state_dict(warm_recovery, strict=False)
    model.repair_enabled.fill_(True)

    # Attach fitting-support recovery envelope
    profile = fit_recovery_envelope(fit, scaler)
    model.set_recovery_envelope(profile)

    # Set finite repair thresholds from bundle
    thresholds = np.asarray(entry["thresholds"], dtype=np.float32)
    model.finite_repair_thresholds.copy_(torch.as_tensor(thresholds, device=dev))

    # Audit recovery across all declared held-validation conditions
    print("Auditing recovery candidate across 41 held-validation conditions...", flush=True)
    matrix = RecoveryValidationMatrix(val, scaler, config, seed=config["validation_seed"])
    with torch.inference_mode():
        audit = audit_recovery_candidate(model.to("cpu"), matrix, config, use_repair=True)
    print(f"Recovery matrix complete: {audit['complete']}, accepted: {audit['accepted']}, "
          f"worst intact RMSE: {audit['worst']['linear_intact_repair_RMSE']:.4f}", flush=True)

    if not audit["accepted"]:
        write_json(output / "recovery_audit_failure.json", audit)
        raise PooledUpgradeRejected("Recovery matrix rejected candidate. See recovery_audit_failure.json")

    # Evaluate clean and mixed validation metrics
    print("Evaluating clean and mixed validation metrics...", flush=True)
    model.to(dev)
    clean_ds = WindowDataset(val, scaler, config, mode="validation", seed=config["validation_seed"], family="clean")
    mixed_ds = WindowDataset(val, scaler, config, mode="validation", seed=config["validation_seed"], family="mixed", severity=0.2)
    clean, _ = evaluate(model, clean_ds, config, fault_thresholds=thresholds)
    mixed, _ = evaluate(model, mixed_ds, config, fault_thresholds=thresholds)

    ref_rmse = entry["pretrained_clean_RMSE"]
    ref_nasa = entry["pretrained_clean_NASA"]
    ref_source = entry["pretrained_clean_by_source"]
    guards = checkpoint_guards(clean, mixed, ref_rmse, config,
                               clean_nasa_reference=ref_nasa, clean_source_reference=ref_source)
    guards.update(matrix_complete=bool(audit["complete"]), matrix_accepted=bool(audit["accepted"]))

    print(f"Validation clean RMSE: {clean['RMSE']:.4f} vs ref {ref_rmse:.4f} (+{(clean['RMSE']/ref_rmse - 1)*100:.2f}%)")
    print(f"Validation clean NASA: {clean.get('NASA_Score_Mean', 0):.4f} vs ref {ref_nasa:.4f}")
    print(f"Guards status: {guards}", flush=True)

    if not all(guards.values()):
        report = {"status": "rejected", "guards": guards, "clean": clean, "mixed": mixed,
                  "audit": audit, "references": {"RMSE": ref_rmse, "NASA": ref_nasa, "sources": ref_source}}
        write_json(output / "upgrade_rejected.json", report)
        raise PooledUpgradeRejected(f"Clean validation guards rejected candidate. Guards: {guards}")

    # Calibrate prediction intervals on independent calibration engines
    print("Calibrating prediction intervals on independent calibration engines...", flush=True)
    calibration, calibration_records = calibrate(model, cal, scaler, config, config["validation_seed"] + 2)
    calibration_records.to_csv(output / "calibration_POOLED.csv", index=False)

    # Save detector audit and recovery validation reports
    write_json(output / "detector_audit_POOLED_42.json", entry.get("detector_audit", {}))
    write_json(output / "recovery_validation_POOLED_42.json", {"selected": audit, "candidates": [audit]})

    # Assemble and export upgraded bundle
    background_meta = entry.get("background_metadata", [])
    background = entry.get("background")
    background_valid = entry.get("background_valid")

    upgraded_entry = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "training_config": config,
        "scaler": scaler.state,
        "thresholds": entry["thresholds"],
        "calibration": calibration,
        "selected_seed": entry["selected_seed"],
        "partitions": groups,
        "engine_manifest": entry["engine_manifest"],
        "correction_bound": float(model.correction_bound),
        "repair_selected": True,
        "pretrained_clean_RMSE": ref_rmse,
        "pretrained_clean_NASA": ref_nasa,
        "pretrained_clean_by_source": ref_source,
        "pipeline_identity": entry.get("pipeline_identity", {}),
        "oof_audit": entry.get("oof_audit", []),
        "source_hashes": entry.get("source_hashes", hashes),
        "split_protocol": entry.get("split_protocol", {}),
        "detector_audit": entry.get("detector_audit", {}),
        "recovery_support": profile,
        "recovery_validation_audit": audit,
        "background_metadata": background_meta,
        "background": background,
        "background_valid": background_valid,
    }

    validate_pooled_entry(upgraded_entry, config)

    upgraded_bundle = {
        "format_version": POOLED_FORMAT_VERSION,
        "architecture": POOLED_ARCHITECTURE,
        "training_scope": "pooled",
        "requires_subset_label": False,
        "feature_columns": FEATURE_COLUMNS,
        "run_type": "post_training_recovery_upgrade",
        "weights_retrained": False,
        "corruption_protocol": CORRUPTION_PROTOCOL,
        "config": config,
        "models": {MODEL_KEY: upgraded_entry},
        "created_at": stamp,
        "torch_version": str(torch.__version__),
    }

    bundle_out_path = output / "grace_rul_ALL.pt"
    save_checkpoint(bundle_out_path, upgraded_bundle)
    load_bundle(bundle_out_path, device="cpu")
    frozen_hash = sha(bundle_out_path)
    print(f"Exported verified upgraded bundle to {bundle_out_path} (sha256: {frozen_hash})", flush=True)

    manifest = {
        "format_version": POOLED_FORMAT_VERSION,
        "architecture": POOLED_ARCHITECTURE,
        "training_scope": "pooled",
        "requires_subset_label": False,
        "quick_run": False,
        "training_recipe_revision": 2,
        "created_at": stamp,
        "source_run": str(source_run),
        "artifact": "grace_rul_ALL.pt",
        "frozen_bundle_sha256": frozen_hash,
        "status": "evaluating",
        "training_device": str(dev),
        "torch_version": str(torch.__version__),
        "repair_enabled": True,
        "clean_guards": guards,
    }
    write_json(output / "run_manifest.json", manifest)

    # Evaluate test holdouts
    print("Evaluating internal 10% test holdout...", flush=True)
    seed = int(entry["selected_seed"])
    test_rows, diagnostics = _evaluate_partition(
        model, scaler, frame[frame.unit.isin(groups["test"])],
        engines, config, calibration, thresholds, output, "internal_test_10pct", seed)

    if config["pooled_evaluate_official"]:
        print("Evaluating official benchmark endpoints...", flush=True)
        official, metadata, truths, official_hashes = [], [], {}, {}
        for index, subset in enumerate(SUBSETS, 5):
            _, test, truth = load_subset(root / "data", subset)
            for name in [f"test_{subset}.txt", f"RUL_{subset}.txt"]:
                official_hashes[name] = sha(root / "data" / name)
            test["source_unit"], test["source_subset"] = test.unit, subset
            test["unit"] = index * 1_000_000 + test.unit
            truths.update({index * 1_000_000 + int(unit): float(value) for unit, value in truth.items()})
            official.append(test)
            metadata.append(test[["unit", "source_unit", "source_subset"]].drop_duplicates())
        external_rows, external_diagnostics = _evaluate_partition(
            model, scaler, pd.concat(official), pd.concat(metadata),
            config, calibration, thresholds, output, "official_benchmark", seed, truths=truths)
        test_rows += external_rows
        diagnostics += external_diagnostics
        manifest["official_benchmark_hashes"] = official_hashes

    table = pd.DataFrame(test_rows)
    table[table.family.eq("clean")].to_csv(output / "test_metrics.csv", index=False)
    table[~table.family.eq("clean")].to_csv(output / "robustness_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "recovery_diagnostics.csv", index=False)

    manifest.update(status="complete", bundle_unchanged_during_evaluation=True,
                    official_benchmark_evaluated=config["pooled_evaluate_official"])
    write_json(output / "run_manifest.json", manifest)
    print(f"Upgrade and evaluation complete! Results saved in {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path,
                        default=ROOT / "artifacts" / "pooled_v9_r2_20261029_114845")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--max-clean-rmse-increase", type=float, default=0.03)
    parser.add_argument("--max-clean-nasa-increase", type=float, default=0.10)
    parser.add_argument("--calibration-policy", choices=["observable_stratified", "engine_prefix_max", "random_prefix"],
                        default="observable_stratified")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    upgrade_pooled_run(
        ROOT,
        args.source_run,
        args.output_dir,
        device=args.device,
        skip_official=args.skip_official,
        max_clean_rmse_increase=args.max_clean_rmse_increase,
        max_clean_nasa_increase=args.max_clean_nasa_increase,
        calibration_policy=args.calibration_policy,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
