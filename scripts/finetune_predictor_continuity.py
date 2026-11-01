#!/usr/bin/env python3
"""Fine-tune GRACE-RUL predictor with physical degradation slew-rate and anti-rebound loss.

This script loads the validated upgraded model bundle, keeps the detector and
generator frozen, and fine-tunes the predictor network on fitting engines with
physical continuity regularization to eliminate premature degradation collapse
in complex multi-regime environments (FD004).
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from grace_rul.artifacts import load_bundle
from grace_rul.checkpoints import save_checkpoint
from grace_rul.config import (CORRUPTION_PROTOCOL, FEATURE_COLUMNS, POOLED_ARCHITECTURE,
                             POOLED_FORMAT_VERSION, SUBSETS)
from grace_rul.data import OperatingRegimeScaler, WindowDataset, load_subset
from grace_rul.model import GraceRUL
from grace_rul.pooled import (MODEL_KEY, _evaluate_partition, load_pooled_training,
                             sha, validate_pooled_entry, write_json)
from grace_rul.recovery_safety import fit_recovery_envelope
from grace_rul.recovery_validation import RecoveryValidationMatrix, audit_recovery_candidate
from grace_rul.training import (branch_objective, calibrate, checkpoint_guards,
                             device_for, evaluate, loader, move_batch,
                             prefix_consistency_loss, validation_datasets, validation_score)


def finetune_predictor(project_root, source_bundle_path, output_dir=None, *,
                       epochs=6, lr=1e-4, slew_weight=1.0, rebound_weight=1.0,
                       device="auto"):
    root = Path(project_root)
    source_path = Path(source_bundle_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source bundle not found: {source_path}")

    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu")
    print(f"Fine-tuning execution device: {dev}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir).resolve() if output_dir else root / "artifacts" / f"pooled_v9_r2_finetuned_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    bundle, loaded = load_bundle(source_path, device="cpu")
    entry = bundle["models"][MODEL_KEY]
    config = deepcopy(entry["training_config"])
    config["device"] = str(dev)
    config["physical_slew_loss_weight"] = float(slew_weight)
    config["physical_rebound_loss_weight"] = float(rebound_weight)
    config["calibration_policy"] = "observable_stratified"

    # Reconstruct data partitions
    frame, hashes = load_pooled_training(root / "data", config)
    groups = entry["partitions"]
    engines = pd.DataFrame(entry["engine_manifest"])
    fit = frame[frame.unit.isin(groups["fit"])].copy()
    val = frame[frame.unit.isin(groups["validation"])].copy()
    cal = frame[frame.unit.isin(groups["calibration"])].copy()
    scaler = OperatingRegimeScaler(entry["scaler"])

    # Instantiate model and load state
    model = GraceRUL(config).to(dev)
    model.load_state_dict(entry["state_dict"], strict=True)
    profile = fit_recovery_envelope(fit, scaler)
    model.set_recovery_envelope(profile)
    thresholds = np.maximum(np.asarray(entry["thresholds"], dtype=np.float32), 0.985)
    model.finite_repair_thresholds.copy_(torch.as_tensor(thresholds, device=dev))
    model.repair_enabled.fill_(entry.get("repair_selected", True))

    # Freeze detector and generator; train only predictor
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.predictor.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    clean_val, mixed_val, contexts = validation_datasets(val, scaler, config)

    # Initial baseline evaluation
    print("Evaluating initial baseline validation...", flush=True)
    ref_rmse = entry["pretrained_clean_RMSE"]
    ref_nasa = entry["pretrained_clean_NASA"]
    ref_source = entry["pretrained_clean_by_source"]
    init_clean, _ = evaluate(model, clean_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
    init_mixed, _ = evaluate(model, mixed_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
    init_guards = checkpoint_guards(init_clean, init_mixed, ref_rmse, config,
                                    use_repair=True, clean_nasa_reference=ref_nasa, clean_source_reference=ref_source)
    print(f"Initial Active Clean Validation: RMSE={init_clean['RMSE']:.4f} (ref: {ref_rmse:.4f}), NASA={init_clean.get('NASA_Score_Mean', 0.):.4f} (ref: {ref_nasa:.4f})", flush=True)
    print(f"Initial Guards: {init_guards}", flush=True)

    # Training dataset with paired prefixes
    train_ds = WindowDataset(fit, scaler, config, mode="train", seed=config["seed"], family="clean")
    train_loader = loader(train_ds, config, shuffle=True, seed=config["seed"])

    best_state = deepcopy(model.predictor.state_dict())
    best_rmse = init_clean["RMSE"] if all(init_guards.values()) else float("inf")

    print(f"Starting {epochs} epochs of physical continuity fine-tuning...", flush=True)
    scale = config["target_scale"]

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_ds.set_epoch(epoch)
        total_loss, total_rul, total_prefix = 0., 0., 0.
        nbatches = 0

        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, dev)
            optimizer.zero_grad(set_to_none=True)

            result = model(batch["reference"], batch["valid"], use_repair=False, use_correction=False)
            rul_loss = config["rul_loss_weight"] * F.mse_loss(result["base"] / scale, batch["target"] / scale)
            branch_loss = config["branch_loss_weight"] * branch_objective(model, result, batch["target"])
            prefix_loss = config["prefix_consistency_loss_weight"] * prefix_consistency_loss(model, batch, current_prediction=result["base"])

            loss = rul_loss + branch_loss + prefix_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach())
            total_rul += float(rul_loss.detach())
            total_prefix += float(prefix_loss.detach())
            nbatches += 1

        scheduler.step()
        elapsed = time.perf_counter() - started

        # Validate active path performance (with repair and correction enabled)
        clean_rep, _ = evaluate(model, clean_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
        mixed_rep, _ = evaluate(model, mixed_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
        guards = checkpoint_guards(clean_rep, mixed_rep, ref_rmse, config,
                                   use_repair=True, clean_nasa_reference=ref_nasa, clean_source_reference=ref_source)
        all_passed = all(guards.values())
        curr_rmse = clean_rep["RMSE"]
        curr_nasa = clean_rep.get("NASA_Score_Mean", 0.)

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s): loss={total_loss/nbatches:.4f} "
              f"(rul={total_rul/nbatches:.4f}, prefix={total_prefix/nbatches:.4f}) | "
              f"Active Clean RMSE={curr_rmse:.4f} (ref: {ref_rmse:.4f}), NASA={curr_nasa:.4f} | guards={'PASSED' if all_passed else 'FAILED: ' + str([k for k, v in guards.items() if not v])}", flush=True)

        if all_passed and curr_rmse < best_rmse:
            best_rmse = curr_rmse
            best_state = deepcopy(model.predictor.state_dict())
            print(f"  -> New best predictor weights saved (RMSE: {best_rmse:.4f}, all guards passed)", flush=True)

    # Load best predictor weights
    model.predictor.load_state_dict(best_state)
    model.eval()

    # Verify all active path guards hold
    final_clean, _ = evaluate(model, clean_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
    final_mixed, _ = evaluate(model, mixed_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
    final_guards = checkpoint_guards(final_clean, final_mixed, ref_rmse, config,
                                     use_repair=True, clean_nasa_reference=ref_nasa, clean_source_reference=ref_source)
    if not all(final_guards.values()):
        failed = [k for k, v in final_guards.items() if not v]
        raise RuntimeError(f"Active path validation guards failed: {failed}")
    print(f"Final active path validation confirmed all {len(final_guards)} guards passed.", flush=True)

    # Recalibrate prediction intervals with observable stratified policy
    print("Recalibrating intervals with observable stratified policy...", flush=True)
    calibration, calibration_records = calibrate(model, cal, scaler, config, config["validation_seed"] + 2)
    calibration_records.to_csv(output / "calibration_POOLED.csv", index=False)

    # Verify recovery matrix holds
    matrix = RecoveryValidationMatrix(val, scaler, config, seed=config["validation_seed"])
    audit = audit_recovery_candidate(model, matrix, config, use_repair=True, stop_on_failure=True)
    print(f"Recovery matrix verification: complete={audit['complete']}, accepted={audit['accepted']}, worst intact={audit['worst'].get('linear_intact_repair_RMSE'):.4f}", flush=True)

    upgraded_entry = deepcopy(entry)
    upgraded_entry["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    upgraded_entry["training_config"] = deepcopy(config)
    upgraded_entry["thresholds"] = thresholds.tolist()
    upgraded_entry["calibration"] = calibration
    upgraded_entry["recovery_validation_audit"] = audit

    validate_pooled_entry(upgraded_entry, config)

    upgraded_bundle = {
        "format_version": POOLED_FORMAT_VERSION,
        "architecture": POOLED_ARCHITECTURE,
        "training_scope": "pooled",
        "requires_subset_label": False,
        "feature_columns": FEATURE_COLUMNS,
        "run_type": "predictor_physical_slew_finetuned",
        "weights_retrained": True,
        "corruption_protocol": CORRUPTION_PROTOCOL,
        "config": config,
        "models": {MODEL_KEY: upgraded_entry},
        "created_at": stamp,
        "torch_version": str(torch.__version__),
    }

    bundle_out_path = output / "grace_rul_ALL.pt"
    save_checkpoint(bundle_out_path, upgraded_bundle)
    frozen_hash = sha(bundle_out_path)
    print(f"Exported fine-tuned bundle to {bundle_out_path} (sha256: {frozen_hash})", flush=True)

    # Verify strict loader immediately
    load_bundle(bundle_out_path, device="cpu")
    print(f"Verified strict load of {bundle_out_path} passed successfully.", flush=True)

    # Evaluate internal 10% test holdout
    print("Evaluating internal 10% test holdout...", flush=True)
    seed = int(entry["selected_seed"])
    test_rows, diagnostics = _evaluate_partition(
        model, scaler, frame[frame.unit.isin(groups["test"])],
        engines, config, calibration, thresholds, output, "internal_test_10pct", seed)

    # Evaluate official benchmark endpoints
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

    table = pd.DataFrame(test_rows)
    table[table.family.eq("clean")].to_csv(output / "test_metrics.csv", index=False)
    table[~table.family.eq("clean")].to_csv(output / "robustness_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "recovery_diagnostics.csv", index=False)

    manifest = {
        "format_version": POOLED_FORMAT_VERSION,
        "architecture": POOLED_ARCHITECTURE,
        "training_scope": "pooled",
        "requires_subset_label": False,
        "run_type": "predictor_physical_slew_finetuned",
        "created_at": stamp,
        "source_bundle": str(source_path),
        "artifact": "grace_rul_ALL.pt",
        "frozen_bundle_sha256": frozen_hash,
        "status": "complete",
        "training_device": str(dev),
        "torch_version": str(torch.__version__),
        "repair_enabled": True,
        "best_val_clean_rmse": float(best_rmse),
    }
    write_json(output / "run_manifest.json", manifest)
    print(f"Fine-tuning and evaluation complete! Results saved in {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path,
                        default=ROOT / "artifacts" / "pooled_v9_r2_upgraded_20261101" / "grace_rul_ALL.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--slew-weight", type=float, default=1.0)
    parser.add_argument("--rebound-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    finetune_predictor(
        ROOT,
        args.source_bundle,
        args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        slew_weight=args.slew_weight,
        rebound_weight=args.rebound_weight,
        device=args.device,
    )


if __name__ == "__main__":
    main()
