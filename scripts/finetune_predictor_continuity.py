#!/usr/bin/env python3
"""Fine-tune GRACE-RUL predictor with physical degradation slew-rate and anti-rebound loss.

This script loads the validated upgraded model bundle, keeps the detector and
generator frozen, and fine-tunes the predictor on fitting engines with continuity,
tail-error and corruption-aware objectives. Held-selection guards decide whether
to retain a candidate; severe multi-regime prediction failures can remain.
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
from grace_rul.checkpoints import random_state, save_checkpoint
from grace_rul.config import CORRUPTION_PROTOCOL, FEATURE_COLUMNS, SUBSETS
from grace_rul.data import OperatingRegimeScaler, WindowDataset, load_subset
from grace_rul.model import GraceRUL
from grace_rul.detection import audit_deployed_thresholds, select_deployment_thresholds
from grace_rul.predictor_safety import (engine_tail_score, fault_accuracy_guards,
                                       restore_refinement_checkpoint, tail_rul_loss)
from grace_rul.pooled import (MODEL_KEY, _evaluate_partition, load_pooled_training,
                             sha, validate_pooled_entry, write_json)
from grace_rul.recovery_safety import fit_recovery_envelope
from grace_rul.recovery_validation import RecoveryValidationMatrix, audit_recovery_candidate
from grace_rul.training import (branch_objective, calibrate, checkpoint_guards,
                             device_for, evaluate, loader, move_batch,
                             prefix_consistency_loss, seed_all, validation_datasets, validation_score)


def finetune_predictor(project_root, source_bundle_path, output_dir=None, *,
                       epochs=6, lr=1e-4, slew_weight=1.0, rebound_weight=1.0,
                       device="auto", tail_weight=1., tail_fraction=.1,
                       maximum_training_windows=None, validation_only=False,
                       threshold_floor=.985, global_pooling=None, corrupted_weight=.5, resume=False):
    root = Path(project_root)
    if resume and output_dir is None:
        raise ValueError("Resume requires the original --output-dir.")
    source_path = Path(source_bundle_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source bundle not found: {source_path}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Fine-tuning needs a positive epoch count and learning rate.")
    if maximum_training_windows is not None and (isinstance(maximum_training_windows, bool) or
            not isinstance(maximum_training_windows, int) or maximum_training_windows < 1):
        raise ValueError("Maximum training windows must be a positive integer or None.")

    dev = device_for({"device": device})
    print(f"Fine-tuning execution device: {dev}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir).resolve() if output_dir else root / "artifacts" / f"pooled_v9_r2_finetuned_{stamp}"
    resume_path = output / "checkpoints/predictor_refinement.pt"
    if resume and not resume_path.is_file():
        raise FileNotFoundError(f"No interrupted refinement checkpoint at {resume_path}")
    if output.exists() and (not resume or not resume_path.is_file() or
            (output/"grace_rul_ALL.pt").exists() or (output/"validation_candidate.pt").exists()):
        raise FileExistsError(f"Fine-tuning output must be a fresh directory: {output}")
    output.mkdir(parents=True, exist_ok=resume)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    bundle, loaded = load_bundle(source_path, device="cpu")
    entry = bundle["models"][MODEL_KEY]
    source_hash = sha(source_path)
    code_paths = [Path(__file__).resolve(), *sorted((ROOT/"grace_rul").glob("*.py"))]
    implementation = {str(path.relative_to(ROOT)):sha(path) for path in code_paths}
    seed_all(int(entry["selected_seed"]))
    config = deepcopy(entry["training_config"])
    config["device"] = str(dev)
    config["physical_slew_loss_weight"] = float(slew_weight)
    config["physical_rebound_loss_weight"] = float(rebound_weight)
    config["calibration_policy"] = "observable_scaled_engine_max"
    config["finite_repair_threshold_floor"] = float(threshold_floor)
    config["finite_repair_threshold_selection"] = "minimal_source_context_FPR_with_floor"
    config["deployment_audit_revision"] = 1
    config["finetune_tail_loss_weight"] = float(tail_weight)
    config["finetune_tail_fraction"] = float(tail_fraction)
    config["finetune_maximum_windows"] = maximum_training_windows
    config["finetune_corrupted_rul_loss_weight"] = float(corrupted_weight)
    config["finetune_epochs"] = epochs
    config["finetune_learning_rate"] = float(lr)
    config["finetune_weight_decay"] = 1e-4
    config["finetune_min_learning_rate"] = min(float(lr), 1e-5)
    if global_pooling is not None:
        if global_pooling not in {"last", "mean"}:
            raise ValueError("Global pooling must be last or mean.")
        config["global_pooling"] = global_pooling
    if (not np.isfinite([tail_weight, tail_fraction, threshold_floor, corrupted_weight]).all() or tail_weight < 0 or corrupted_weight < 0 or
            not 0 < tail_fraction <= 1 or not 0 <= threshold_floor <= 1):
        raise ValueError("Invalid tail objective or finite detector threshold floor.")

    # Reconstruct data partitions
    frame, hashes = load_pooled_training(root / "data", config)
    if hashes != entry["source_hashes"]:
        raise ValueError("Source training data hashes changed.")
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
    thresholds = np.maximum(np.asarray(entry["thresholds"], dtype=np.float32), threshold_floor)
    thresholds = np.asarray(select_deployment_thresholds(model, val, scaler, config,
                            config["validation_seed"], minimum=thresholds), dtype=np.float32)
    model.finite_repair_thresholds.copy_(torch.as_tensor(thresholds, device=dev))
    model.repair_enabled.fill_(entry.get("repair_selected", True))

    # Freeze detector and generator; train only predictor
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.predictor.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                         eta_min=config["finetune_min_learning_rate"])

    clean_val, mixed_val, contexts = validation_datasets(val, scaler, config)
    stuck_val = WindowDataset(val, scaler, config, mode="validation", family="stuck", severity=.2,
                               seed=config["validation_seed"])

    # Initial baseline evaluation
    print("Evaluating initial baseline validation...", flush=True)
    ref_rmse = entry["pretrained_clean_RMSE"]
    ref_nasa = entry["pretrained_clean_NASA"]
    ref_source = entry["pretrained_clean_by_source"]
    init_clean, init_predictions = evaluate(model, clean_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
    init_mixed, _ = evaluate(model, mixed_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
    init_stuck, _ = evaluate(model, stuck_val, config, fault_thresholds=thresholds)
    source_model = loaded[MODEL_KEY]["model"].to(dev)
    source_clean, source_predictions = evaluate(source_model, clean_val, config, fault_thresholds=entry["thresholds"])
    source_mixed, _ = evaluate(source_model, mixed_val, config, fault_thresholds=entry["thresholds"])
    source_stuck, _ = evaluate(source_model, stuck_val, config, fault_thresholds=entry["thresholds"])
    source_tail = engine_tail_score(source_predictions, tail_fraction)
    fault_references = {"mixed": source_mixed, "stuck": source_stuck}
    init_guards = checkpoint_guards(init_clean, init_mixed, ref_rmse, config,
                                    use_repair=True, clean_nasa_reference=ref_nasa, clean_source_reference=ref_source)
    init_guards.update(fault_accuracy_guards({"mixed": init_mixed, "stuck": init_stuck},
                       fault_references, config["max_clean_rmse_increase"]))
    init_guards["engine_tail"] = engine_tail_score(init_predictions, tail_fraction) <= source_tail
    print(f"Initial Active Clean Validation: RMSE={init_clean['RMSE']:.4f} (ref: {ref_rmse:.4f}), NASA={init_clean.get('NASA_Score_Mean', 0.):.4f} (ref: {ref_nasa:.4f})", flush=True)
    print(f"Initial Guards: {init_guards}", flush=True)

    # Training dataset with paired prefixes
    # Keep all fitting windows eligible. A pilot budget limits draws per epoch,
    # rather than permanently discarding most of the fitting trajectories.
    training_config = {**config, "maximum_windows": None}
    train_ds = WindowDataset(fit, scaler, training_config, mode="train", seed=config["seed"],
                             family="mixed" if corrupted_weight else "clean")
    train_loader = loader(train_ds, config, shuffle=True, seed=config["seed"])

    best_state = deepcopy(model.predictor.state_dict())
    best_rmse = init_clean["RMSE"] if all(init_guards.values()) else float("inf")
    best_tail = engine_tail_score(init_predictions, tail_fraction) if all(init_guards.values()) else float("inf")
    best_epoch, history = 0, []
    best_reports = {"clean": init_clean, "mixed": init_mixed, "stuck": init_stuck, "guards": init_guards,
                    "engine_tail_score": best_tail}
    start_epoch = 1
    if resume:
        saved = restore_refinement_checkpoint(resume_path, model, optimizer, scheduler,
            train_loader.generator, config=config, source_hash=source_hash, implementation=implementation)
        best_state, best_rmse, best_tail = saved["best_predictor"], saved["best_rmse"], saved["best_tail"]
        best_epoch, best_reports, history = saved["best_epoch"], saved["best_reports"], saved["history"]
        start_epoch = int(saved["epoch"])+1
        print(f"Resuming after completed epoch {saved['epoch']}.",flush=True)

    print(f"Starting {epochs} epochs of physical continuity fine-tuning...", flush=True)
    scale = config["target_scale"]

    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_ds.set_epoch(epoch)
        total_loss, total_rul, total_prefix = 0., 0., 0.
        nbatches = 0

        for step, batch in enumerate(train_loader):
            if maximum_training_windows is not None and step * config["batch_size"] >= maximum_training_windows:
                break
            batch = move_batch(batch, dev)
            optimizer.zero_grad(set_to_none=True)

            result = model(batch["reference"], batch["valid"], use_repair=False, use_correction=False)
            rul_loss = config["rul_loss_weight"] * F.mse_loss(result["base"] / scale, batch["target"] / scale)
            branch_loss = config["branch_loss_weight"] * branch_objective(model, result, batch["target"])
            # Both sides of the pair use the same deterministic mode. Passing
            # the dropout-enabled result here manufactured apparent rebounds.
            prefix_loss = config["prefix_consistency_loss_weight"] * prefix_consistency_loss(model, batch)
            tail_loss = tail_weight * tail_rul_loss(result["base"], batch["target"],
                                                   scale=scale, fraction=tail_fraction)
            corrupted_loss = result["base"].sum() * 0.
            if corrupted_weight:
                active = model(batch["x"], batch["valid"], use_repair=True, use_correction=True)
                corrupted_loss = corrupted_weight * F.mse_loss(active["prediction"] / scale, batch["target"] / scale)

            loss = rul_loss + branch_loss + prefix_loss + tail_loss + corrupted_loss
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
        clean_rep, clean_predictions = evaluate(model, clean_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
        mixed_rep, _ = evaluate(model, mixed_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
        stuck_rep, _ = evaluate(model, stuck_val, config, fault_thresholds=thresholds)
        guards = checkpoint_guards(clean_rep, mixed_rep, ref_rmse, config,
                                   use_repair=True, clean_nasa_reference=ref_nasa, clean_source_reference=ref_source)
        guards.update(fault_accuracy_guards({"mixed": mixed_rep, "stuck": stuck_rep},
                      fault_references, config["max_clean_rmse_increase"]))
        guards["engine_tail"] = engine_tail_score(clean_predictions, tail_fraction) <= source_tail
        all_passed = all(guards.values())
        curr_rmse = clean_rep["RMSE"]
        curr_nasa = clean_rep.get("NASA_Score_Mean", 0.)
        curr_tail = engine_tail_score(clean_predictions, tail_fraction)
        history.append({"epoch": epoch, "clean_RMSE": curr_rmse, "clean_NASA": curr_nasa,
                        "mixed_RMSE": mixed_rep["RMSE"], "engine_tail_score": curr_tail,
                        "accepted": bool(all_passed and (curr_tail, curr_rmse) < (best_tail, best_rmse)),
                        **{f"guard_{key}": bool(value) for key, value in guards.items()}})

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s): loss={total_loss/nbatches:.4f} "
              f"(rul={total_rul/nbatches:.4f}, prefix={total_prefix/nbatches:.4f}) | "
              f"Active Clean RMSE={curr_rmse:.4f} (ref: {ref_rmse:.4f}), NASA={curr_nasa:.4f} | guards={'PASSED' if all_passed else 'FAILED: ' + str([k for k, v in guards.items() if not v])}", flush=True)

        if all_passed and (curr_tail, curr_rmse) < (best_tail, best_rmse):
            best_rmse = curr_rmse
            best_tail, best_epoch = curr_tail, epoch
            best_state = deepcopy(model.predictor.state_dict())
            best_reports = {"clean": clean_rep, "mixed": mixed_rep, "stuck": stuck_rep, "guards": guards,
                            "engine_tail_score": curr_tail}
            print(f"  -> New best predictor weights saved (RMSE: {best_rmse:.4f}, all guards passed)", flush=True)
        save_checkpoint(resume_path, {"stage":"predictor_refinement", "epoch":epoch,
            "source_bundle_sha256":source_hash, "implementation_sha256":implementation, "config":config,
            "state_dict":{key:value.detach().cpu() for key,value in model.state_dict().items()},
            "optimizer":optimizer.state_dict(), "scheduler":scheduler.state_dict(), "rng":random_state(),
            "loader_rng":train_loader.generator.get_state(), "best_predictor":best_state,
            "best_rmse":best_rmse, "best_tail":best_tail, "best_epoch":best_epoch,
            "best_reports":best_reports, "history":history})

    # Load best predictor weights
    model.predictor.load_state_dict(best_state)
    model.eval()

    # Verify all active path guards hold
    final_clean, _ = evaluate(model, clean_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
    final_mixed, _ = evaluate(model, mixed_val, config, fault_thresholds=thresholds, use_repair=True, use_correction=True)
    final_stuck, _ = evaluate(model, stuck_val, config, fault_thresholds=thresholds)
    final_guards = checkpoint_guards(final_clean, final_mixed, ref_rmse, config,
                                     use_repair=True, clean_nasa_reference=ref_nasa, clean_source_reference=ref_source)
    final_guards.update(fault_accuracy_guards({"mixed": final_mixed, "stuck": final_stuck},
                        fault_references, config["max_clean_rmse_increase"]))
    final_guards["engine_tail"] = best_tail <= source_tail
    if not all(final_guards.values()):
        failed = [k for k, v in final_guards.items() if not v]
        pd.DataFrame(history).to_csv(output / "validation_history.csv", index=False)
        write_json(output / "validation_rejected.json", {"status": "rejected", "guards": final_guards,
                   "selected_epoch": best_epoch, "source_bundle_sha256": source_hash,
                   "selection_uses_test_metrics": False, "deployable_exported": False})
        raise RuntimeError(f"Active path validation guards failed: {failed}")
    print(f"Final active path validation confirmed all {len(final_guards)} guards passed.", flush=True)

    pd.DataFrame(history).to_csv(output / "validation_history.csv", index=False)
    selection = {"source_bundle": str(source_path), "source_bundle_sha256": source_hash,
                 "selected_epoch": best_epoch, "changed_predictor": best_epoch > 0,
                 "selection_uses_test_metrics": False,
                 "selection_scope": "Held validation engines; tail of engine maximum-prefix errors with all declared guards",
                 "source_reference": {"clean": source_clean, "mixed": source_mixed, "stuck": source_stuck,
                                      "engine_tail_score": source_tail},
                 "initial": {"clean": init_clean, "mixed": init_mixed, "stuck": init_stuck,
                             "engine_tail_score": engine_tail_score(init_predictions, tail_fraction)},
                 "selected": best_reports, "final_guards": final_guards}
    write_json(output / "validation_selection.json", selection)
    if validation_only:
        save_checkpoint(output / "validation_candidate.pt", {
            "deployable": False, "status": "validation_only", "config": config,
            "source_bundle": str(source_path), "source_bundle_sha256": source_hash,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "selection": selection, "source_hashes": hashes, "partitions": groups})
        if sha(source_path) != source_hash:
            raise RuntimeError("Source artifact changed during validation-only study.")
        print(f"Validation-only candidate saved; selected epoch {best_epoch}. No calibration or test data evaluated.", flush=True)
        return output

    # Recalibrate prediction intervals with observable stratified policy
    print("Recalibrating intervals with observable stratified policy...", flush=True)
    calibration, calibration_records = calibrate(model, cal, scaler, config,
                                                config["validation_seed"] + 2, scale_frame=val)
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
    upgraded_entry["detector_audit"] = audit_deployed_thresholds(model, val, scaler, config,
        config["validation_seed"], previous_audit=entry.get("detector_audit"))
    write_json(output / "detector_audit_POOLED_42.json", upgraded_entry["detector_audit"])

    validate_pooled_entry(upgraded_entry, config)

    upgraded_bundle = {
        "format_version": bundle["format_version"],
        "architecture": bundle["architecture"],
        "training_scope": "pooled",
        "requires_subset_label": False,
        "feature_columns": FEATURE_COLUMNS,
        "run_type": "predictor_physical_slew_finetuned",
        "weights_retrained": best_epoch > 0,
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
        "format_version": bundle["format_version"],
        "architecture": bundle["architecture"],
        "training_scope": "pooled",
        "requires_subset_label": False,
        "run_type": "predictor_physical_slew_finetuned",
        "created_at": stamp,
        "source_bundle": str(source_path),
        "source_bundle_sha256": source_hash,
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
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--tail-fraction", type=float, default=.1)
    parser.add_argument("--corrupted-weight", type=float, default=.5)
    parser.add_argument("--maximum-training-windows", type=int)
    parser.add_argument("--threshold-floor", type=float, default=.985)
    parser.add_argument("--global-pooling", choices=["last", "mean"])
    parser.add_argument("--validation-only", action="store_true",
                        help="Save a non-deployable selection candidate; do not calibrate or evaluate holdouts.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted matching run in --output-dir.")
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
        tail_weight=args.tail_weight, tail_fraction=args.tail_fraction,
        maximum_training_windows=args.maximum_training_windows,
        validation_only=args.validation_only, threshold_floor=args.threshold_floor,
        global_pooling=args.global_pooling,
        corrupted_weight=args.corrupted_weight,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
