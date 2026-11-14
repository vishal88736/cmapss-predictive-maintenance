#!/usr/bin/env python3
"""Build the format-10 repair from a frozen format-9 pooled bundle.

The repair is deliberately narrower than a full retrain:

* the predictor gains a zero-initialised engine-age channel;
* continuity is supervised on the deployed (repair + soft-gate) path;
* a branch-agreement term discourages local/global divergence;
* the inert corrector is disabled;
* the detector gate becomes probability-weighted instead of a hard threshold;
* a monotone plateau stretch and fresh conformal intervals are fitted on held
  validation/calibration engines only.

Detector, generator, recovery support and RUL data splits are inherited from
the source bundle. Official test data is read only after the candidate and its
calibration are frozen.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
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
from grace_rul.config import (CORRUPTION_PROTOCOL, FEATURE_COLUMNS, POOLED_REPAIR_ARCHITECTURE,
                              POOLED_REPAIR_FORMAT_VERSION, SUBSETS)
from grace_rul.data import WindowDataset, load_subset
from grace_rul.detection import (audit_deployed_thresholds, refine_bypassed_detector,
                                  select_deployment_thresholds)
from grace_rul.model import GraceRUL, age_from_cycles, expand_predictor_for_age
from grace_rul.pooled import (MODEL_KEY, _annotate, _point_rows, load_pooled_training,
                              sha, validate_pooled_entry, write_json)
from grace_rul.predictor_safety import engine_tail_score, tail_rul_loss
from grace_rul.repair import (apply_plateau_calibration, engine_bootstrap, fit_plateau_calibration,
                              interval_band_report, long_rul_report, per_engine_tail)
from grace_rul.recovery_safety import fit_recovery_envelope
from grace_rul.recovery_validation import RecoveryValidationMatrix, audit_recovery_candidate
from grace_rul.training import (branch_objective, calibrate, device_for, evaluate, interval_bounds,
                                loader, move_batch, prefix_consistency_loss, seed_all)


def _json_safe(value):
    """Convert numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _dataset(frame, scaler, config, mode, family="clean", severity=.0, seed=None, truths=None):
    return WindowDataset(frame, scaler, config, mode=mode,
                         seed=config["validation_seed"] if seed is None else seed,
                         family=family, severity=severity, truths=truths)


def _evaluate(model, dataset, config, thresholds):
    return evaluate(model, dataset, config, fault_thresholds=thresholds,
                    use_repair=True, use_correction=False)


def _score(report, tail):
    return (0.5 * report["RMSE"] + 0.25 * report.get("mixed_RMSE", report["RMSE"]) +
            0.25 * report.get("stuck_RMSE", report["RMSE"]) + 0.2 * tail)


def finetune(project_root, source_bundle_path, output_dir, *, epochs=12, lr=2e-4,
             device="auto", soft_temperature=1.0, soft_power=3.0, branch_weight=0.1,
             prefix_weight=1.0, active_weight=1.0, clean_weight=1.0,
             tail_weight=0.1, max_windows=None, final_evaluation=True, seed_override=None):
    root = Path(project_root)
    source = Path(source_bundle_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()

    source_bundle, loaded = load_bundle(source, device="cpu")
    entry = source_bundle["models"][MODEL_KEY]
    config = deepcopy(entry["training_config"])
    dev = device_for({"device": device})
    config.update({
        "device": str(dev), "age_feature": True, "age_normalizer": 500.0,
        "finite_repair_gate": "soft", "soft_gate_temperature": float(soft_temperature),
        "soft_gate_power": float(soft_power),
        "soft_gate_respect_disabled": True, "use_corrector": False,
        "prefix_consistency_path": "active", "prefix_consistency_loss_weight": float(prefix_weight),
        "branch_agreement_loss_weight": float(branch_weight),
        "physical_slew_loss_weight": 0.5, "physical_rebound_loss_weight": 0.5,
        "train_repair": True, "run_extended_evaluation": False,
        "pooled_evaluate_official": False, "calibration_policy": "observable_scaled_engine_max",
        "deployment_audit_revision": 1,
    })
    seed = int(entry["selected_seed"]) if seed_override is None else int(seed_override)
    seed_all(seed, config["cpu_threads"])
    config["seed"] = seed

    frame, hashes = load_pooled_training(root / "data", config)
    if hashes != entry["source_hashes"]:
        raise ValueError("Training sources changed since the frozen bundle.")
    groups = entry["partitions"]
    fit = frame[frame.unit.isin(groups["fit"])].copy()
    val = frame[frame.unit.isin(groups["validation"])].copy()
    cal = frame[frame.unit.isin(groups["calibration"])].copy()
    engines = pd.DataFrame(entry["engine_manifest"])
    scaler = loaded[MODEL_KEY]["scaler"]

    model = GraceRUL(config).to(dev)
    model.load_state_dict(expand_predictor_for_age(entry["state_dict"], config), strict=True)
    profile = entry.get("recovery_support") or fit_recovery_envelope(fit, scaler)
    model.set_recovery_envelope(profile)
    model.finite_repair_thresholds.copy_(torch.as_tensor(entry["thresholds"], device=dev, dtype=torch.float32))
    model.repair_enabled.fill_(bool(entry.get("repair_selected", True)))
    model.correction_bound.fill_(0.)
    model.config["plateau_calibration"] = None
    model.eval()

    # Refine the bypassed detector under the strict 5% false-positive rate guard,
    # boosting recall across drift, bias, noise, stuck, and mixed families.
    print("Refining detector on fitting and validation engines...", flush=True)
    model.repair_enabled.fill_(False)
    detector_config = deepcopy(config)
    detector_config.update({
        "detector_refinement_epochs": 4,
        "detector_refinement_patience": 2,
        "maximum_batches": 25,
    })
    detector_history = refine_bypassed_detector(model, fit, val, scaler, detector_config, seed)
    model.repair_enabled.fill_(bool(entry.get("repair_selected", True)))
    initial_th = detector_history.attrs["audit"]["thresholds"]
    model.finite_repair_thresholds.zero_()
    deployed_thresholds = select_deployment_thresholds(
        model, val, scaler, config, int(config["validation_seed"]), minimum=initial_th
    )
    model.finite_repair_thresholds.copy_(torch.as_tensor(deployed_thresholds, device=dev, dtype=torch.float32))
    print(f"Deployed detector thresholds refined: min={min(deployed_thresholds):.3f} max={max(deployed_thresholds):.3f} "
          f"active={(np.array(deployed_thresholds) < 1.0).sum()}/21 channels", flush=True)
    post_audit = audit_deployed_thresholds(model, val, scaler, config, int(config["validation_seed"]))
    for fam in ["drift", "bias", "noise", "stuck", "mixed"]:
        m = post_audit["selected"]["contexts"].get(fam, {})
        print(f"  {fam} recall={m.get('recall', 0.):.3f} F1={m.get('F1', 0.):.3f}", flush=True)

    # Baseline is the unmodified source model. The candidate must improve or
    # preserve every declared validation slice.
    source_model = loaded[MODEL_KEY]["model"].to(dev)
    clean_val = _dataset(val, scaler, config, "validation", "clean")
    mixed_val = _dataset(val, scaler, config, "validation", "mixed", .2)
    stuck_val = _dataset(val, scaler, config, "validation", "stuck", .2)
    ref_clean, ref_clean_records = _evaluate(source_model, clean_val, entry["training_config"], entry["thresholds"])
    ref_mixed, _ = _evaluate(source_model, mixed_val, entry["training_config"], entry["thresholds"])
    ref_stuck, _ = _evaluate(source_model, stuck_val, entry["training_config"], entry["thresholds"])
    ref_tail = engine_tail_score(ref_clean_records)
    print(f"Source validation clean={ref_clean['RMSE']:.4f} mixed={ref_mixed['RMSE']:.4f} "
          f"stuck={ref_stuck['RMSE']:.4f} tail={ref_tail:.4f}", flush=True)

    init_clean, init_records = _evaluate(model, clean_val, config, deployed_thresholds)
    init_mixed, _ = _evaluate(model, mixed_val, config, deployed_thresholds)
    init_stuck, _ = _evaluate(model, stuck_val, config, deployed_thresholds)
    init_tail = engine_tail_score(init_records)
    print(f"Initial v10 validation clean={init_clean['RMSE']:.4f} mixed={init_mixed['RMSE']:.4f} "
          f"stuck={init_stuck['RMSE']:.4f} tail={init_tail:.4f}", flush=True)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.predictor.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min(lr, 1e-5))

    train_config = {**config, "maximum_windows": max_windows}
    train_ds = WindowDataset(fit, scaler, train_config, mode="train", seed=config["seed"], family=None)
    train_loader = loader(train_ds, train_config, shuffle=True, seed=config["seed"])

    best_state = deepcopy(model.predictor.state_dict())
    best = {"score": _score({"RMSE": init_clean["RMSE"], "mixed_RMSE": init_mixed["RMSE"],
                             "stuck_RMSE": init_stuck["RMSE"]}, init_tail),
            "clean": init_clean, "mixed": init_mixed, "stuck": init_stuck,
            "tail": init_tail, "epoch": 0}
    history = []
    scale = config["target_scale"]

    print(f"Fine-tuning predictor for {epochs} epochs on {dev}...", flush=True)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        model.detector.eval()
        model.generator.eval()
        train_ds.set_epoch(epoch)
        totals = {"loss": 0., "clean": 0., "active": 0., "prefix": 0., "branch": 0., "tail": 0.}
        batches = 0
        for step, batch in enumerate(train_loader):
            if max_windows is not None and step * config["batch_size"] >= max_windows:
                break
            batch = move_batch(batch, dev)
            age = age_from_cycles(batch.get("cycle"), config)
            optimizer.zero_grad(set_to_none=True)
            clean = model(batch["reference"], batch["valid"], age=age, use_repair=False)
            active = model(batch["x"], batch["valid"], age=age, use_repair=True)
            clean_loss = F.mse_loss(clean["base"] / scale, batch["target"] / scale)
            active_loss = F.mse_loss(active["prediction"] / scale, batch["target"] / scale)
            prefix_loss = prefix_consistency_loss(model, batch, active["base"])
            branch_loss = branch_objective(model, active, batch["target"])
            agreement = F.mse_loss(active["local_prediction"] / scale, active["global_prediction"] / scale)
            tail_loss = 0.5 * (tail_rul_loss(active["prediction"], batch["target"], scale=scale, fraction=.1) +
                               tail_rul_loss(clean["prediction"], batch["target"], scale=scale, fraction=.1))
            loss = (clean_weight * clean_loss + active_weight * active_loss +
                    prefix_weight * prefix_loss + 0.1 * branch_loss +
                    branch_weight * agreement + tail_weight * tail_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach()); totals["clean"] += float(clean_loss.detach())
            totals["active"] += float(active_loss.detach()); totals["prefix"] += float(prefix_loss.detach())
            totals["branch"] += float(agreement.detach()); totals["tail"] += float(tail_loss.detach())
            batches += 1
        scheduler.step()
        clean_rep, clean_records = _evaluate(model, clean_val, config, deployed_thresholds)
        mixed_rep, _ = _evaluate(model, mixed_val, config, deployed_thresholds)
        stuck_rep, _ = _evaluate(model, stuck_val, config, deployed_thresholds)
        tail = engine_tail_score(clean_records)
        guards = {
            "clean_rmse": clean_rep["RMSE"] <= ref_clean["RMSE"] * 1.01,
            "clean_nasa": clean_rep.get("NASA_Score_Mean", float("inf")) <= ref_clean.get("NASA_Score_Mean", float("inf")) * 1.05,
            "mixed_rmse": mixed_rep["RMSE"] <= ref_mixed["RMSE"] * 1.02,
            "stuck_rmse": stuck_rep["RMSE"] <= ref_stuck["RMSE"] * 1.02,
            "engine_tail": tail <= ref_tail * 1.02,
        }
        score = _score({"RMSE": clean_rep["RMSE"], "mixed_RMSE": mixed_rep["RMSE"],
                        "stuck_RMSE": stuck_rep["RMSE"]}, tail)
        accepted = all(guards.values()) and score < best["score"]
        history.append({"epoch": epoch, "seconds": time.perf_counter() - started,
                        "loss": totals["loss"] / max(batches, 1),
                        "clean_RMSE": clean_rep["RMSE"], "clean_NASA": clean_rep.get("NASA_Score_Mean"),
                        "mixed_RMSE": mixed_rep["RMSE"], "stuck_RMSE": stuck_rep["RMSE"],
                        "engine_tail": tail, "score": score, "accepted": bool(accepted),
                        **{f"guard_{key}": bool(value) for key, value in guards.items()}})
        if accepted:
            best = {"score": score, "clean": clean_rep, "mixed": mixed_rep, "stuck": stuck_rep,
                    "tail": tail, "epoch": epoch}
            best_state = deepcopy(model.predictor.state_dict())
        print(f"Epoch {epoch}/{epochs} ({history[-1]['seconds']:.1f}s): "
              f"clean={clean_rep['RMSE']:.4f} mixed={mixed_rep['RMSE']:.4f} "
              f"stuck={stuck_rep['RMSE']:.4f} tail={tail:.4f} "
              f"{'ACCEPTED' if accepted else 'not selected'}", flush=True)

    model.predictor.load_state_dict(best_state)
    model.eval()
    pd.DataFrame(history).to_csv(output / "validation_history.csv", index=False)

    # Fit the plateau stretch on selection engines only, searching multiple knees,
    # then freeze it before touching calibration or test engines.
    _, clean_records = _evaluate(model, clean_val, config, deployed_thresholds)
    plateau = fit_plateau_calibration(clean_records, knee=[100., 105., 110., 115., 120.], max_capped_increase=0.01)
    config["plateau_calibration"] = plateau
    model.config["plateau_calibration"] = plateau
    print(f"Plateau calibration: knee={plateau['knee']:.1f} alpha={plateau['alpha']:.3f} raw {plateau['baseline_raw_RMSE']:.3f}"
          f" -> {plateau['raw_RMSE']:.3f}, capped {plateau['baseline_capped_RMSE']:.3f}"
          f" -> {plateau['capped_RMSE']:.3f}", flush=True)

    clean_rep, clean_records = _evaluate(model, clean_val, config, deployed_thresholds)
    mixed_rep, _ = _evaluate(model, mixed_val, config, deployed_thresholds)
    stuck_rep, _ = _evaluate(model, stuck_val, config, deployed_thresholds)
    tail = engine_tail_score(clean_records)
    final = {"clean": clean_rep, "mixed": mixed_rep, "stuck": stuck_rep, "tail": tail,
             "plateau": plateau, "best_epoch": best["epoch"]}
    print(f"Selected validation: clean={clean_rep['RMSE']:.4f} mixed={mixed_rep['RMSE']:.4f} "
          f"stuck={stuck_rep['RMSE']:.4f} tail={tail:.4f}", flush=True)

    calibration, calibration_records = calibrate(model, cal, scaler, config,
                                                 config["validation_seed"] + 2, scale_frame=val)
    calibration_records.to_csv(output / "calibration_POOLED.csv", index=False)

    detector = audit_deployed_thresholds(model, val, scaler, config, config["validation_seed"],
                                         previous_audit=entry.get("detector_audit"))
    write_json(output / "detector_audit_POOLED.json", detector)
    if not detector["accepted"]:
        raise RuntimeError("Soft-gate detector audit failed its false-positive guards.")

    print("Re-auditing recovery with the soft gate...", flush=True)
    matrix = RecoveryValidationMatrix(val, scaler, config, seed=config["validation_seed"])
    recovery = audit_recovery_candidate(model, matrix, config, use_repair=True, stop_on_failure=False)
    write_json(output / "recovery_validation_POOLED.json", {"selected": recovery, "candidates": [recovery]})
    if not recovery.get("complete") or not recovery.get("accepted"):
        raise RuntimeError("Soft-gate recovery audit failed intact-cell guards.")

    upgraded = deepcopy(entry)
    upgraded["state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    upgraded["training_config"] = deepcopy(config)
    upgraded["thresholds"] = deployed_thresholds
    upgraded["calibration"] = calibration
    upgraded["correction_bound"] = 0.
    upgraded["repair_selected"] = bool(model.repair_enabled)
    upgraded["detector_audit"] = detector
    upgraded["recovery_validation_audit"] = recovery
    upgraded["v10_selection"] = final
    upgraded["source_bundle_sha256"] = sha(source)
    validate_pooled_entry(upgraded, config)

    bundle = {
        "format_version": POOLED_REPAIR_FORMAT_VERSION,
        "architecture": POOLED_REPAIR_ARCHITECTURE,
        "training_scope": "pooled",
        "requires_subset_label": False,
        "feature_columns": FEATURE_COLUMNS,
        "run_type": "v10_active_path_age_soft_gate_repair",
        "weights_retrained": False,
        "refined_components": ["predictor"],
        "corruption_protocol": CORRUPTION_PROTOCOL,
        "config": config,
        "models": {MODEL_KEY: upgraded},
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "torch_version": str(torch.__version__),
    }
    bundle_path = output / "grace_rul_ALL.pt"
    save_checkpoint(bundle_path, bundle)
    frozen_hash = sha(bundle_path)
    load_bundle(bundle_path, device="cpu")
    print(f"Exported {bundle_path} sha256={frozen_hash}", flush=True)

    if not final_evaluation:
        report = {"source_bundle_sha256": sha(source), "v10_bundle_sha256": frozen_hash,
                  "validation": final, "detector": detector,
                  "recovery_accepted": bool(recovery.get("accepted")),
                  "final_evaluation": "skipped"}
        write_json(output / "v10_report.json", _json_safe(report))
        return bundle_path, report

    # Evaluate the frozen bundle on internal and official partitions.
    internal_rows, internal_diag = [], []
    internal_frame = frame[frame.unit.isin(groups["test"])].copy()
    for family, severity, replicate in [("clean", 0., 0), ("stuck", .2, 0), ("stuck", .2, 1),
                                        ("stuck", .2, 2), ("mixed", .2, 0), ("mixed", .2, 1),
                                        ("mixed", .2, 2)]:
        ds = WindowDataset(internal_frame, scaler, config, mode="heldout",
                           seed=config["validation_seed"] + 100000 + replicate,
                           family=family, severity=severity)
        report, predictions = evaluate(model, ds, config, fault_thresholds=deployed_thresholds,
                                       capture_recovery_tails=True)
        predictions = _annotate(predictions, engines)
        for protocol, name in [(f"capped_{config['train_rul_cap']:g}", "capped"), ("official_uncapped", "raw")]:
            lo, hi = interval_bounds(predictions.prediction, calibration, protocol=protocol)
            predictions[f"interval_lower_{name}"], predictions[f"interval_upper_{name}"] = lo, hi
        predictions = predictions.assign(partition="internal_test_10pct", family=family,
                                          severity=severity, replicate=replicate, seed=config["seed"])
        predictions.to_csv(output / "predictions" / f"internal_{family}_s{severity:g}_r{replicate}.csv.gz",
                           index=False, compression="gzip")
        internal_rows += _point_rows(predictions, calibration, config, partition="internal_test_10pct",
                                     family=family, severity=severity, replicate=replicate, seed=config["seed"])
        internal_diag.append({"partition": "internal_test_10pct", "family": family,
                              "severity": severity, "replicate": replicate, **report})
        print(f"internal {family} r{replicate}: RMSE={report['RMSE']:.3f} raw={report.get('RMSE'):.3f}", flush=True)

    official_frames, official_meta, truths = [], [], {}
    for index, subset in enumerate(SUBSETS, 5):
        _, test, truth = load_subset(root / "data", subset)
        test["source_unit"], test["source_subset"] = test.unit, subset
        test["unit"] = index * 1_000_000 + test.unit
        truths.update({index * 1_000_000 + int(unit): float(value) for unit, value in truth.items()})
        official_frames.append(test)
        official_meta.append(test[["unit", "source_unit", "source_subset"]].drop_duplicates())
    official_frame = pd.concat(official_frames, ignore_index=True)
    official_engines = pd.concat(official_meta, ignore_index=True)
    official_rows, official_diag = [], []
    for family, severity, replicate in [("clean", 0., 0), ("stuck", .2, 0), ("stuck", .2, 1),
                                        ("stuck", .2, 2), ("mixed", .2, 0), ("mixed", .2, 1),
                                        ("mixed", .2, 2)]:
        ds = WindowDataset(official_frame, scaler, config, mode="test",
                           seed=config["validation_seed"] + 100000 + replicate,
                           family=family, severity=severity, truths=truths)
        report, predictions = evaluate(model, ds, config, fault_thresholds=deployed_thresholds,
                                       capture_recovery_tails=True)
        predictions = _annotate(predictions, official_engines)
        for protocol, name in [(f"capped_{config['train_rul_cap']:g}", "capped"), ("official_uncapped", "raw")]:
            lo, hi = interval_bounds(predictions.prediction, calibration, protocol=protocol)
            predictions[f"interval_lower_{name}"], predictions[f"interval_upper_{name}"] = lo, hi
        predictions = predictions.assign(partition="official_benchmark", family=family,
                                          severity=severity, replicate=replicate, seed=config["seed"])
        predictions.to_csv(output / "predictions" / f"official_{family}_s{severity:g}_r{replicate}.csv.gz",
                           index=False, compression="gzip")
        official_rows += _point_rows(predictions, calibration, config, partition="official_benchmark",
                                     family=family, severity=severity, replicate=replicate, seed=config["seed"])
        official_diag.append({"partition": "official_benchmark", "family": family,
                              "severity": severity, "replicate": replicate, **report})
        print(f"official {family} r{replicate}: RMSE={report['RMSE']:.3f}", flush=True)

    all_rows = pd.DataFrame(internal_rows + official_rows)
    all_rows.to_csv(output / "test_metrics.csv", index=False)
    all_rows[~all_rows.family.eq("clean")].to_csv(output / "robustness_summary.csv", index=False)
    pd.DataFrame(internal_diag + official_diag).to_csv(output / "recovery_diagnostics.csv", index=False)

    clean_official = all_rows[(all_rows.partition == "official_benchmark") &
                              (all_rows.family == "clean") & (all_rows.subset == MODEL_KEY) &
                              (all_rows.protocol == "capped_125")].iloc[0]
    clean_pred_path = output / "predictions" / "official_clean_s0_r0.csv.gz"
    clean_pred = pd.read_csv(clean_pred_path)
    report = {
        "source_bundle_sha256": sha(source),
        "v10_bundle_sha256": frozen_hash,
        "validation": final,
        "official_clean": clean_official.to_dict(),
        "official_long_rul": long_rul_report(clean_pred),
        "official_tail": per_engine_tail(clean_pred),
        "official_bootstrap": engine_bootstrap(clean_pred),
        "interval_bands": interval_band_report(clean_pred),
        "detector": detector,
        "recovery_accepted": bool(recovery.get("accepted")),
    }
    write_json(output / "v10_report.json", _json_safe(report))
    print(json.dumps({key: report[key] for key in ["official_clean", "official_long_rul", "official_tail"]}, indent=2), flush=True)
    return bundle_path, report


def rebuild_report(output, source_bundle_path):
    """Recompute the saved report from an exported run without retraining."""
    output = Path(output).resolve()
    source = Path(source_bundle_path).resolve()
    bundle_path = output / "grace_rul_ALL.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    entry = bundle["models"][MODEL_KEY]
    table = pd.read_csv(output / "test_metrics.csv")
    clean_official = table[(table.partition == "official_benchmark") &
                           (table.family == "clean") & (table.subset == MODEL_KEY) &
                           (table.protocol == "capped_125")].iloc[0]
    clean_pred = pd.read_csv(output / "predictions" / "official_clean_s0_r0.csv.gz")
    detector = json.loads((output / "detector_audit_POOLED.json").read_text())
    recovery = json.loads((output / "recovery_validation_POOLED.json").read_text())["selected"]
    report = {
        "source_bundle_sha256": sha(source),
        "v10_bundle_sha256": sha(bundle_path),
        "validation": entry["v10_selection"],
        "official_clean": clean_official.to_dict(),
        "official_long_rul": long_rul_report(clean_pred),
        "official_tail": per_engine_tail(clean_pred),
        "official_bootstrap": engine_bootstrap(clean_pred),
        "interval_bands": interval_band_report(clean_pred),
        "detector": detector,
        "recovery_accepted": bool(recovery.get("accepted")),
    }
    write_json(output / "v10_report.json", _json_safe(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path,
                        default=ROOT / "artifacts/pooled_repair_20261101T230338Z_9ea6_dpk/final_model_verified/grace_rul_ALL.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild v10_report.json from an already exported run.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--soft-power", type=float, default=3.0)
    parser.add_argument("--branch-weight", type=float, default=0.1)
    parser.add_argument("--prefix-weight", type=float, default=1.0)
    parser.add_argument("--active-weight", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--tail-weight", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--no-final-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Override training seed")
    args = parser.parse_args()
    if args.report_only:
        if args.output_dir is None:
            parser.error("--report-only requires --output-dir")
        report = rebuild_report(args.output_dir, args.source_bundle)
        print(json.dumps({key: report[key] for key in ["official_clean", "official_long_rul", "official_tail"]},
                         indent=2), flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ROOT / "artifacts" / f"pooled_v10_repair_{stamp}"
    finetune(ROOT, args.source_bundle, output, epochs=args.epochs, lr=args.lr,
             device=args.device, soft_temperature=args.soft_temperature, soft_power=args.soft_power,
             branch_weight=args.branch_weight, prefix_weight=args.prefix_weight,
             active_weight=args.active_weight, clean_weight=args.clean_weight,
             tail_weight=args.tail_weight, max_windows=args.max_windows,
             final_evaluation=not args.no_final_evaluation, seed_override=args.seed)


if __name__ == "__main__":
    main()
