#!/usr/bin/env python3
"""Freeze a validation-selected repair, refresh audits/calibration, then evaluate.

The source run is immutable. An optional validation-only predictor checkpoint
must match its exact source hash and partitions. No test result participates in
selection. Unchanged point predictions may be reused only for identical forward
state; their targets, intervals and metrics are checked/recomputed explicitly.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from grace_rul.artifacts import load_bundle
from grace_rul.calibration import interval_bounds
from grace_rul.checkpoints import save_checkpoint
from grace_rul.config import SUBSETS
from grace_rul.data import WindowDataset, load_subset
from grace_rul.detection import audit_deployed_thresholds, select_deployment_thresholds
from grace_rul.model import GraceRUL
from grace_rul.pooled import (_evaluate_partition, _point_rows, evaluation_cases,
                             load_pooled_training, sha, validate_pooled_entry, write_json)
from grace_rul.predictor_safety import engine_tail_score, fault_accuracy_guards
from grace_rul.recovery_validation import RecoveryValidationMatrix, audit_recovery_candidate
from grace_rul.training import calibrate, checkpoint_guards, device_for, evaluate, seed_all


def official_inputs(root):
    frames, metadata, truths, hashes = [], [], {}, {}
    for index, subset in enumerate(SUBSETS, 5):
        _, frame, truth = load_subset(root / "data", subset)
        frame["source_unit"], frame["source_subset"] = frame.unit, subset
        frame["unit"] = index*1_000_000 + frame.unit
        truths.update({index*1_000_000+int(unit): float(value) for unit, value in truth.items()})
        frames.append(frame)
        metadata.append(frame[["unit", "source_unit", "source_subset"]].drop_duplicates())
        for name in [f"test_{subset}.txt", f"RUL_{subset}.txt"]:
            hashes[name] = sha(root / "data" / name)
    return pd.concat(frames), pd.concat(metadata), truths, hashes


def reuse_point_predictions(source, output, frame, groups, official, truths, config, calibration):
    """Recompute every interval/metric while retaining verified identical points."""
    raw_end = frame.groupby("unit").cycle.max().to_dict()
    official_end = official.groupby("unit").cycle.max().to_dict()
    records, inputs, observed = [], {}, {"internal_test_10pct": set(), "official_benchmark": set()}
    for path in sorted((source / "predictions").glob("*.csv.gz")):
        p = pd.read_csv(path)
        if p.empty or p.duplicated(["unit", "cycle"]).any():
            raise ValueError(f"Invalid saved point predictions: {path}")
        case = {key: p[key].iloc[0] for key in ["partition", "family", "severity", "replicate", "seed"]}
        if any(p[key].nunique() != 1 for key in case):
            raise ValueError("A prediction file mixes evaluation cases.")
        partition = str(case["partition"])
        if partition not in observed:
            raise ValueError("Unexpected prediction partition.")
        if partition == "internal_test_10pct":
            if set(p.unit) != set(groups["test"]):
                raise ValueError("Internal predictions do not match the saved test engines.")
            target = p.unit.map(raw_end)-p.cycle
        else:
            if len(p) != len(truths) or set(p.unit) != set(truths):
                raise ValueError("Official predictions do not match original endpoints.")
            target = p.unit.map(official_end)-p.cycle+p.unit.map(truths)
        np.testing.assert_array_equal(p.raw_target, target)
        np.testing.assert_array_equal(p.target, np.minimum(target, config["train_rul_cap"]))
        if not np.isfinite(p.prediction).all():
            raise ValueError("Saved point predictions are non-finite.")
        for protocol, suffix in [(f"capped_{config['train_rul_cap']:g}", "capped"), ("official_uncapped", "raw")]:
            lo, hi = interval_bounds(p.prediction, calibration, protocol)
            p[f"interval_lower_{suffix}"], p[f"interval_upper_{suffix}"] = lo, hi
        p.to_csv(output / "predictions" / path.name, index=False, compression="gzip")
        observed[partition].add((str(case["family"]), float(case["severity"]), int(case["replicate"])))
        records += _point_rows(p, calibration, config, **case)
        inputs[str(path)] = sha(path)
    expected = set(evaluation_cases(config))
    if any(cases != expected for cases in observed.values()):
        raise ValueError("Source predictions do not cover the complete declared evaluation matrix.")
    if (source / "recovery_diagnostics.csv").exists():
        shutil.copy2(source / "recovery_diagnostics.csv", output / "recovery_diagnostics.csv")
    return records, inputs


def refresh_pooled_bundle(project_root, source_run, output_dir=None, *, candidate=None, device="auto"):
    root, source = Path(project_root).resolve(), Path(source_run).resolve()
    source_path = source / "grace_rul_ALL.pt"
    source_hash = sha(source_path)
    source_bundle, loaded = load_bundle(source_path, device="cpu")
    source_entry = source_bundle["models"]["POOLED"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir).resolve() if output_dir else root / "artifacts" / f"pooled_repaired_{stamp}"
    if output.exists() or source == output or source in output.parents:
        raise FileExistsError("Use a fresh output directory outside the preserved source run.")
    output.mkdir(parents=True)
    (output / "predictions").mkdir()
    config = deepcopy(source_bundle["config"])
    state = source_entry["state_dict"]
    selection = None
    if candidate is not None:
        provisional = torch.load(candidate, map_location="cpu", weights_only=True)
        if provisional.get("deployable") is not False or provisional.get("status") != "validation_only":
            raise ValueError("Expected a non-deployable validation-only candidate.")
        if provisional["source_bundle_sha256"] != source_hash or provisional["partitions"] != source_entry["partitions"]:
            raise ValueError("Candidate source hash or partitions differ from the preserved source.")
        allowed = {"device", "physical_slew_loss_weight", "physical_rebound_loss_weight", "calibration_policy",
                   "finite_repair_threshold_floor", "deployment_audit_revision", "finetune_tail_loss_weight",
                   "finetune_tail_fraction", "finetune_maximum_windows", "global_pooling",
                   "finite_repair_threshold_selection", "finetune_corrupted_rul_loss_weight",
                   "finetune_epochs", "finetune_learning_rate", "finetune_weight_decay", "finetune_min_learning_rate"}
        proposed = provisional["config"]
        changed = {key for key in set(config)|set(proposed) if config.get(key) != proposed.get(key)}
        if changed-allowed:
            raise ValueError(f"Candidate changed unsupported protocol fields: {sorted(changed-allowed)}")
        config, state, selection = deepcopy(proposed), provisional["state_dict"], provisional["selection"]
    config.update(device=str(device_for({"device": device})), calibration_policy="observable_scaled_engine_max",
                  deployment_audit_revision=1, finite_repair_threshold_selection="minimal_source_context_FPR_with_floor")
    # Retain the conservative floor; any minimal source/context tightening below
    # is recorded and must pass all final guards before calibration.
    config.setdefault("finite_repair_threshold_floor", float(np.min(state["finite_repair_thresholds"].numpy())))
    seed_all(int(source_entry["selected_seed"]))
    model = GraceRUL(config).to(config["device"])
    model.load_state_dict(state, strict=True)
    model.eval()
    frame, hashes = load_pooled_training(root / "data", config)
    if hashes != source_entry["source_hashes"]:
        raise ValueError("Training source files do not match the preserved model.")
    groups = source_entry["partitions"]
    val = frame[frame.unit.isin(groups["validation"])].copy()
    cal = frame[frame.unit.isin(groups["calibration"])].copy()
    scaler = loaded["POOLED"]["scaler"]
    original_thresholds = model.finite_repair_thresholds.detach().cpu().tolist()
    thresholds = select_deployment_thresholds(model, val, scaler, config, config["validation_seed"])
    model.finite_repair_thresholds.copy_(torch.tensor(thresholds, device=config["device"]))
    write_json(output / "threshold_selection.json", {"previous": original_thresholds, "selected": thresholds,
               "scope": "Held selection engines only; every source/channel/context FPR <= configured limit",
               "selection_uses_test_metrics": False})
    same_points = (config.get("global_pooling", "last") == source_bundle["config"].get("global_pooling", "last") and
                   all(torch.equal(value.cpu(), source_entry["state_dict"][key]) for key, value in model.state_dict().items()))
    weights_changed = any(not torch.equal(value.detach().cpu(), source_entry["state_dict"][key])
                          for key,value in model.named_parameters())
    reference_model = loaded["POOLED"]["model"].to(config["device"])
    current, previous, predictions = {}, {}, {}
    for family in ["clean", "mixed", "stuck"]:
        ds = WindowDataset(val, scaler, config, mode="validation", family=family, severity=.2, seed=config["validation_seed"])
        current[family], predictions[family] = evaluate(model, ds, config, fault_thresholds=thresholds)
        previous[family], baseline_predictions = evaluate(reference_model, ds, config,
                                                          fault_thresholds=source_entry["thresholds"])
        if family == "clean":
            baseline_tail = engine_tail_score(baseline_predictions)
    guards = checkpoint_guards(current["clean"], current["mixed"], source_entry["pretrained_clean_RMSE"], config,
        clean_nasa_reference=source_entry["pretrained_clean_NASA"], clean_source_reference=source_entry["pretrained_clean_by_source"])
    guards.update(fault_accuracy_guards({key: current[key] for key in ["mixed", "stuck"]},
        {key: previous[key] for key in ["mixed", "stuck"]}, config["max_clean_rmse_increase"]))
    if candidate is not None:
        guards["engine_tail"] = engine_tail_score(predictions["clean"]) <= baseline_tail
    write_json(output / "active_path_validation.json", {"guards": guards, "reports": current,
               "baseline_reports": previous, "baseline_engine_tail": baseline_tail,
               "selected_engine_tail": engine_tail_score(predictions["clean"])})
    if not all(guards.values()):
        raise RuntimeError(f"Final active-path validation rejected candidate: {[k for k,v in guards.items() if not v]}")
    detector = audit_deployed_thresholds(model, val, scaler, config, config["validation_seed"],
                                         previous_audit=source_entry.get("detector_audit"))
    write_json(output / "detector_audit_POOLED_42.json", detector)
    if not detector["accepted"]:
        raise RuntimeError("Deployed detector failed validation false-positive guards.")
    matrix = RecoveryValidationMatrix(val, scaler, config, seed=config["validation_seed"])
    recovery = audit_recovery_candidate(model, matrix, config, use_repair=True, stop_on_failure=False)
    write_json(output / "recovery_validation_POOLED_42.json", {"selected": recovery, "candidates": [recovery]})
    if not recovery["complete"] or not recovery["accepted"]:
        raise RuntimeError("Complete recovery matrix rejected candidate.")
    print("All active-path, detector and 41-context recovery guards passed; calibrating frozen model.", flush=True)
    calibration, cal_records = calibrate(model, cal, scaler, config, config["validation_seed"]+2, scale_frame=val)
    cal_records.to_csv(output / "calibration_POOLED.csv", index=False)
    write_json(output / "calibration_contract.json", calibration)
    entry = deepcopy(source_entry)
    entry.update(state_dict={key: value.detach().cpu() for key,value in model.state_dict().items()},
                 training_config=deepcopy(config), thresholds=thresholds, calibration=calibration,
                 detector_audit=detector, recovery_validation_audit=recovery,
                 original_training_config=source_entry.get("original_training_config", source_entry["training_config"]))
    validate_pooled_entry(entry, config)
    bundle = deepcopy(source_bundle)
    bundle.update(config=config, models={"POOLED": entry}, created_at=stamp,
                  run_type="validated_predictor_repair" if not same_points else "deployment_audit_and_calibration_refresh",
                  weights_retrained=weights_changed, source_bundle_sha256=source_hash)
    artifact = output / "grace_rul_ALL.pt"
    save_checkpoint(artifact, bundle)
    load_bundle(artifact, device="cpu")
    frozen_hash = sha(artifact)
    manifest = {"format_version": 9, "architecture": bundle["architecture"], "training_scope": "pooled",
                "requires_subset_label": False, "created_at": stamp, "source_bundle": str(source_path),
                "source_bundle_sha256": source_hash, "artifact": artifact.name, "frozen_bundle_sha256": frozen_hash,
                "status": "evaluating", "selection_uses_test_metrics": False, "candidate": str(candidate) if candidate else None,
                "selection": selection, "weights_retrained": weights_changed,
                "forward_changed": not same_points, "point_predictions_reused": same_points,
                "calibration_policy": config["calibration_policy"], "partition_counts": {k:len(v) for k,v in groups.items()},
                "training_device": config["device"], "torch_version": str(torch.__version__)}
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "evaluation_plan.json", {"frozen_bundle_sha256": frozen_hash,
               "selection_or_refitting_during_evaluation": False, "point_predictions_reused": same_points,
               "cases": [{"family": f,"severity": s,"replicate": r} for f,s,r in evaluation_cases(config)]})
    # Everything below is evaluation after model/configuration/calibration freeze.
    official, metadata, truths, official_hashes = official_inputs(root)
    seed = int(entry["selected_seed"])
    if same_points:
        rows, prediction_sources = reuse_point_predictions(source, output, frame, groups, official, truths, config, calibration)
        write_json(output / "reused_point_prediction_hashes.json", prediction_sources)
    else:
        engines = pd.DataFrame(entry["engine_manifest"])
        rows, diagnostics = _evaluate_partition(model, scaler, frame[frame.unit.isin(groups["test"])],
            engines, config, calibration, thresholds, output, "internal_test_10pct", seed)
        extra, extra_diagnostics = _evaluate_partition(model, scaler, official, metadata,
            config, calibration, thresholds, output, "official_benchmark", seed, truths=truths)
        rows += extra
        pd.DataFrame(diagnostics+extra_diagnostics).to_csv(output / "recovery_diagnostics.csv", index=False)
    table = pd.DataFrame(rows)
    for partition, values in table.groupby("partition"):
        values.to_csv(output / f"{partition}_metrics.csv", index=False)
    table[table.family.eq("clean")].to_csv(output / "test_metrics.csv", index=False)
    table[~table.family.eq("clean")].to_csv(output / "robustness_summary.csv", index=False)
    if sha(source_path) != source_hash or sha(artifact) != frozen_hash:
        raise RuntimeError("A source/frozen bundle changed during evaluation.")
    manifest.update(status="complete", official_source_hashes=official_hashes,
                    source_unchanged=True, evaluation_complete=True)
    write_json(output / "run_manifest.json", manifest)
    print(f"Completed frozen evaluation: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT/"artifacts/pooled_v9_r2_finetuned_final")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    refresh_pooled_bundle(ROOT, args.source_run, args.output_dir, candidate=args.candidate, device=args.device)


if __name__ == "__main__":
    main()
