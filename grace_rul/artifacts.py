"""Versioned tensor-only artifact contract and experiment driver."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import ARCHITECTURE, CORRUPTION_PROTOCOL, FEATURE_COLUMNS, FORMAT_VERSION, SUPPORTED_ARCHITECTURES
from .data import FAULTS, WindowDataset, load_subset, select_background_indices, source_hashes, split_engines
from .training import calibrate, device_for, evaluate, exposure_metrics, fit_corrector, fit_pipeline, interval_bounds, metrics, select_fault_thresholds
from .training import _Progress, validate_training_config
from .checkpoints import pipeline_identity, save_checkpoint


def run_experiment(project_root, config, output_dir=None):
    root = Path(project_root)
    validate_training_config(config)
    device = device_for(config)  # Fail before creating a run directory if requested CUDA is unavailable.
    print(f"Training device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "") +
          f" | PyTorch {torch.__version__}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir else root / "artifacts" / (("quick_" if config["quick_run"] else "full_") + stamp)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    print(f"Run directory: {output.resolve()} (completed epochs are checkpointed here)", flush=True)
    run_config = dict(config)
    search_table = None
    if config["tune_before_training"]:
        from .tuning import run_validation_search
        overrides, search_table = run_validation_search(root, config, output / "tuning")
        run_config["subset_overrides"] = overrides
    models, histories, correction_histories, clean_metrics, robustness, ablations = {}, [], [], [], [], []
    seed_summary, cross_validation_reports, background_audits = [], [], []
    for subset in run_config["subsets"]:
        config = {**run_config, **run_config.get("subset_overrides", {}).get(subset, {})}
        train, test, truths = load_subset(root / "data", subset)
        if config["quick_run"]:
            test = test[test.unit.isin(sorted(test.unit.unique())[:3])].copy()
        best_selection, selected = float("inf"), None
        for seed in config["seeds"]:
            progress = _Progress(f"{subset} seed={seed}", config)
            progress.emit("starting staged training", force=True)
            groups = split_engines(train, config["split_seed"], config.get("maximum_engines"),
                                   config["validation_fraction"], config["calibration_fraction_of_training"])
            fit, val, cal = [train[train.unit.isin(groups[name])].copy() for name in ["fit", "validation", "calibration"]]
            model, scaler, history, selection = fit_pipeline(fit, val, subset, config, seed,
                checkpoint_path=checkpoint_dir / f"pipeline_{subset}_{seed}.pt")
            model.scaler = scaler
            corrector_history, oof_audit, cv_report = fit_corrector(model, fit, val, subset, config, seed, checkpoint_dir=checkpoint_dir)
            cross_validation_reports.append(cv_report)
            progress.emit("selecting validation fault thresholds", force=True)
            thresholds = select_fault_thresholds(model, val, scaler, config, config["validation_seed"])
            progress.emit("calibrating prediction intervals", force=True)
            calibration, calibration_records = calibrate(model, cal, scaler, config, config["validation_seed"] + 2)
            histories.append(history); correction_histories.append(corrector_history)
            upstream_selection = selection
            selection = float(corrector_history.loc[corrector_history.checkpoint_accepted, "validation_macro_RMSE"].min())
            seed_summary.append({"subset": subset, "seed": seed, "validation_selection_RMSE": selection,
                                 "upstream_validation_selection_RMSE": upstream_selection,
                                 "repair_selected": bool(model.repair_enabled),
                                 "pretrained_clean_RMSE": model.pretrained_clean_rmse,
                                 "selected_adversarial_weight": config["adversarial_loss_weight"],
                                 "selected_correction_bound": float(model.correction_bound),
                                 "correction_selected": bool(corrector_history.iloc[0].validation_macro_RMSE > selection),
                                 "validation_no_correction_RMSE": float(corrector_history.iloc[0].validation_macro_RMSE)})
            test_dataset = WindowDataset(test, scaler, config, mode="test", seed=seed + 3, family="clean", truths=truths)
            progress.emit("evaluating clean official test engines", force=True)
            report, predictions = evaluate(model, test_dataset, config, fault_thresholds=thresholds)
            for protocol, target in [(f"capped_{config['train_rul_cap']:g}", predictions.target), ("official_uncapped", predictions.raw_target)]:
                row = {"subset": subset, "seed": seed, "protocol": protocol, **metrics(predictions.prediction, target)}
                lo, hi = interval_bounds(predictions.prediction, calibration)
                row["Interval_Coverage"] = float(((target >= lo) & (target <= hi)).mean())
                clean_metrics.append(row)
            if config.get("run_extended_evaluation", False):
                families = ["intermittent", "bias"] if config["quick_run"] else [*FAULTS, "mixed"]
                severities = [.2] if config["quick_run"] else [.1, .2, .3, .4, .5]
                replicates = [0] if config["quick_run"] else [0, 1, 2]
                for family in families:
                    progress.emit(f"evaluating robustness: {family}", force=True)
                    for severity in severities:
                        for replicate in replicates:
                            ds = WindowDataset(test, scaler, config, mode="test", seed=config["validation_seed"] + 100000 + replicate, family=family, severity=severity, truths=truths)
                            fault_report, fault_predictions = evaluate(model, ds, config, fault_thresholds=thresholds)
                            progress.emit(f"robustness: {family} severity={severity} replicate={replicate + 1}/{len(replicates)} complete")
                            lo, hi = interval_bounds(fault_predictions.prediction, calibration)
                            for protocol, target in [(f"capped_{config['train_rul_cap']:g}", fault_predictions.target), ("official_uncapped", fault_predictions.raw_target)]:
                                measured = metrics(fault_predictions.prediction, target)
                                clean_rmse = metrics(predictions.prediction, predictions.raw_target if protocol == "official_uncapped" else predictions.target)["RMSE"]
                                robustness.append({"subset": subset, "seed": seed, "family": family, "severity": severity,
                                    "replicate": replicate, "protocol": protocol, **fault_report, **measured,
                                    **exposure_metrics(fault_predictions, target),
                                    "Interval_Coverage": float(((target >= lo) & (target <= hi)).mean()),
                                    "relative_RMSE_increase": measured["RMSE"] / max(clean_rmse, 1e-8) - 1})
                for variant, options in [("full", {}), ("no_repair", {"use_repair": False}), ("local_only", {"branch": "local"}), ("global_only", {"branch": "global"}), ("equal_fusion", {"equal_fusion": True}), ("no_correction", {"use_correction": False})]:
                    progress.emit(f"evaluating ablation: {variant}")
                    ds = WindowDataset(test, scaler, config, mode="test", seed=config["validation_seed"] + 100000, family="bias", severity=.2, truths=truths)
                    ablation, _ = evaluate(model, ds, config, fault_thresholds=thresholds, **options)
                    ablations.append({"subset": subset, "seed": seed, "variant": variant, "kind": "inference intervention; not a retrained baseline", **ablation})
            background_ds = WindowDataset(fit, scaler, config, mode="correction", seed=seed, family="clean")
            indices, background_metadata = select_background_indices(background_ds, config["background_size"], seed)
            background = [background_ds[i] for i in indices]
            background_audits.extend({"subset": subset, "seed": seed, **row} for row in background_metadata)
            entry = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "scaler": scaler.state,
                "thresholds": thresholds, "calibration": calibration, "selected_seed": int(seed), "partitions": groups,
                "training_config": config, "correction_bound": float(model.correction_bound),
                "repair_selected": bool(model.repair_enabled),
                "background_metadata": background_metadata,
                "pipeline_identity": pipeline_identity(fit, val, subset, config, seed),
                "oof_audit": oof_audit, "source_hashes": source_hashes(root / "data", subset),
                "split_protocol": {"validation_fraction": config["validation_fraction"],
                                   "calibration_fraction_of_training": config["calibration_fraction_of_training"],
                                   "training_pool_units": groups["fit"] + groups["calibration"]},
                "background": torch.stack([b["reference"] for b in background]), "background_valid": torch.stack([b["valid"] for b in background])}
            save_checkpoint(output / "seed_models" / f"{subset}_{seed}.pt", entry)
            # Selection is validation-only; never choose a seed by official test results.
            if selection < best_selection:
                best_selection, selected = selection, entry
            calibration_records.assign(subset=subset, seed=seed).to_csv(output / f"calibration_{subset}_{seed}.csv", index=False)
            predictions.assign(subset=subset, seed=seed).to_csv(output / f"predictions_{subset}_{seed}.csv", index=False)
            progress.emit("seed complete; saving results", force=True)
            del model
        models[subset] = selected
    config = run_config
    tables = {"training_history": pd.concat(histories, ignore_index=True), "correction_history": pd.concat(correction_histories, ignore_index=True),
        "test_metrics": pd.DataFrame(clean_metrics), "seed_summary": pd.DataFrame(seed_summary),
        "cross_validation_metrics": pd.concat(cross_validation_reports, ignore_index=True),
        "background_audit": pd.DataFrame(background_audits)}
    if config.get("run_extended_evaluation", False):
        tables["robustness_summary"] = pd.DataFrame(robustness)
        tables["ablation_metrics"] = pd.DataFrame(ablations)
    tables["cross_validation_summary"] = tables["cross_validation_metrics"].groupby(["subset", "seed", "family", "scope"], as_index=False).agg(
        folds=("fold", "nunique"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
    tables["test_seed_summary"] = tables["test_metrics"].groupby(["subset", "protocol"], as_index=False).agg(
        seeds=("seed", "nunique"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), NASA_Score_mean=("NASA_Score", "mean"))
    if search_table is not None:
        tables["validation_search"] = search_table
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    bundle = {"format_version": FORMAT_VERSION, "architecture": ARCHITECTURE,
              "corruption_protocol": CORRUPTION_PROTOCOL, "feature_columns": FEATURE_COLUMNS,
              "config": config, "models": models, "created_at": stamp, "torch_version": str(torch.__version__)}
    path = output / "grace_rul_ALL.pt"
    temporary = output / "bundle.tmp"
    torch.save(bundle, temporary); os.replace(temporary, path)
    (output / "run_manifest.json").write_text(json.dumps({"format_version": FORMAT_VERSION, "architecture": ARCHITECTURE,
        "corruption_protocol": CORRUPTION_PROTOCOL,
        "quick_run": config["quick_run"], "created_at": stamp, "subsets": list(models), "artifact": path.name}, indent=2))
    print(f"Saved {'QUICK EXECUTION CHECK' if config['quick_run'] else 'FULL TRAINING RUN'}: {path}", flush=True)
    if config["run_matched_benchmarks"]:
        from .benchmark import run_matched_benchmarks
        tables.update(run_matched_benchmarks(root, output, config))
    return path, tables


def load_bundle(path, device="cpu"):
    from .data import OperatingRegimeScaler
    from .model import GraceRUL
    from .normalization import transform_spec
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if (bundle.get("format_version") not in SUPPORTED_ARCHITECTURES or
            bundle.get("architecture") != SUPPORTED_ARCHITECTURES.get(bundle.get("format_version")) or
            bundle.get("feature_columns") != FEATURE_COLUMNS):
        raise ValueError(f"Expected a GRACE-RUL format-{FORMAT_VERSION} or supported format-5 artifact. Legacy formats below 5 are incompatible.")
    if set(bundle["models"]) != set(bundle["config"]["subsets"]):
        raise ValueError("Bundle subsets do not match its configuration.")
    loaded = {}
    for subset, entry in bundle["models"].items():
        thresholds = np.asarray(entry["thresholds"], float)
        calibration = entry["calibration"]
        if thresholds.shape != (21,) or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
            raise ValueError("Invalid sensor fault thresholds.")
        if not np.isfinite([calibration["lower_offset"], calibration["upper_offset"]]).all() or calibration["lower_offset"] > calibration["upper_offset"]:
            raise ValueError("Invalid empirical interval calibration.")
        model = GraceRUL(entry.get("training_config", bundle["config"])).to(device)
        if transform_spec(entry["scaler"]) != transform_spec(model.config):
            raise ValueError("Saved scaler and model sensor representations disagree.")
        model.load_state_dict(entry["state_dict"], strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("Non-finite model weights.")
        if not bool(torch.isfinite(model.correction_bound)) or float(model.correction_bound) <= 0:
            raise ValueError("Invalid selected correction bound.")
        if (entry.get("correction_bound") != float(model.correction_bound) or
                entry.get("repair_selected") != bool(model.repair_enabled)):
            raise ValueError("Selected repair/correction metadata does not match the model.")
        loaded[subset] = {"model": model.eval(), "scaler": OperatingRegimeScaler(entry["scaler"]), "entry": entry}
    return bundle, loaded
