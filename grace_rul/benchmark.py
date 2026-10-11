"""Matched architectural comparators and retrained ablations.

The paper comparators are explicit adaptations to a common, deployment-compatible
protocol. In particular, the Transformer corrector is supervised rather than TD3.
These experiments cannot establish reproduction of the original published scores.
"""

from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoints import pipeline_identity
from .config import SUBSETS
from .data import FAULTS, OperatingRegimeScaler, WindowDataset, load_subset, source_hashes
from .model import GraceRUL
from .training import device_for, evaluate, fit_corrector, fit_pipeline, metrics, select_fault_thresholds


VARIANTS = {
    "paper_A_cnn_gru": {"predictor_family": "cnn_gru", "train_repair": False, "correction": False},
    "paper_B_repair_cnn_lstm": {"predictor_family": "cnn_lstm", "train_repair": True,
        "trend_loss_weight": 0., "generator_reconstruction_loss_weight": 0., "correction": False},
    "paper_C_transformer_corrected": {"predictor_family": "transformer", "train_repair": False, "correction": True},
    "pure_transformer": {"predictor_family": "transformer", "train_repair": False, "correction": False},
    "grace_without_adversarial": {"adversarial_loss_weight": 0., "correction": True},
    "grace_without_trend": {"trend_loss_weight": 0., "correction": True},
    "grace_without_repair": {"train_repair": False, "correction": True},
    "grace_local": {"predictor_family": "cnn_gru", "correction": True},
    "grace_global": {"predictor_family": "transformer", "correction": True},
}


def paired_bootstrap(grace_squared, comparator_squared, seed_indices, engine_indices):
    """Resample seeds and whole engines; corruption replicates stay together."""
    grace_squared, comparator_squared = np.asarray(grace_squared), np.asarray(comparator_squared)
    if grace_squared.shape != comparator_squared.shape or grace_squared.ndim != 2:
        raise ValueError("Paired squared errors need matching [seed, engine] arrays.")
    sampled_grace = grace_squared[seed_indices[:, :, None], engine_indices[:, None, :]]
    sampled_comparator = comparator_squared[seed_indices[:, :, None], engine_indices[:, None, :]]
    grace_rmse = np.sqrt(sampled_grace.mean((1, 2)))
    comparator_rmse = np.sqrt(sampled_comparator.mean((1, 2)))
    return grace_rmse, comparator_rmse


def _comparison_tables(errors, config):
    rng = np.random.default_rng(config["split_seed"])
    indices, comparison, aggregate = {}, [], {}
    for key, by_seed in errors.items():
        variant, subset, family, severity, protocol = key
        if variant == "GRACE_RUL":
            continue
        primary = errors[("GRACE_RUL", subset, family, severity, protocol)]
        # Average each engine's dependent corruption replicates before resampling.
        base = np.stack([np.mean(list(by_seed[s].values()), axis=0) for s in config["seeds"]])
        grace = np.stack([np.mean(list(primary[s].values()), axis=0) for s in config["seeds"]])
        index_key = subset, base.shape
        if index_key not in indices:
            b, (nseeds, nengines) = config["bootstrap_samples"], base.shape
            indices[index_key] = (rng.integers(nseeds, size=(b, nseeds)), rng.integers(nengines, size=(b, nengines)))
        grace_draws, base_draws = paired_bootstrap(grace, base, *indices[index_key])
        delta = grace_draws - base_draws
        lo, hi = np.quantile(delta, [.025, .975])
        comparison.append({"variant": variant, "subset": subset, "family": family, "severity": severity,
            "protocol": protocol, "GRACE_RMSE": float(np.sqrt(grace.mean())),
            "comparator_RMSE": float(np.sqrt(base.mean())), "delta_RMSE": float(np.sqrt(grace.mean()) - np.sqrt(base.mean())),
            "delta_RMSE_CI_lower": float(lo), "delta_RMSE_CI_upper": float(hi),
            "engines": base.shape[1], "seeds": base.shape[0],
            "scope": "paired seed/engine bootstrap; replicates retained per engine"})
        aggregate.setdefault((variant, protocol), []).append((subset, family, grace, base, grace_draws, base_draws))
    checks = []
    for (variant, protocol), cells in aggregate.items():
        corrupt = [cell for cell in cells if cell[1] != "clean"]
        clean = [cell for cell in cells if cell[1] == "clean"]
        grace_mean = np.mean([np.sqrt(cell[2].mean()) for cell in corrupt])
        base_mean = np.mean([np.sqrt(cell[3].mean()) for cell in corrupt])
        draws_grace = np.mean([cell[4] for cell in corrupt], axis=0)
        draws_base = np.mean([cell[5] for cell in corrupt], axis=0)
        improvement = 1 - grace_mean / max(base_mean, 1e-12)
        lower, upper = np.quantile(1 - draws_grace / np.maximum(draws_base, 1e-12), [.025, .975])
        clean_ok = all(np.sqrt(cell[2].mean()) <= 1.05 * np.sqrt(cell[3].mean()) for cell in clean)
        complete = (not config["quick_run"] and set(config["subsets"]) == set(SUBSETS) and
                    len(set(config["seeds"])) >= 5 and
                    all(config.get(key) is None for key in ["maximum_batches", "maximum_engines", "maximum_windows"]) and
                    len(clean) == len(SUBSETS) and len(corrupt) == len(SUBSETS) * (len(FAULTS) + 1) * 5)
        checks.append({"variant": variant, "protocol": protocol, "GRACE_corrupted_macro_RMSE": grace_mean,
            "comparator_corrupted_macro_RMSE": base_mean, "relative_improvement": improvement,
            "relative_improvement_CI_lower": float(lower), "relative_improvement_CI_upper": float(upper),
            "clean_within_five_percent": clean_ok, "full_protocol_complete": complete,
            "acceptance_target_met": bool(complete and improvement >= .1 and lower > 0 and clean_ok),
            "scope": "harmonized adaptations; not reproduction of published paper scores"})
    return pd.DataFrame(comparison), pd.DataFrame(checks)


def run_matched_benchmarks(project_root, run_dir, config):
    root, output = Path(project_root), Path(run_dir)
    unknown = set(config["benchmark_variants"]) - set(VARIANTS)
    if unknown or not config["benchmark_variants"]:
        raise ValueError(f"Invalid benchmark variants: {sorted(unknown)}")
    records, errors, training_audit = [], {}, []
    prediction_file = output / "benchmark_predictions.tmp.csv.gz"
    first_predictions = True
    families = ["intermittent", "bias", "mixed"] if config["quick_run"] else [*FAULTS, "mixed"]
    severities = [.2] if config["quick_run"] else [.1, .2, .3, .4, .5]
    replicates = [0] if config["quick_run"] else [0, 1, 2]
    scenarios = [("clean", 0., 0), *product(families, severities, replicates)]
    for subset in config["subsets"]:
        train, test, truths = load_subset(root / "data", subset)
        if config["quick_run"]:
            test = test[test.unit.isin(sorted(test.unit.unique())[:3])]
        for seed in config["seeds"]:
            entry = torch.load(output / "seed_models" / f"{subset}_{seed}.pt", weights_only=True, map_location="cpu")
            cfg, groups = entry["training_config"], entry["partitions"]
            fit, validation = [train[train.unit.isin(groups[name])] for name in ["fit", "validation"]]
            if (entry["source_hashes"] != source_hashes(root / "data", subset) or
                    entry["pipeline_identity"] != pipeline_identity(fit, validation, subset, cfg, seed)):
                raise ValueError("Benchmark seed artifact no longer matches the run's data/configuration.")
            primary = GraceRUL(cfg).to(device_for(cfg))
            primary.load_state_dict(entry["state_dict"])
            primary.eval()
            scaler = OperatingRegimeScaler(entry["scaler"])
            trained = {"GRACE_RUL": (primary, cfg, entry["thresholds"])}
            for variant in config["benchmark_variants"]:
                overrides = VARIANTS[variant]
                trial = {**cfg, **{k: v for k, v in overrides.items() if k != "correction"},
                         "tune_before_training": False, "run_matched_benchmarks": False}
                directory = output / "benchmarks" / variant / "checkpoints"
                model, variant_scaler, history, score = fit_pipeline(fit, validation, subset, trial, seed,
                    checkpoint_path=directory / f"{subset}_{seed}.pt", progress_label=f"{subset} {variant} seed={seed}")
                # All variants use exactly the same fitting-only preprocessing.
                if variant_scaler.state != scaler.state:
                    raise ValueError("Matched variants must use identical fitting-only preprocessing.")
                model.scaler = variant_scaler
                if overrides["correction"]:
                    correction, _, _ = fit_corrector(model, fit, validation, subset, trial, seed, checkpoint_dir=directory)
                    score = float(correction.loc[correction.checkpoint_accepted, "validation_macro_RMSE"].min())
                thresholds = select_fault_thresholds(model, validation, scaler, trial, trial["validation_seed"])
                trained[variant] = model.eval(), trial, thresholds
                training_audit.append({"variant": variant, "subset": subset, "seed": seed, "upstream_epochs": len(history),
                    "validation_RMSE": score, "repair_selected": bool(model.repair_enabled),
                    "correction_bound": float(model.correction_bound) if overrides["correction"] else 0.,
                    "kind": "separately trained harmonized comparator or ablation"})
            for family, severity, replicate in scenarios:
                dataset = WindowDataset(test, scaler, cfg, mode="test", truths=truths,
                    seed=cfg["validation_seed"] + 100000 + replicate, family=family, severity=severity)
                for variant, (model, trial, thresholds) in trained.items():
                    report, predictions = evaluate(model, dataset, trial, fault_thresholds=thresholds)
                    # Unit ordering is fixed for all compared models/replicates.
                    predictions = predictions.sort_values("unit")
                    for protocol, target in [("capped_125", predictions.target), ("official_uncapped", predictions.raw_target)]:
                        records.append({"variant": variant, "subset": subset, "seed": seed, "family": family,
                            "severity": severity, "replicate": replicate, "protocol": protocol,
                            **report, **metrics(predictions.prediction, target)})
                        key = variant, subset, family, severity, protocol
                        errors.setdefault(key, {}).setdefault(seed, {})[replicate] = (predictions.prediction.to_numpy() - target.to_numpy()) ** 2
                    predictions.assign(variant=variant, subset=subset, seed=seed, family=family,
                        severity=severity, replicate=replicate).to_csv(prediction_file, mode="w" if first_predictions else "a",
                            compression="gzip", header=first_predictions, index=False)
                    first_predictions = False
                print(f"Matched benchmark {subset} seed={seed}: {family} severity={severity} replicate={replicate + 1}", flush=True)
            del trained, primary
    table = pd.DataFrame(records)
    # Collapse dependent corruption replicates before reporting seed variation.
    per_seed = table.groupby(["variant", "subset", "seed", "family", "severity", "protocol"], as_index=False).agg(
        RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), NASA_Score=("NASA_Score", "mean"))
    summary = per_seed.groupby(["variant", "subset", "family", "severity", "protocol"], as_index=False).agg(
        seeds=("seed", "nunique"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), NASA_Score_mean=("NASA_Score", "mean"))
    comparison, checks = _comparison_tables(errors, config)
    tables = {"benchmark_metrics": table, "benchmark_summary": summary, "benchmark_comparison": comparison,
              "superiority_checks": checks, "benchmark_training_audit": pd.DataFrame(training_audit)}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    prediction_file.replace(output / "benchmark_predictions.csv.gz")
    (output / "benchmark_manifest.json").write_text(json.dumps({"quick_run": config["quick_run"],
        "variants": config["benchmark_variants"], "config": config,
        "scope": "harmonized architecture adaptations, shared selected training settings; no independent baseline hyperparameter search",
        "paper_C_correction": "observable supervised residual; not original TD3",
        "published_paper_superiority_established": False}, indent=2))
    return tables
