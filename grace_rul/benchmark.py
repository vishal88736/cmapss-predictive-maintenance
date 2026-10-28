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

from .checkpoints import matches_pipeline_identity, pipeline_identity
from .config import DEFAULT_CONFIG, SUBSETS
from .data import FAULTS, OperatingRegimeScaler, WindowDataset, load_subset, source_hashes, window_view
from .model import GraceRUL
from .training import device_for, evaluate, exposure_metrics, fit_corrector, fit_pipeline, metrics, select_fault_thresholds, validation_datasets, validation_score


VARIANTS = {
    "paper_A_cnn_gru": {"predictor_family": "cnn_gru", "train_repair": False, "train_corrupted_predictor": True, "correction": False},
    "paper_B_repair_cnn_lstm": {"predictor_family": "cnn_lstm", "train_repair": True,
        "trend_loss_weight": 0., "generator_reconstruction_loss_weight": 0.,
        "adversarial_loss_weight": DEFAULT_CONFIG["adversarial_loss_weight"], "correction": False},
    "paper_C_transformer_corrected": {"predictor_family": "transformer", "train_repair": False, "train_corrupted_predictor": True, "correction": True},
    "pure_transformer": {"predictor_family": "transformer", "train_repair": False, "train_corrupted_predictor": True, "correction": False},
    "grace_without_adversarial": {"adversarial_loss_weight": 0., "correction": True},
    "grace_without_trend": {"trend_loss_weight": 0., "correction": True},
    "grace_without_repair": {"train_repair": False, "train_corrupted_predictor": True, "correction": True},
    "grace_local": {"predictor_branch": "local", "correction": True},
    "grace_global": {"predictor_branch": "global", "correction": True},
    "grace_equal_fusion": {"equal_fusion": True, "correction": True},
    "grace_without_correction": {"correction": False},
    "cnn_gru_clean": {"predictor_family": "cnn_gru", "train_repair": False, "train_corrupted_predictor": False, "correction": False},
    "transformer_clean": {"predictor_family": "transformer", "train_repair": False, "train_corrupted_predictor": False, "correction": False},
}
PAPER_VARIANTS = {"paper_A_cnn_gru", "paper_B_repair_cnn_lstm", "paper_C_transformer_corrected", "pure_transformer"}


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
        clean_five = all(np.sqrt(cell[2].mean()) <= 1.05 * np.sqrt(cell[3].mean()) for cell in clean)
        clean_ok = all(np.sqrt(cell[2].mean()) <= (1 + config["max_clean_rmse_increase"]) * np.sqrt(cell[3].mean()) for cell in clean)
        complete = (not config["quick_run"] and set(config["subsets"]) == set(SUBSETS) and
                    len(set(config["seeds"])) >= 5 and
                    all(config.get(key) is None for key in ["maximum_batches", "maximum_engines", "maximum_windows"]) and
                    len(clean) == len(SUBSETS) and len(corrupt) == len(SUBSETS) * (len(FAULTS) + 1) * 5)
        checks.append({"variant": variant, "protocol": protocol, "GRACE_corrupted_macro_RMSE": grace_mean,
            "comparator_corrupted_macro_RMSE": base_mean, "relative_improvement": improvement,
            "relative_improvement_CI_lower": float(lower), "relative_improvement_CI_upper": float(upper),
            "clean_within_five_percent": clean_five, "clean_within_selection_tolerance": clean_ok,
            "clean_RMSE_tolerance": config["max_clean_rmse_increase"], "full_protocol_complete": complete,
            "independent_comparator_tuning": config.get("benchmark_tune", False) if variant in PAPER_VARIANTS else None,
            "acceptance_target_met": bool(complete and improvement >= .1 and lower > 0 and clean_ok and
                                          (variant not in PAPER_VARIANTS or (config.get("benchmark_tune", False) and config["tune_before_training"]))),
            "scope": "natural fault process, all endpoints; consult active-fault metrics; adapted methods, not published-score reproductions"})
    return pd.DataFrame(comparison), pd.DataFrame(checks)


def _active_comparison_tables(errors, masks, config):
    """Paired comparisons on faults present in both models' observed horizons.

    Retain the full engine axis for shared cluster resampling. Inactive endpoints
    contribute zero squared error and zero exposure, rather than clean examples.
    """
    rng = np.random.default_rng(config["split_seed"])
    indices, rows, grouped = {}, [], {}
    for key, by_seed in errors.items():
        variant, subset, family, severity, protocol = key
        if variant == "GRACE_RUL" or family == "clean":
            continue
        primary_key = ("GRACE_RUL", subset, family, severity, protocol)
        grace, base, exposure = [], [], []
        for seed in config["seeds"]:
            replicates = sorted(by_seed[seed])
            common = np.stack([masks[key][seed][r] & masks[primary_key][seed][r] for r in replicates])
            grace.append(np.mean(np.stack([errors[primary_key][seed][r] for r in replicates]) * common, axis=0))
            base.append(np.mean(np.stack([by_seed[seed][r] for r in replicates]) * common, axis=0))
            exposure.append(common.mean(0))
        grace, base, exposure = np.stack(grace), np.stack(base), np.stack(exposure)
        total = exposure.sum()
        index_key = subset, grace.shape
        if index_key not in indices:
            count, (nseeds, nengines) = config["bootstrap_samples"], grace.shape
            indices[index_key] = (rng.integers(nseeds, size=(count, nseeds)), rng.integers(nengines, size=(count, nengines)))
        seeds, engines = indices[index_key]
        weights = exposure[seeds[:, :, None], engines[:, None, :]].sum((1, 2))
        draws = []
        for squared in [grace, base]:
            sums = squared[seeds[:, :, None], engines[:, None, :]].sum((1, 2))
            draws.append(np.sqrt(np.divide(sums, weights, out=np.full_like(sums, np.nan), where=weights > 0)))
        g, b = (float(np.sqrt(squared.sum() / total)) if total else None for squared in [grace, base])
        finite = np.isfinite(draws[0]) & np.isfinite(draws[1])
        delta = draws[0][finite] - draws[1][finite]
        ci = np.quantile(delta, [.025, .975]) if len(delta) else [None, None]
        rows.append({"variant": variant, "subset": subset, "family": family, "severity": severity,
            "protocol": protocol, "GRACE_RMSE": g, "comparator_RMSE": b,
            "delta_RMSE": g - b if total else None,
            "delta_RMSE_CI_lower": None if ci[0] is None else float(ci[0]),
            "delta_RMSE_CI_upper": None if ci[1] is None else float(ci[1]),
            "common_affected_endpoint_fraction": float(exposure.mean()), "valid_bootstrap_draws": int(finite.sum()),
            "scope": "faults present in both observed contexts; paired engine/seed bootstrap"})
        grouped.setdefault((variant, protocol), []).append((g, b, draws[0], draws[1], bool(total)))
    checks = []
    for (variant, protocol), cells in grouped.items():
        exposed = [cell for cell in cells if cell[4]]
        g = float(np.mean([cell[0] for cell in exposed])) if exposed else None
        b = float(np.mean([cell[1] for cell in exposed])) if exposed else None
        gd = np.mean([cell[2] for cell in cells], axis=0)
        bd = np.mean([cell[3] for cell in cells], axis=0)
        finite = np.isfinite(gd) & np.isfinite(bd)
        ratios = 1 - gd[finite] / np.maximum(bd[finite], 1e-12)
        ci = np.quantile(ratios, [.025, .975]) if len(ratios) else [None, None]
        complete = (not config["quick_run"] and set(config["subsets"]) == set(SUBSETS) and
                    len(config["seeds"]) >= 5 and all(config.get(key) is None for key in ["maximum_batches", "maximum_engines", "maximum_windows"]) and
                    len(cells) == len(SUBSETS) * (len(FAULTS) + 1) * 5 and len(exposed) == len(cells))
        improvement = 1 - g / max(b, 1e-12) if exposed else None
        checks.append({"variant": variant, "protocol": protocol, "GRACE_active_macro_RMSE": g,
            "comparator_active_macro_RMSE": b, "active_relative_improvement": improvement,
            "active_improvement_CI_lower": None if ci[0] is None else float(ci[0]),
            "active_improvement_CI_upper": None if ci[1] is None else float(ci[1]),
            "all_active_scenarios_exposed": complete, "active_valid_bootstrap_draws": int(finite.sum()),
            "active_fault_acceptance": bool(complete and improvement >= .1 and len(ratios) >= .9 * config["bootstrap_samples"] and ci[0] > 0)})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def run_matched_benchmarks(project_root, run_dir, config):
    root, output = Path(project_root), Path(run_dir)
    unknown = set(config["benchmark_variants"]) - set(VARIANTS)
    if unknown or not config["benchmark_variants"]:
        raise ValueError(f"Invalid benchmark variants: {sorted(unknown)}")
    records, errors, masks, training_audit, selected_configs = [], {}, {}, [], {}
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
                    not matches_pipeline_identity(entry["pipeline_identity"], pipeline_identity(fit, validation, subset, cfg, seed))):
                raise ValueError("Benchmark seed artifact no longer matches the run's data/configuration.")
            primary = GraceRUL(cfg).to(device_for(cfg))
            primary.load_state_dict(entry["state_dict"])
            primary.eval()
            scaler = OperatingRegimeScaler(entry["scaler"])
            trained = {"GRACE_RUL": (primary, cfg, entry["thresholds"])}
            for variant in config["benchmark_variants"]:
                overrides = VARIANTS[variant]
                # Paper baselines start from the shared untuned configuration;
                # controlled ablations inherit the selected primary settings.
                initial = config if variant in PAPER_VARIANTS else cfg
                trial = {**initial, **{k: v for k, v in overrides.items() if k != "correction"},
                         "tune_before_training": False, "run_matched_benchmarks": False}
                trial.pop("subset_overrides", None)
                directory = output / "benchmarks" / variant / "checkpoints"
                search_trials = 0
                if variant in PAPER_VARIANTS and config["benchmark_tune"]:
                    from .tuning import comparison_candidates, run_validation_search
                    key = subset, variant
                    if key not in selected_configs:
                        selected_overrides, search = run_validation_search(root, {**trial, "subsets": [subset]},
                            directory.parent / "tuning", candidates=comparison_candidates(trial))
                        selected_configs[key] = selected_overrides[subset], len(search)
                    selected_overrides, search_trials = selected_configs[key]
                    trial.update(selected_overrides)
                reused = variant not in PAPER_VARIANTS and all(cfg.get(k) == v for k, v in overrides.items() if k != "correction")
                if reused:
                    # Correction fitting freezes upstream weights. Its ablation
                    # can reuse them exactly; identical selected ablations need no fit.
                    model = GraceRUL(trial).to(device_for(trial))
                    model.load_state_dict(entry["state_dict"])
                    if not overrides["correction"]:
                        torch.nn.init.zeros_(model.corrector[-1].weight)
                        torch.nn.init.zeros_(model.corrector[-1].bias)
                    variant_scaler, history = scaler, pd.DataFrame()
                    clean_val, mixed_val, selection_contexts = validation_datasets(validation, scaler, trial)
                    selection_options = {"selection_datasets": selection_contexts} if selection_contexts is not None else {}
                    score, _, _ = validation_score(model, clean_val, mixed_val, trial, **selection_options)
                else:
                    model, variant_scaler, history, score = fit_pipeline(fit, validation, subset, trial, seed,
                        checkpoint_path=directory / f"{subset}_{seed}.pt", progress_label=f"{subset} {variant} seed={seed}")
                # All variants use exactly the same fitting-only preprocessing.
                if variant_scaler.state != scaler.state:
                    raise ValueError("Matched variants must use identical fitting-only preprocessing.")
                model.scaler = variant_scaler
                if overrides["correction"] and not reused:
                    correction, _, _ = fit_corrector(model, fit, validation, subset, trial, seed, checkpoint_dir=directory)
                    score = float(correction.loc[correction.checkpoint_accepted, "validation_macro_RMSE"].min())
                thresholds = entry["thresholds"] if reused else select_fault_thresholds(model, validation, scaler, trial, trial["validation_seed"])
                trained[variant] = model.eval(), trial, thresholds
                training_audit.append({"variant": variant, "subset": subset, "seed": seed, "upstream_epochs": len(history),
                    "validation_RMSE": score, "repair_selected": bool(model.repair_enabled),
                    "correction_bound": float(model.correction_bound) if overrides["correction"] else 0.,
                    "corruption_trained_predictor": trial["train_corrupted_predictor"],
                    "predictor_branch": trial["predictor_branch"], "equal_fusion": trial["equal_fusion"],
                    "search_trials": search_trials, "independent_tuning": search_trials > 0,
                    "reused_identical_weights": reused,
                    "selected_training_config": json.dumps(trial, sort_keys=True),
                    "kind": "exact selected/upstream weight reuse" if reused else "independently tuned adaptation" if search_trials else "controlled retrained ablation or untuned reference"})
            for family, severity, replicate in scenarios:
                dataset = WindowDataset(test, scaler, cfg, mode="test", truths=truths,
                    seed=cfg["validation_seed"] + 100000 + replicate, family=family, severity=severity)
                for variant, (model, trial, thresholds) in trained.items():
                    report, predictions = evaluate(model, window_view(dataset, trial["sequence_length"]), trial, fault_thresholds=thresholds)
                    # Unit ordering is fixed for all compared models/replicates.
                    predictions = predictions.sort_values("unit")
                    for protocol, target in [(f"capped_{cfg['train_rul_cap']:g}", predictions.target), ("official_uncapped", predictions.raw_target)]:
                        records.append({"variant": variant, "subset": subset, "seed": seed, "family": family,
                            "severity": severity, "replicate": replicate, "protocol": protocol,
                            **report, **metrics(predictions.prediction, target), **exposure_metrics(predictions, target)})
                        key = variant, subset, family, severity, protocol
                        errors.setdefault(key, {}).setdefault(seed, {})[replicate] = (predictions.prediction.to_numpy() - target.to_numpy()) ** 2
                        masks.setdefault(key, {}).setdefault(seed, {})[replicate] = predictions.active_fault.to_numpy(bool)
                    predictions.assign(variant=variant, subset=subset, seed=seed, family=family,
                        severity=severity, replicate=replicate).to_csv(prediction_file, mode="w" if first_predictions else "a",
                            compression="gzip", header=first_predictions, index=False)
                    first_predictions = False
                print(f"Matched benchmark {subset} seed={seed}: {family} severity={severity} replicate={replicate + 1}", flush=True)
            del trained, primary
    table = pd.DataFrame(records)
    # Collapse dependent corruption replicates before reporting seed variation.
    per_seed = table.groupby(["variant", "subset", "seed", "family", "severity", "protocol"], as_index=False).agg(
        RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), NASA_Score=("NASA_Score", "mean"),
        active_fault_RMSE=("active_fault_RMSE", "mean"), active_fault_N=("active_fault_N", "mean"),
        affected_endpoint_fraction=("affected_endpoint_fraction", "mean"), damaged_entry_fraction=("damaged_entry_fraction", "mean"))
    summary = per_seed.groupby(["variant", "subset", "family", "severity", "protocol"], as_index=False).agg(
        seeds=("seed", "nunique"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), NASA_Score_mean=("NASA_Score", "mean"),
        active_fault_RMSE_mean=("active_fault_RMSE", "mean"), active_fault_N_mean=("active_fault_N", "mean"),
        affected_endpoint_fraction=("affected_endpoint_fraction", "mean"), damaged_entry_fraction=("damaged_entry_fraction", "mean"))
    comparison, checks = _comparison_tables(errors, config)
    active_comparison, active_checks = _active_comparison_tables(errors, masks, config)
    if not active_checks.empty:
        checks = checks.merge(active_checks, on=["variant", "protocol"], how="left")
        checks["acceptance_target_met"] &= checks.active_fault_acceptance.fillna(False)
    else:
        checks["acceptance_target_met"] = False
    tables = {"benchmark_metrics": table, "benchmark_summary": summary, "benchmark_comparison": comparison,
              "benchmark_active_comparison": active_comparison,
              "superiority_checks": checks, "benchmark_training_audit": pd.DataFrame(training_audit)}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    prediction_file.replace(output / "benchmark_predictions.csv.gz")
    (output / "benchmark_manifest.json").write_text(json.dumps({"quick_run": config["quick_run"],
        "variants": config["benchmark_variants"], "config": config,
        "scope": ("shared partitions/corruptions; independent method-specific baseline tuning" if config["benchmark_tune"]
                  else "shared partitions/corruptions; untuned baseline references") + "; controlled ablations inherit primary settings",
        "independent_baseline_tuning": config["benchmark_tune"],
        "fault_metric_scope": "all natural-process endpoints and separately reported active-fault endpoints",
        "paper_C_correction": "observable supervised residual; not original TD3",
        "published_paper_superiority_established": False}, indent=2))
    return tables
