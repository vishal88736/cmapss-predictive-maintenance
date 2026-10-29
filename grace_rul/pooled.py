"""One condition-aware model; engine-disjoint pooled 80/10/10 research protocol.

Only the four training files enter the split. The 10% validation pool is split
between selection and independent interval calibration. Official test files are
read only after the selected model is serialized and frozen.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoints import REPORT_OPTIONS, pipeline_identity, save_checkpoint
from .config import (CORRUPTION_PROTOCOL, DEFAULT_CONFIG, FEATURE_COLUMNS, POOLED_ARCHITECTURE,
                     POOLED_FORMAT_VERSION, SUBSETS, quick_config)
from .data import FAULTS, WindowDataset, load_subset, load_training_subset, select_background_indices
from .training import (calibrate, device_for, evaluate, exposure_metrics, fit_corrector, fit_pipeline,
                       interval_bounds, metrics, select_fault_thresholds, validate_training_config)


MODEL_KEY = "POOLED"
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pooled_config(*, quick=False):
    config = {**DEFAULT_CONFIG, **json.loads((ROOT / "configs/pooled_v9.json").read_text())}
    if quick:
        config = {**quick_config(config), "subsets": list(SUBSETS), "maximum_engines": None,
                  "pooled_maximum_engines_per_source": 20}
    return config


def validate_pooled_config(config):
    validate_training_config(config)
    if (config.get("training_scope") != "pooled" or config.get("operating_regimes") != "auto" or
            config.get("evaluation_prefix_policy") != "fixed_grid" or set(config["subsets"]) != set(SUBSETS) or
            config.get("pooled_protocol_revision") not in {1, 2}):
        raise ValueError("Pooled runs require all four sources, automatic fitting-only regimes and a fixed prefix grid.")
    if config.get("subset_overrides") or config["tune_before_training"] or config["run_matched_benchmarks"]:
        raise ValueError("The shared protocol has no per-FD overrides or per-FD tuning/comparator driver.")
    if config["pooled_validation_fraction"] != .1 or config["pooled_test_fraction"] != .1:
        raise ValueError("This protocol declares an 80/10/10 engine split.")
    if config["pooled_calibration_share_of_validation"] != .5:
        raise ValueError("Reserve half of the 10% validation pool for independent calibration.")
    maximum = config.get("pooled_maximum_engines_per_source")
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 20):
        raise ValueError("A capped source needs at least 20 engines for disjoint selection/calibration/test groups.")


def load_pooled_training(data_dir, config):
    """Namespace engine IDs; source identity is bookkeeping, never a feature."""
    frames, hashes = [], {}
    for index, subset in enumerate(SUBSETS, 1):
        path = Path(data_dir) / f"train_{subset}.txt"
        hashes[path.name] = sha(path)
        frame = load_training_subset(data_dir, subset)
        if config.get("pooled_maximum_engines_per_source") is not None:
            units = np.sort(frame.unit.unique())
            rng = np.random.default_rng(config["split_seed"] + index)
            chosen = rng.permutation(units)[:config["pooled_maximum_engines_per_source"]]
            frame = frame[frame.unit.isin(chosen)].copy()
        if frame.unit.nunique() < 20 or (frame.unit >= 1_000_000).any():
            raise ValueError("Each source needs at least 20 engines with IDs below one million.")
        frame["source_unit"], frame["source_subset"], frame["source_split"] = frame.unit, subset, "original_train"
        frame["unit"] = index * 1_000_000 + frame.unit
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(["unit", "cycle"]).reset_index(drop=True)
    if combined.duplicated(["unit", "cycle"]).any():
        raise ValueError("Pooled engine namespace collision.")
    return combined, hashes


def split_pooled_engines(frame, seed):
    groups = {name: [] for name in ["fit", "validation", "calibration", "test", "validation_pool"]}
    for index, (_, source) in enumerate(frame.groupby("source_subset", sort=True), 1):
        units = np.random.default_rng(np.random.SeedSequence([seed, index, 801010])).permutation(np.sort(source.unit.unique()))
        n = len(units)
        if n < 20:
            raise ValueError("At least 20 engines per source are required.")
        ntest, nval = int(np.ceil(.1 * n)), int(np.ceil(.1 * n))
        fit, validation, test = units[:n - nval - ntest], units[n - nval - ntest:n - ntest], units[n - ntest:]
        ncal = len(validation) // 2
        for key, values in [("fit", fit), ("calibration", validation[:ncal]),
                            ("validation", validation[ncal:]), ("test", test), ("validation_pool", validation)]:
            groups[key].extend(int(unit) for unit in values)
    return {key: sorted(values) for key, values in groups.items()}


def validate_pooled_entry(entry, config):
    """Check outer/OOF scopes before accepting a shared inference artifact."""
    validate_pooled_config(config)
    groups, records = entry["partitions"], entry["engine_manifest"]
    keys = ["fit", "validation", "calibration", "test"]
    scopes = [set(groups[key]) for key in keys]
    if any(not group or len(group) != len(groups[key]) for key, group in zip(keys, scopes)):
        raise ValueError("Pooled partitions must contain unique nonempty engine IDs.")
    if any(scopes[i] & scopes[j] for i in range(4) for j in range(i)):
        raise ValueError("Pooled fitting, selection, calibration and test engines overlap.")
    units = [row["unit"] for row in records]
    if len(set(units)) != len(units) or set(units) != set.union(*scopes):
        raise ValueError("Pooled engine manifest does not cover exactly the split.")
    if set(groups["validation_pool"]) != scopes[1] | scopes[2]:
        raise ValueError("Pooled validation pool must contain selection and calibration engines only.")
    for source in SUBSETS:
        source_ids = {row["unit"] for row in records if row["source_subset"] == source}
        n = len(source_ids)
        expected = int(np.ceil(.1 * n))
        if (n < 20 or len(source_ids & scopes[3]) != expected or
                len(source_ids & set(groups["validation_pool"])) != expected or
                len(source_ids & scopes[0]) != n - 2 * expected):
            raise ValueError("Pooled source proportions disagree with 80/10/10 rounding.")
    if set(entry["scaler"]["fit_units"]) != scopes[0] or "regime_selection" not in entry["scaler"]:
        raise ValueError("Shared preprocessing must be learned from exactly the fitting engines.")
    if set(entry["calibration"]["units"]) != scopes[2]:
        raise ValueError("Shared interval calibration must use exactly the separate calibration engines.")
    held = []
    for fold in entry["oof_audit"]:
        a, b, c = [set(fold[key]) for key in ["fit_units", "inner_validation_units", "held_out_units"]]
        if a & b or a & c or b & c or a | b | c != scopes[0]:
            raise ValueError("Pooled OOF correction uses an invalid fitting-engine scope.")
        held.extend(c)
    if len(held) != len(scopes[0]) or set(held) != scopes[0]:
        raise ValueError("Each fitting engine needs exactly one held-out correction fold.")
    if entry["repair_selected"]:
        from .recovery_validation import validate_recovery_audit
        audit = entry["recovery_validation_audit"]
        by_source = {source: sorted(row["unit"] for row in records if row["source_subset"] == source and row["unit"] in scopes[1])
                     for source in SUBSETS}
        if set(audit.get("source_audits", {})) != set(SUBSETS):
            raise ValueError("Shared recovery needs accepted source-level sensor audits for all four sources.")
        for source, units in by_source.items():
            measured = audit["source_audits"][source]
            validate_recovery_audit(measured, config, {"fit": groups["fit"], "validation": units, "calibration": groups["calibration"]})
            if measured["recovery_state_sha256"] != audit["recovery_state_sha256"]:
                raise ValueError("Source recovery audit fingerprint disagrees with the shared model audit.")


def prepare_pooled_run(project_root, config, output_dir):
    validate_pooled_config(config)
    root, output = Path(project_root), Path(output_dir)
    frame, hashes = load_pooled_training(root / "data", config)
    groups = split_pooled_engines(frame, config["split_seed"])
    engines = frame.groupby("unit", sort=True).agg(source_subset=("source_subset", "first"),
        source_unit=("source_unit", "first"), observations=("cycle", "size"), last_cycle=("cycle", "max")).reset_index()
    for key in ["fit", "validation", "calibration", "test"]:
        engines.loc[engines.unit.isin(groups[key]), "partition"] = key
    identity = {"format_version": POOLED_FORMAT_VERSION, "source_hashes": hashes, "partitions": groups,
                "training_config": {k: v for k, v in config.items() if k not in REPORT_OPTIONS}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    plan_path = output / "pooled_plan.json"
    if output.exists() and any(output.iterdir()) and not plan_path.is_file():
        raise ValueError("Existing directory is not this pooled run; use a fresh directory.")
    if plan_path.is_file() and json.loads(plan_path.read_text())["identity_sha256"] != fingerprint:
        raise ValueError("Pooled data/configuration changed; use a fresh directory.")
    output.mkdir(parents=True, exist_ok=True)
    counts = {key: len(value) for key, value in groups.items()}
    plan = {"identity_sha256": fingerprint, **identity, "engine_manifest": engines.to_dict("records"),
        "counts": counts, "total_engines": len(engines), "outer_split": "80% fitting / 10% validation pool / 10% internal test; whole engines, source-stratified",
        "validation_pool": "selection and calibration are disjoint halves of the 10% pool",
        "test_endpoint_policy": {"mode": "absolute_cycle_grid", "first_cycle": config["evaluation_start_cycle"], "stride": config["evaluation_stride"]},
        "metadata_not_model_features": ["unit", "source_unit", "source_subset", "source_split", "partition"],
        "official_test_data_used_for_fitting": False,
        "evidence_scope": "Exploratory engine holdout within previously used simulated C-MAPSS data; not an independent fleet validation."}
    write_json(plan_path, plan)
    engines.to_csv(output / "engine_partitions.csv", index=False)
    return frame, groups, plan


def _annotate(predictions, engines):
    return predictions.merge(engines[["unit", "source_subset", "source_unit"]], on="unit", how="left", validate="many_to_one")


def _point_rows(records, calibration, config, *, partition, family="clean", severity=0., replicate=0, seed=42):
    rows = []
    for subset in [MODEL_KEY, *SUBSETS]:
        selected = records if subset == MODEL_KEY else records[records.source_subset.eq(subset)]
        if selected.empty:
            continue
        for protocol, target in [(f"capped_{config['train_rul_cap']:g}", "target"), ("official_uncapped", "raw_target")]:
            errors = (selected.prediction - selected[target]).pow(2)
            lo, hi = interval_bounds(selected.prediction, calibration, protocol=protocol)
            rows.append({"partition": partition, "subset": subset, "seed": seed, "family": family,
                "severity": severity, "replicate": replicate, "protocol": protocol,
                **metrics(selected.prediction, selected[target]), **exposure_metrics(selected, selected[target]),
                "engines": int(selected.unit.nunique()), "engine_balanced_RMSE": float(np.sqrt(errors.groupby(selected.unit).mean().mean())),
                "Interval_Coverage": float(((selected[target] >= lo) & (selected[target] <= hi)).mean()),
                "metric_scope": "fixed-grid prefixes" if partition == "internal_test_10pct" else "official final endpoints"})
    return rows


def evaluation_cases(config):
    cases = [("clean", 0., 0)]
    if config["run_extended_evaluation"]:
        families = ["intermittent", "bias"] if config["quick_run"] else [*FAULTS, "mixed"]
        levels = [.2] if config["quick_run"] else [.1, .2, .3, .4, .5]
        repeats = [0] if config["quick_run"] else [0, 1, 2]
        cases += [(f, s, r) for f in families for s in levels for r in repeats]
    return cases


def _evaluate_partition(model, scaler, frame, engines, config, calibration, thresholds, output, partition, seed, truths=None):
    rows, diagnostics = [], []
    mode = "heldout" if truths is None else "test"
    for family, severity, replicate in evaluation_cases(config):
        dataset = WindowDataset(frame, scaler, config, mode=mode, seed=config["validation_seed"] + 100000 + replicate,
                                family=family, severity=severity, truths=truths)
        report, predictions = evaluate(model, dataset, config, fault_thresholds=thresholds, capture_recovery_tails=True)
        predictions = _annotate(predictions, engines)
        for protocol, name in [(f"capped_{config['train_rul_cap']:g}", "capped"), ("official_uncapped", "raw")]:
            lo, hi = interval_bounds(predictions.prediction, calibration, protocol=protocol)
            predictions[f"interval_lower_{name}"], predictions[f"interval_upper_{name}"] = lo, hi
        predictions = predictions.assign(partition=partition, family=family, severity=severity, replicate=replicate, seed=seed)
        name = f"{partition}_{family}_s{severity:g}_r{replicate}.csv.gz"
        predictions.to_csv(output / "predictions" / name, index=False, compression="gzip")
        rows += _point_rows(predictions, calibration, config, partition=partition, family=family,
                            severity=severity, replicate=replicate, seed=seed)
        diagnostics.append({"partition": partition, "family": family, "severity": severity, "replicate": replicate, **report})
        pd.DataFrame(rows).to_csv(output / f"{partition}_metrics.csv", index=False)
        print(f"{partition}: {family} severity={severity:g} replicate={replicate} complete", flush=True)
    return rows, diagnostics


def run_pooled_experiment(project_root, config=None, output_dir=None):
    from .artifacts import load_bundle
    config = pooled_config() if config is None else deepcopy(config)
    validate_pooled_config(config)
    device = device_for(config)
    root = Path(project_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir else root / "artifacts" / f"pooled_{'quick_' if config['quick_run'] else ''}{stamp}"
    frame, groups, plan = prepare_pooled_run(root, config, output)
    engines = pd.DataFrame(plan["engine_manifest"])
    fit, val, cal = [frame[frame.unit.isin(groups[key])].copy() for key in ["fit", "validation", "calibration"]]
    manifest = {"format_version": POOLED_FORMAT_VERSION, "architecture": POOLED_ARCHITECTURE,
        "training_scope": "pooled", "requires_subset_label": False, "quick_run": config["quick_run"],
        "training_recipe_revision": config["pooled_protocol_revision"],
        "generator_selection": config.get("generator_selection", "rul"),
        "finite_recovery_loss_weight": config.get("finite_recovery_loss_weight", 0.),
        "prefix_consistency_loss_weight": config.get("prefix_consistency_loss_weight", 0.),
        "created_at": stamp, "subsets": [MODEL_KEY], "training_sources": list(SUBSETS),
        "corruption_protocol": CORRUPTION_PROTOCOL, "artifact": "grace_rul_ALL.pt", "status": "training",
        "training_device": str(device), "torch_version": str(torch.__version__),
        "split_identity_sha256": plan["identity_sha256"], "counts": plan["counts"], "selection_uses_test_metrics": False}
    write_json(output / "run_manifest.json", manifest)
    print(f"Pooled training device: {device}; run directory: {output.resolve()}; engines: {plan['counts']}", flush=True)
    histories, corrections, cv, summaries, best_score, best = [], [], [], [], float("inf"), None
    for seed in config["seeds"]:
        model, scaler, history, upstream = fit_pipeline(fit, val, MODEL_KEY, config, seed,
            checkpoint_path=output / "checkpoints" / f"pipeline_{MODEL_KEY}_{seed}.pt")
        model.scaler = scaler
        correction, oof, cross_validation = fit_corrector(model, fit, val, MODEL_KEY, config, seed, checkpoint_dir=output / "checkpoints")
        model.detector_history.to_csv(output / f"detector_history_{MODEL_KEY}_{seed}.csv", index=False)
        write_json(output / f"detector_audit_{MODEL_KEY}_{seed}.json", model.detector_audit)
        write_json(output / f"recovery_validation_{MODEL_KEY}_{seed}.json",
            {"selected": model.recovery_validation_audit, "candidates": model.recovery_validation_history})
        score = float(correction.loc[correction.checkpoint_accepted, "validation_macro_RMSE"].min())
        thresholds = select_fault_thresholds(model, val, scaler, config, config["validation_seed"])
        summaries.append({"subset": MODEL_KEY, "seed": seed, "validation_selection_RMSE": score,
            "upstream_validation_selection_RMSE": upstream, "repair_selected": bool(model.repair_enabled),
            "selected_correction_bound": float(model.correction_bound)})
        histories.append(history); corrections.append(correction); cv.append(cross_validation)
        if score < best_score:
            best_score = score
            best = {"model": deepcopy(model).cpu(), "scaler": scaler, "oof": oof, "seed": int(seed), "thresholds": thresholds}
        pd.concat(histories, ignore_index=True).to_csv(output / "training_history.csv", index=False)
        pd.concat(corrections, ignore_index=True).to_csv(output / "correction_history.csv", index=False)
        pd.concat(cv, ignore_index=True).to_csv(output / "cross_validation_metrics.csv", index=False)
        pd.DataFrame(summaries).to_csv(output / "seed_summary.csv", index=False)
        del model
    # Selection is complete before any calibration or test prediction is made.
    model, scaler, seed = best["model"].to(device), best["scaler"], best["seed"]
    manifest.update(status="calibrating", selected_seed=seed, validation_selection_RMSE=best_score)
    write_json(output / "run_manifest.json", manifest)
    calibration, calibration_records = calibrate(model, cal, scaler, config, config["validation_seed"] + 2)
    calibration_records.to_csv(output / "calibration_POOLED.csv", index=False)
    background_ds = WindowDataset(fit, scaler, config, mode="correction", seed=seed, family="clean")
    indices, background_metadata = select_background_indices(background_ds, config["background_size"], seed)
    background = [background_ds[i] for i in indices]
    entry = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "training_config": config, "scaler": scaler.state, "thresholds": best["thresholds"], "calibration": calibration,
        "selected_seed": seed, "partitions": groups, "engine_manifest": plan["engine_manifest"],
        "correction_bound": float(model.correction_bound), "repair_selected": bool(model.repair_enabled),
        "pretrained_clean_RMSE": model.pretrained_clean_rmse, "pretrained_clean_NASA": model.pretrained_clean_nasa,
        "pretrained_clean_by_source": model.pretrained_clean_by_source,
        "pipeline_identity": pipeline_identity(fit, val, MODEL_KEY, config, seed), "oof_audit": best["oof"],
        "source_hashes": plan["source_hashes"], "split_protocol": {"outer": [0.8, 0.1, 0.1], "unit": "whole_engine",
            "validation_pool_calibration_share": .5, "identity_sha256": plan["identity_sha256"]},
        "detector_audit": model.detector_audit, "recovery_support": model.recovery_support,
        "recovery_validation_audit": model.recovery_validation_audit,
        "background_metadata": background_metadata, "background": torch.stack([b["reference"] for b in background]),
        "background_valid": torch.stack([b["valid"] for b in background])}
    validate_pooled_entry(entry, config)
    bundle = {"format_version": POOLED_FORMAT_VERSION, "architecture": POOLED_ARCHITECTURE,
        "training_scope": "pooled", "requires_subset_label": False, "feature_columns": FEATURE_COLUMNS,
        "run_type": "pooled_training", "weights_retrained": True,
        "corruption_protocol": CORRUPTION_PROTOCOL, "config": config, "models": {MODEL_KEY: entry},
        "created_at": stamp, "torch_version": str(torch.__version__)}
    if any(sha(root / "data" / name) != digest for name, digest in plan["source_hashes"].items()):
        raise ValueError("Training source files changed during fitting; the candidate is not exported.")
    bundle_path = output / "grace_rul_ALL.pt"
    save_checkpoint(bundle_path, bundle)
    load_bundle(bundle_path, device="cpu")
    frozen_hash = sha(bundle_path)
    (output / "predictions").mkdir(exist_ok=True)
    manifest.update(status="evaluating", frozen_bundle_sha256=frozen_hash)
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "evaluation_plan.json", {"frozen_bundle_sha256": frozen_hash, "selected_seed": seed,
        "internal_test_units": groups["test"], "prefix_policy": plan["test_endpoint_policy"],
        "cases": [{"family": f, "severity": s, "replicate": r} for f, s, r in evaluation_cases(config)],
        "quick_execution_scope": config["quick_run"],
        "official_evaluation_requested": config["pooled_evaluate_official"], "selection_or_refitting_during_evaluation": False})
    test_rows, diagnostics = _evaluate_partition(model, scaler, frame[frame.unit.isin(groups["test"])],
        engines, config, calibration, best["thresholds"], output, "internal_test_10pct", seed)
    if config["pooled_evaluate_official"]:
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
        external_rows, external_diagnostics = _evaluate_partition(model, scaler, pd.concat(official), pd.concat(metadata),
            config, calibration, best["thresholds"], output, "official_benchmark", seed, truths=truths)
        test_rows += external_rows; diagnostics += external_diagnostics
        manifest["official_benchmark_hashes"] = official_hashes
    table = pd.DataFrame(test_rows)
    table[table.family.eq("clean")].to_csv(output / "test_metrics.csv", index=False)
    table[~table.family.eq("clean")].to_csv(output / "robustness_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "recovery_diagnostics.csv", index=False)
    if sha(bundle_path) != frozen_hash:
        raise ValueError("Evaluation changed the frozen pooled bundle.")
    manifest.update(status="complete", bundle_unchanged_during_evaluation=True,
        evaluation_status=plan["evidence_scope"], official_benchmark_evaluated=config["pooled_evaluate_official"])
    write_json(output / "run_manifest.json", manifest)
    print(f"Saved {'QUICK EXECUTION CHECK' if config['quick_run'] else 'POOLED TRAINING'}: {bundle_path}", flush=True)
    return bundle_path, {"test_metrics": table[table.family.eq("clean")], "robustness_summary": table[~table.family.eq("clean")]}
