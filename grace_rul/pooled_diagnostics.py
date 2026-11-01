"""Validation-only interventions on saved pooled weights, with an optional fit pilot.

These experiments do not select a production model or recalibrate intervals.
Completed legacy checkpoints can contain only their accepted fallback, not the
rejected joint predictor. A warm generator must never be mislabeled as that
missing predictor. The exported corrector is transferred, not refitted, whenever
its upstream path changes.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import load_bundle
from .checkpoints import matches_pipeline_identity, pipeline_identity, save_checkpoint
from .config import SUBSETS
from .data import WindowDataset
from .pooled import load_pooled_training, sha, split_pooled_engines, write_json
from .training import (device_for, evaluate, exposure_metrics, loader, metrics,
                       move_batch, objectives, seed_all)


def checked_frames(root, entry, config):
    """Reconstruct saved scopes and hashes using original training files only."""
    frame, hashes = load_pooled_training(Path(root) / "data", config)
    if hashes != entry["source_hashes"]:
        raise ValueError("Training data hashes disagree with the frozen pooled bundle.")
    groups = split_pooled_engines(frame, config["split_seed"])
    if groups != entry["partitions"]:
        raise ValueError("Saved engine partitions do not match the original pooled split.")
    scopes = [set(groups[k]) for k in ["fit", "validation", "calibration", "test"]]
    if any(scopes[i] & scopes[j] for i in range(4) for j in range(i)):
        raise ValueError("Fitting, selection, calibration and test scopes must be disjoint.")
    if scopes[0] != set(entry["scaler"]["fit_units"]):
        raise ValueError("The diagnostic must retain the original fitting-only scaler.")
    fit = frame[frame.unit.isin(scopes[0])].copy()
    validation = frame[frame.unit.isin(scopes[1])].copy()
    identity = pipeline_identity(fit, validation, "POOLED", config, entry["selected_seed"])
    if not matches_pipeline_identity(entry["pipeline_identity"], identity):
        raise ValueError("Reconstructed fitting/selection data disagree with the training identity.")
    return fit, validation, hashes


def warm_intervention(original, checkpoint, entry):
    """Copy only matching warm recovery tensors; preserve predictor/support/corrector."""
    if not matches_pipeline_identity(checkpoint.get("identity"), entry["pipeline_identity"]):
        raise ValueError("The checkpoint is not from this frozen model's training run.")
    warm = checkpoint.get("warm_recovery")
    expected = {k for k in original.state_dict() if k.startswith(("detector.", "generator."))}
    if not isinstance(warm, dict) or set(warm) != expected:
        raise ValueError("A complete warm detector/generator state is required; other tensors are forbidden.")
    source = original.state_dict()
    for name, value in warm.items():
        if (not isinstance(value, torch.Tensor) or value.shape != source[name].shape or
                value.dtype != source[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid warm recovery tensor: {name}")
    # Legacy warm snapshots did not save thresholds. Reuse the exported ones
    # only if the detector is identical; otherwise the pairing is unverified.
    if any(not torch.equal(warm[k].cpu(), source[k].cpu()) for k in expected if k.startswith("detector.")):
        raise ValueError("Warm detector differs from the exported detector; matching thresholds are unavailable.")
    model = deepcopy(original)
    model.load_state_dict({**source, **warm}, strict=True)
    model.repair_enabled.fill_(True)  # Counterfactual in memory; never exported for deployment.
    return model


def identity_intervention(model):
    """Keep fault/quality routing but make generated finite sensor values identical."""
    result = deepcopy(model)
    with torch.no_grad():
        result.generator.decoder[-1].weight.zero_()
        result.generator.decoder[-1].bias.zero_()
    return result


def fit_predictor_pilot(model, fit, scaler, config, *, steps, seed):
    """Fixed-step predictor adaptation, with fixed warm repair/detector and no critic.

    This is a new fitting-only diagnostic, not a recovery of the discarded joint
    checkpoint or a full joint-training reproduction. Both repair and bypass RUL,
    clean anchors, branch supervision and paired-prefix losses use the existing
    objectives. The exported corrector is never used for fitting or updated.
    """
    seed_all(seed, config["cpu_threads"])
    model = deepcopy(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("predictor."))
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=config["joint_learning_rate"], weight_decay=.01)
    dataset = WindowDataset(fit, scaler, config, mode="train", seed=seed)
    history, sampled, epoch = [], set(), 0
    while len(history) < steps:
        dataset.set_epoch(epoch, progress=1.)
        batches = loader(dataset, config, shuffle=True, seed=seed + epoch)
        if not len(batches):
            raise ValueError("The fitting pilot has no batches.")
        for batch in batches:
            model.train(); model.detector.eval(); model.generator.eval()
            sampled.update(int(unit) for unit in batch["unit"])
            batch = move_batch(batch, next(model.parameters()).device)
            optimizer.zero_grad(set_to_none=True)
            loss, result = objectives(model, batch, critic=None)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite predictor-pilot loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            row = {"step": len(history) + 1, "epoch": epoch, "loss": float(loss.detach()),
                   "gradient_norm": float(norm.detach()),
                   **{f"loss_{k}": float(v) for k, v in result["loss_terms"].items()}}
            history.append(row)
            if len(history) == 1 or len(history) % 10 == 0 or len(history) == steps:
                print(f"Predictor-only pilot {len(history)}/{steps}: loss={row['loss']:.6f}", flush=True)
            if len(history) == steps:
                break
        epoch += 1
    return model.eval(), pd.DataFrame(history), sorted(sampled)


def path_specs(predictor):
    return [{"path": f"{predictor}_{'repair' if repair else 'bypass'}_{'corrected' if correction else 'base'}",
             "predictor": predictor, "repair": repair, "correction": correction,
             "corrector_status": ("off" if not correction else "original" if predictor == "saved" and not repair
                                  else "transferred_without_refitting")}
            for repair in [False, True] for correction in [False, True]]


def paired_engine_bootstrap(reference, intervention, *, draws=2000, seed=42):
    """Paired engine-balanced RMSE difference; resample engines, never prefixes."""
    keys = ["unit", "cycle", "family", "replicate"]
    if reference.duplicated(keys).any() or intervention.duplicated(keys).any():
        raise ValueError("Paired diagnostic observations must be unique.")
    joined = reference.merge(intervention, on=keys, suffixes=("_ref", "_new"), validate="one_to_one")
    if len(joined) != len(reference) or len(joined) != len(intervention) or not len(joined):
        raise ValueError("Contrasts require identical engines, prefixes and corruption cases.")
    for column in ["target", "raw_target", "altered_entries", "valid_sensor_entries"]:
        if not np.array_equal(joined[f"{column}_ref"], joined[f"{column}_new"]):
            raise ValueError("Contrasts require identical targets and fault exposure.")
    joined["ref_mse"] = (joined.prediction_ref - joined.target_ref) ** 2
    joined["new_mse"] = (joined.prediction_new - joined.target_new) ** 2
    engine = joined.groupby("unit", sort=True)[["ref_mse", "new_mse"]].mean().to_numpy()
    delta = float(np.sqrt(engine[:, 1].mean()) - np.sqrt(engine[:, 0].mean()))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(engine), size=(draws, len(engine)))
    means = engine[indices].mean(1)
    distribution = np.sqrt(means[:, 1]) - np.sqrt(means[:, 0])
    return {"engines": len(engine), "observations": len(joined),
            "engine_balanced_RMSE_delta": delta, "bootstrap_95_low": float(np.quantile(distribution, .025)),
            "bootstrap_95_high": float(np.quantile(distribution, .975)), "bootstrap_draws": draws,
            "bootstrap_scope": "exploratory paired engine bootstrap; fixed training seed; no multiplicity adjustment"}


def summarize_case(records, report, spec, entry, config, family, replicate):
    rows = []
    for source in ["POOLED", *SUBSETS]:
        selected = records if source == "POOLED" else records[records.source_subset.eq(source)]
        if not len(selected):
            continue
        counts = selected.intact_sensor_entries.to_numpy()
        intact = selected.linear_intact_repair_RMSE.to_numpy()
        supported = counts > 0
        healthy = float(np.sqrt(np.sum(intact[supported] ** 2 * counts[supported]) / counts[supported].sum())) if supported.any() else None
        radius = entry["calibration"]["protocols"][f"capped_{config['train_rul_cap']:g}"]["radius"]
        covered = (selected.prediction - selected.target).abs() <= radius if radius is not None else np.ones(len(selected), bool)
        errors = (selected.prediction - selected.target).abs()
        maximum = errors.groupby(selected.unit).max()
        reference = ({"RMSE": entry["pretrained_clean_RMSE"], "NASA_Score_Mean": entry["pretrained_clean_NASA"]}
                     if source == "POOLED" else entry["pretrained_clean_by_source"][source])
        measured = metrics(selected.prediction, selected.target)
        rows.append({**spec, "family": family, "severity": 0. if family == "clean" else .2,
            "replicate": replicate, "subset": source, "engines": selected.unit.nunique(),
            **measured, **exposure_metrics(selected),
            "linear_intact_repair_RMSE": healthy,
            "linear_intact_window_RMSE_max": float(intact[supported].max()) if supported.any() else None,
            "absolute_error_p90": float(errors.quantile(.9)), "engine_max_error_p90": float(maximum.quantile(.9)),
            "reference_radius_coverage": float(np.mean(covered)),
            "reference_radius_engine_simultaneous_coverage": float(covered.groupby(selected.unit).all().mean()) if isinstance(covered, pd.Series) else 1.,
            "interval_scope": "unchanged exported path; descriptive coverage on reused selection engines" if spec["path"] == "saved_bypass_corrected" and family == "clean"
                              else "transferred reference radius only; this path/corruption has no calibration guarantee",
            "clean_RMSE_guard_pass": measured["RMSE"] <= reference["RMSE"] * (1 + config["max_clean_rmse_increase"]) if family == "clean" else None,
            "clean_NASA_guard_pass": measured["NASA_Score_Mean"] <= reference["NASA_Score_Mean"] * (1 + config["max_clean_nasa_increase"]) + 1e-12 if family == "clean" else None,
            **({k: report.get(k) for k in ["linear_reconstruction_ratio", "fault_false_positive_rate", "fault_recall"]} if source == "POOLED" else {})})
    return rows


def write_contrasts(predictions, specs, output, draws):
    pairs = []
    for predictor in sorted({x["predictor"] for x in specs}):
        for correction in ["base", "corrected"]:
            pairs.append((f"{predictor}_bypass_{correction}", f"{predictor}_repair_{correction}", "enable_warm_repair"))
        for path in ["bypass", "repair"]:
            pairs.append((f"{predictor}_{path}_base", f"{predictor}_{path}_corrected", "enable_exported_corrector"))
    if any(s["predictor"] == "pilot" for s in specs):
        pairs.append(("saved_bypass_base", "pilot_bypass_base", "new_predictor_only_corruption_pilot"))
    rows = []
    for before, after, effect in pairs:
        for family in ["clean", "stuck", "mixed"]:
            a = predictions[predictions.path.eq(before) & predictions.family.eq(family)]
            b = predictions[predictions.path.eq(after) & predictions.family.eq(family)]
            for subset in ["POOLED", *SUBSETS]:
                aa = a if subset == "POOLED" else a[a.source_subset.eq(subset)]
                bb = b if subset == "POOLED" else b[b.source_subset.eq(subset)]
                if len(aa):
                    rows.append({"reference": before, "intervention": after, "effect": effect, "family": family,
                                 "subset": subset, **paired_engine_bootstrap(aa, bb, draws=draws)})
    pd.DataFrame(rows).to_csv(output / "paired_contrasts.csv", index=False)


def plot_summary(table, output):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pooled = table[table.subset.eq("POOLED")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, family in zip(axes, ["clean", "stuck", "mixed"]):
        values = pooled[pooled.family.eq(family)].groupby("path", sort=False).RMSE.mean().iloc[::-1]
        ax.barh(values.index, values.values)
        ax.set(title=family.capitalize(), xlabel="Capped RMSE (cycles)")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Model-selection engines only; saved-weight interventions and optional short pilot")
    fig.savefig(output / "validation_paths.png", dpi=160)
    plt.close(fig)


def run_diagnostic(project_root, source_run, output_dir, *, device="cpu", pilot_steps=0,
                   batch_size=32, replicates=3, bootstrap_draws=2000, seed=20261101, plot=True):
    for name, value, minimum in [("pilot_steps", pilot_steps, 0), ("batch_size", batch_size, 1),
                                  ("replicates", replicates, 1), ("bootstrap_draws", bootstrap_draws, 1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    chosen = device_for({"device": device})
    root, source, output = [Path(x).resolve() for x in [project_root, source_run, output_dir]]
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate diagnostic directory outside the source run.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output directory must be empty; source evidence is never overwritten.")
    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 9 or manifest.get("status") != "complete":
        raise ValueError("Use a completed pooled format-9 run.")
    bundle_path = source / "grace_rul_ALL.pt"
    if sha(bundle_path) != manifest["frozen_bundle_sha256"]:
        raise ValueError("Frozen bundle hash disagrees with its run manifest.")
    bundle, loaded = load_bundle(bundle_path, device=str(chosen))
    item = loaded["POOLED"]
    entry, original, scaler = item["entry"], item["model"], item["scaler"]
    if bool(original.repair_enabled):
        raise ValueError("This counterfactual diagnostic expects an exported bypass model.")
    original_config = dict(entry["training_config"])
    fit, validation, data_hashes = checked_frames(root, entry, original_config)
    checkpoint_path = source / "checkpoints" / f"pipeline_POOLED_{entry['selected_seed']}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not checkpoint.get("stage_complete"):
        raise ValueError("Use a completed epoch checkpoint; a live unselected training state needs a separate study.")
    if any(not torch.equal(value.cpu(), checkpoint["model"][name].cpu())
           for name, value in original.state_dict().items() if name.startswith("predictor.")):
        raise ValueError("Exported predictor differs from the checkpoint's accepted predictor.")
    warm = warm_intervention(original, checkpoint, entry)
    config = {**original_config, "device": str(chosen), "batch_size": batch_size}
    original.config = dict(config); warm.config = dict(config)
    seed_all(seed, config["cpu_threads"])
    watched = [bundle_path, checkpoint_path, manifest_path, *[root / "data" / name for name in data_hashes]]
    before_hashes = {str(path): sha(path) for path in watched}
    output.mkdir(parents=True, exist_ok=True)
    info = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Exploratory model-selection diagnostic; no production selection, full sensor matrix, recalibration or test evaluation.",
        "device": str(chosen), "torch_version": torch.__version__, "training_seed": entry["selected_seed"],
        "diagnostic_seed": seed, "pilot_steps": pilot_steps, "batch_size": batch_size, "replicates": replicates,
        "fit_units": sorted(int(x) for x in fit.unit.unique()),
        "selection_units": sorted(int(x) for x in validation.unit.unique()),
        "calibration_and_test_engines_evaluated": False, "official_files_opened": False,
        "source_hashes": before_hashes, "training_files_read": list(data_hashes),
        "data_read_scope": "Original train files contain all engine partitions; only fit and selection rows construct datasets.",
        "corrector_scope": "Exported OOF corrector; transferred without refitting when predictor or repair path changes.",
        "interval_scope": "Fixed exported radius measured descriptively; altered paths and faults have no new calibration guarantee.",
        "uncertainty_scope": "Paired engine bootstrap on reused selection engines; not independent confirmation or training-seed uncertainty.",
        "quick_source": bool(bundle["config"]["quick_run"])}
    write_json(output / "manifest.json", info)
    selected = [(row["stage"], int(row["epoch"])) for row in checkpoint["history"] if row.get("checkpoint_selected")]
    same_predictor = all(torch.equal(value.cpu(), checkpoint["best"][name].cpu())
                         for name, value in checkpoint["model"].items() if name.startswith("predictor."))
    inventory = {"checkpoint_stage": checkpoint["stage"], "stage_complete": checkpoint["stage_complete"],
        "accepted_upstream_checkpoints": selected, "checkpoint_predictor_equals_accepted_best": same_predictor,
        "historic_rejected_joint_predictor_available": False,
        "historic_joint_scope": "This diagnostic does not reconstruct discarded joint weights or infer their predictions from logs.",
        "saved_generator_decoder_norm": float(original.generator.decoder[-1].weight.detach().norm()),
        "warm_generator_decoder_norm": float(warm.generator.decoder[-1].weight.detach().norm()),
        "warm_recovery_score": checkpoint.get("warm_recovery_score"),
        "exported_repair_enabled": bool(original.repair_enabled),
        "exported_corrector_bound": float(original.correction_bound)}
    write_json(output / "state_inventory.json", inventory)
    groups = [("saved", warm)]
    if pilot_steps:
        print(f"Fitting predictor-only pilot on {fit.unit.nunique()} eligible fitting engines; selection remains separate.", flush=True)
        pilot, history, sampled = fit_predictor_pilot(warm, fit, scaler, config, steps=pilot_steps, seed=seed)
        history.to_csv(output / "pilot_history.csv", index=False)
        save_checkpoint(output / "predictor_pilot.pt", {"kind": "diagnostic_predictor_only_not_deployable",
            "source_bundle_sha256": before_hashes[str(bundle_path)], "pipeline_identity": entry["pipeline_identity"],
            "steps": pilot_steps, "seed": seed, "sampled_fit_units": sampled, "model": pilot.state_dict()})
        info.update(pilot_sampled_fit_units=sampled,
            pilot_scope="New fixed-step predictor-only adaptation; warm detector/generator/support fixed; no critic, OOF refit or checkpoint selection.")
        groups.append(("pilot", pilot))
    metadata = validation.groupby("unit")["source_subset"].first()
    all_records, all_rows, specs = [], [], []
    cases = [("clean", 0)] + [(family, repeat) for family in ["stuck", "mixed"] for repeat in range(replicates)]
    for predictor, model in groups:
        current = path_specs(predictor)
        specs.extend(current)
        identity = identity_intervention(model)
        identity_spec = {"path": f"{predictor}_identity_quality_probe", "predictor": predictor,
                         "repair": True, "correction": False, "corrector_status": "off"}
        for spec, evaluated in [*[(spec, model) for spec in current], (identity_spec, identity)]:
            for family, repeat in cases:
                if spec is identity_spec and family != "clean":
                    continue
                dataset = WindowDataset(validation, scaler, config, mode="validation", family=family,
                    severity=0. if family == "clean" else .2, seed=config["validation_seed"] + repeat)
                with torch.no_grad():
                    report, records = evaluate(evaluated, dataset, config, fault_thresholds=entry["thresholds"],
                        capture_recovery_tails=True, use_repair=spec["repair"], use_correction=spec["correction"])
                if set(records.unit) != set(info["selection_units"]):
                    raise ValueError("Evaluation did not cover exactly the held selection engines.")
                records["source_subset"] = records.unit.map(metadata)
                all_rows.extend(summarize_case(records, report, spec, entry, config, family, repeat))
                records["path"], records["family"], records["replicate"] = spec["path"], family, repeat
                all_records.append(records)
                print(f"{spec['path']} / {family} r{repeat}: capped RMSE={report['RMSE']:.4f}", flush=True)
    predictions, table = pd.concat(all_records, ignore_index=True), pd.DataFrame(all_rows)
    predictions.to_csv(output / "predictions.csv.gz", index=False)
    table.to_csv(output / "metrics.csv", index=False)
    maxima = predictions.assign(absolute_error=(predictions.prediction-predictions.target).abs()).groupby(
        ["path", "family", "replicate", "source_subset", "unit"], as_index=False).agg(
        max_prefix_error=("absolute_error", "max"), mean_absolute_error=("absolute_error", "mean"), prefixes=("cycle", "size"))
    maxima.to_csv(output / "engine_errors.csv", index=False)
    write_contrasts(predictions, specs, output, bootstrap_draws)
    if plot:
        plot_summary(table, output)
    after_hashes = {str(path): sha(path) for path in watched}
    if before_hashes != after_hashes:
        raise RuntimeError("Source evidence changed during the diagnostic; results must not be treated as frozen.")
    info.update(status="complete", source_files_unchanged=True, paths=specs,
        selection_prefixes_per_case=len(predictions[predictions.path.eq("saved_bypass_base") & predictions.family.eq("clean")]),
        historic_joint_predictor_evaluated=False, bootstrap_draws=bootstrap_draws,
        identity_probe_scope="Clean inputs only; zero generator residual isolates quality/routing changes, allowing floating-point interpolation roundoff.")
    write_json(output / "manifest.json", info)
    (output / "README.md").write_text(
        "# Validation-only pooled path diagnostic\n\n"
        "Only the saved fitting/selection engines participate. Calibration/internal-test engines and official test files are excluded from fitting/evaluation.\n\n"
        "The saved paths use the exported predictor plus the separately retained warm generator. They are inference interventions, not retrained component ablations. "
        "Discarded historical joint predictors are unavailable. Any pilot paths are new fixed-step predictor-only fitting experiments.\n\n"
        "The existing corrector is transferred whenever its upstream path changes. Its effect is not evidence about an independently refitted OOF corrector. "
        "Coverage columns apply the exported radius descriptively; no new calibration is fitted or guaranteed for changed paths/faults. "
        "Engine bootstrap intervals describe paired selection-engine variability, not independent test confirmation or multiple training seeds.\n\n"
        "Identity quality probes zero the generator residual on clean inputs while retaining repair-quality routing. "
        "No production artifact, checkpoint-selection threshold or guard budget is changed. Read state_inventory.json and manifest.json before interpreting metrics.csv.\n")
    return output, table
