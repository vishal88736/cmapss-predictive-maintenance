"""Actual Kernel SHAP and LIME explanations over coherent sensor/time blocks."""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from .config import SENSORS, SENSOR_NAMES


def explain_window(model, x, valid, background, background_valid, target_sensor=None, samples=256, seed=42, time_blocks=3):
    """Explain final RUL and optionally a suspect sensor's finite-fault score.

    Coalition features choose observed versus reference temporal blocks. Settings,
    actual missingness and padding stay fixed. Every evaluation reruns the pipeline.
    """
    import shap
    from lime.lime_base import LimeBase

    model.eval()
    device = next(model.parameters()).device
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    valid = torch.as_tensor(valid, dtype=torch.bool, device=device)
    background = torch.as_tensor(background, dtype=torch.float32, device=device)
    background_valid = torch.as_tensor(background_valid, dtype=torch.bool, device=device)
    positions = torch.where(valid)[0].cpu().numpy()
    if not len(positions) or time_blocks < 1 or samples < 1:
        raise ValueError("Explanations require observed positions and positive block/sample counts.")
    blocks = [(j, block) for j in range(21) for block in np.array_split(positions, min(time_blocks, len(positions))) if len(block)]
    names = [f"{SENSORS[j]} ({SENSOR_NAMES[j]}), window positions {int(p[0]) + 1}–{int(p[-1]) + 1}" for j, p in blocks]
    # Match the observed setting history. Prefer a complete reference horizon;
    # short references are resampled in time, never cyclically repeated.
    candidates = []
    for index in range(len(background)):
        real = background[index][background_valid[index]]
        if not len(real):
            continue
        short = len(real) < len(positions)
        real = real[torch.linspace(0, len(real) - 1, len(positions), device=device).round().long()] if short else real[-len(positions):]
        distance = float((real[:, :3] - x[valid, :3]).square().mean())
        candidates.append((short, distance, index, real))
    nearest = sorted(candidates, key=lambda c: c[:3])[:3]
    replacements = []
    for _, _, _, real in nearest:
        replacement = x.clone()
        replacement[valid, 3:] = real[:, 3:]
        replacements.append(replacement)
    if not replacements:
        raise ValueError("The bundle has no fitting-data explanation background.")
    finite = torch.isfinite(x[..., 3:]) & valid[:, None]
    targets = [None] + ([] if target_sensor is None else [int(target_sensor)])

    def evaluate_coalitions(coalitions):
        coalitions = np.asarray(coalitions)
        all_predictions = []
        for replacement in replacements:
            predictions = []
            for start in range(0, len(coalitions), 32):
                z = coalitions[start:start + 32]
                inputs = x.unsqueeze(0).repeat(len(z), 1, 1)
                for block_index, (sensor, pos) in enumerate(blocks):
                    rows = np.flatnonzero(z[:, block_index] < .5)
                    if not len(rows):
                        continue
                    for row in rows:
                        locations = torch.as_tensor(pos, device=device)
                        locations = locations[finite[locations, sensor]]
                        inputs[row, locations, 3 + sensor] = replacement[locations, 3 + sensor]
                with torch.no_grad():
                    out = model(inputs, valid.unsqueeze(0).expand(len(z), -1))
                    values = [out["prediction"]]
                    if target_sensor is not None:
                        observed = finite[:, target_sensor]
                        values.append(out["fault_scores"][:, observed, target_sensor].mean(1) if bool(observed.any()) else torch.zeros(len(z), device=device))
                    predictions.append(torch.stack(values, -1).cpu().numpy())
            all_predictions.append(np.concatenate(predictions))
        return np.mean(all_predictions, axis=0)

    n = len(blocks)
    rng = np.random.default_rng(seed)
    neighborhood = rng.integers(0, 2, (max(samples, 64), n)).astype(float)
    neighborhood[0] = 1.
    distances_lime = np.sqrt(np.mean((1 - neighborhood) ** 2, axis=1))
    labels = evaluate_coalitions(neighborhood)
    kernel = lambda d: np.exp(-(d ** 2) / .75 ** 2)
    lime = LimeBase(kernel_fn=kernel, random_state=seed)
    result = {"policy": "coherent observed/reference temporal blocks; actual missingness and settings fixed; backgrounds from fitting engines",
              "samples": int(max(samples, 64)), "seed": seed, "targets": [],
              "background_indices": [int(row[2]) for row in nearest],
              "background_setting_distances": [row[1] for row in nearest]}
    for column, sensor in enumerate(targets):
        function = lambda z, col=column: evaluate_coalitions(z)[:, col]
        # KernelExplainer samples with NumPy's global RNG; restore it for callers.
        random_state = np.random.get_state()
        try:
            np.random.seed(seed)
            explainer = shap.KernelExplainer(function, np.zeros((1, n)), feature_names=names)
            shap_values = np.asarray(explainer.shap_values(np.ones((1, n)), nsamples=max(samples, 64), l1_reg=0., silent=True)).reshape(-1)
        finally:
            np.random.set_state(random_state)
        cutoff = max(2, int(len(neighborhood) * .75))
        regressor = Ridge(alpha=1.)
        intercept, coefficients, training_fidelity, local_prediction = lime.explain_instance_with_data(
            neighborhood[:cutoff], labels[:cutoff, column:column + 1], distances_lime[:cutoff], 0,
            num_features=n, feature_selection="none", model_regressor=regressor)
        fidelity = r2_score(labels[cutoff:, column], regressor.predict(neighborhood[cutoff:]), sample_weight=kernel(distances_lime[cutoff:]))
        lime_values = np.zeros(n)
        for index, coefficient in coefficients:
            lime_values[index] = coefficient
        blocks_report = [{"sensor": SENSORS[j], "name": SENSOR_NAMES[j], "window_start": int(pos[0]) + 1,
                          "window_end": int(pos[-1]) + 1, "shap": float(shap_values[k]), "lime": float(lime_values[k])}
                         for k, (j, pos) in enumerate(blocks)]
        aggregate = [{"sensor": SENSORS[j], "name": SENSOR_NAMES[j],
            "shap": float(sum(r["shap"] for r in blocks_report if r["sensor"] == SENSORS[j])),
            "lime": float(sum(r["lime"] for r in blocks_report if r["sensor"] == SENSORS[j]))} for j in range(21)]
        shap_ranks = np.abs([row["shap"] for row in aggregate])
        lime_ranks = np.abs([row["lime"] for row in aggregate])
        agreement = float(spearmanr(shap_ranks, lime_ranks).statistic) if shap_ranks.std() > 0 and lime_ranks.std() > 0 else None
        result["targets"].append({"target": "final_rul" if sensor is None else "finite_fault_score", "sensor": None if sensor is None else SENSORS[sensor],
            "units": "cycles" if sensor is None else "detector score", "prediction": float(labels[0, column]),
            "shap_baseline": float(np.asarray(explainer.expected_value)), "lime_intercept": float(intercept),
            "lime_training_fidelity": float(training_fidelity) if np.isfinite(training_fidelity) else None,
            "lime_holdout_fidelity": float(fidelity) if np.isfinite(fidelity) else None,
            "shap_local_accuracy_error": float(abs(float(np.asarray(explainer.expected_value)) + shap_values.sum() - labels[0, column])),
            "lime_local_error": float(abs(float(np.asarray(local_prediction).item()) - labels[0, column])),
            "shap_lime_absolute_rank_correlation": agreement,
            "sensors": sorted(aggregate, key=lambda r: abs(r["shap"]), reverse=True), "blocks": blocks_report})
    return result


def explain_dataset(model, dataset, background, background_valid, examples=12, repeats=3, samples=256, seed=42):
    """Aggregate actual frozen-model explanations over distinct held-out engines."""
    from .data import select_background_indices

    if repeats < 1:
        raise ValueError("Explanation repeats must be positive.")
    indices, metadata = select_background_indices(dataset, examples, seed)
    if not indices:
        raise ValueError("Global explanations require held-out endpoints.")
    if len({row["unit"] for row in metadata}) != len(metadata):
        raise ValueError("Global explanations require one endpoint per engine.")
    cases, shap_vectors, lime_vectors = [], [], []
    for index, meta in zip(indices, metadata):
        sample = dataset[index]
        repeat_reports, shap_repeats, lime_repeats = [], [], []
        for repeat in range(repeats):
            explanation = explain_window(model, sample["x"], sample["valid"], background, background_valid,
                samples=samples, seed=seed + repeat, time_blocks=3)
            target = explanation["targets"][0]
            sensors = {row["sensor"]: row for row in target["sensors"]}
            shap_repeats.append([sensors[name]["shap"] for name in SENSORS])
            lime_repeats.append([sensors[name]["lime"] for name in SENSORS])
            repeat_reports.append(explanation)
        shap_vectors.append(shap_repeats); lime_vectors.append(lime_repeats)
        cases.append({**meta, "active_fault": bool(sample["altered"].any()), "repeats": repeat_reports})
    shap_vectors, lime_vectors = np.asarray(shap_vectors), np.asarray(lime_vectors)
    importance = np.abs(shap_vectors).mean((0, 1))
    lime_importance = np.abs(lime_vectors).mean((0, 1))
    stability = shap_vectors.std(1, ddof=1).mean(0) if repeats > 1 else [None] * 21
    between = np.abs(shap_vectors).mean(1).std(0, ddof=1) if len(cases) > 1 else [None] * 21
    global_sensors = [{"sensor": name, "name": SENSOR_NAMES[j], "mean_abs_shap": float(importance[j]),
        "between_engine_abs_shap_std": None if between[j] is None else float(between[j]),
        "mean_within_engine_shap_std": None if stability[j] is None else float(stability[j]),
        "mean_abs_lime_coefficient": float(lime_importance[j])} for j, name in enumerate(SENSORS)]
    return {"scope": "frozen model; held-out endpoints; equal engine weighting; SHAP and LIME reported separately",
            "engine_count": len(cases), "repeats": repeats, "samples": samples,
            "global_sensors": sorted(global_sensors, key=lambda row: row["mean_abs_shap"], reverse=True), "cases": cases}


def run_explanation_study(project_root, bundle_path, output_dir=None, *, subsets=None, examples=None,
                          repeats=None, samples=None, family="clean", device="cpu"):
    """Write genuine per-case explanations, global importance and seed stability."""
    import json
    from pathlib import Path
    import pandas as pd
    from .artifacts import load_bundle
    from .data import WindowDataset, load_subset

    bundle, loaded = load_bundle(bundle_path, device=device)
    requested = list(loaded) if subsets is None else list(subsets)
    if not requested or set(requested) - set(loaded):
        raise ValueError("Requested explanation subsets must exist in the trained bundle.")
    if family not in {"clean", "mixed", "bias", "noise"}:
        raise ValueError("Unsupported explanation fault family.")
    output = Path(output_dir) if output_dir else Path(bundle_path).parent / "explanations"
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for subset in requested:
        item = loaded[subset]
        config = item["model"].config
        _, test, truths = load_subset(Path(project_root) / "data", subset)
        if bundle["config"]["quick_run"]:
            test = test[test.unit.isin(sorted(test.unit.unique())[:3])]
        dataset = WindowDataset(test, item["scaler"], config, mode="test", family=family,
            seed=config["validation_seed"] + 100000, truths=truths)
        report = explain_dataset(item["model"], dataset, item["entry"]["background"], item["entry"]["background_valid"],
            examples=examples if examples is not None else config.get("explanation_examples", 12),
            repeats=repeats if repeats is not None else config.get("explanation_repeats", 3),
            samples=samples if samples is not None else config["explanation_samples"], seed=config["seed"])
        report.update({"subset": subset, "family": family, "quick_run": bundle["config"]["quick_run"],
                       "selected_seed": item["entry"]["selected_seed"], "model_bundle": str(Path(bundle_path).resolve())})
        (output / f"global_explanation_{subset}_{family}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        summary.extend({"subset": subset, "family": family, "engine_count": report["engine_count"],
                        "repeats": report["repeats"], **row} for row in report["global_sensors"])
        print(f"Explained {subset}: {report['engine_count']} engines, {report['repeats']} seeds", flush=True)
    frame = pd.DataFrame(summary)
    frame.to_csv(output / f"global_sensor_importance_{family}.csv", index=False)
    return frame
