"""Actual Kernel SHAP and LIME explanations over coherent sensor/time blocks."""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from .config import SENSORS, SENSOR_NAMES


def explain_window(model, x, valid, background, background_valid, target_sensor=None, samples=128, seed=42, time_blocks=3):
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
    blocks = [(j, block) for j in range(21) for block in np.array_split(positions, min(time_blocks, len(positions))) if len(block)]
    names = [f"{SENSORS[j]} ({SENSOR_NAMES[j]}), window positions {int(p[0]) + 1}–{int(p[-1]) + 1}" for j, p in blocks]
    # Compare regimes by standardized observed settings, not by an arbitrary cluster ID.
    distances = (background[:, -1, :3] - x[-1, :3]).square().sum(-1)
    nearest = distances.argsort()[:min(3, len(background))]
    replacements = []
    for index in nearest:
        real = background[index][background_valid[index]]
        real = real[torch.arange(len(positions), device=device).sub(len(positions)).remainder(len(real))]
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
              "samples": int(samples), "seed": seed, "targets": []}
    for column, sensor in enumerate(targets):
        function = lambda z, col=column: evaluate_coalitions(z)[:, col]
        # KernelExplainer samples with NumPy's global RNG; restore it for callers.
        random_state = np.random.get_state()
        try:
            np.random.seed(seed)
            explainer = shap.KernelExplainer(function, np.zeros((1, n)), feature_names=names)
            shap_values = np.asarray(explainer.shap_values(np.ones((1, n)), nsamples=max(samples, 64), l1_reg="num_features(10)", silent=True)).reshape(-1)
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
        result["targets"].append({"target": "final_rul" if sensor is None else "finite_fault_score", "sensor": None if sensor is None else SENSORS[sensor],
            "units": "cycles" if sensor is None else "detector score", "prediction": float(labels[0, column]),
            "shap_baseline": float(np.asarray(explainer.expected_value)), "lime_intercept": float(intercept),
            "lime_training_fidelity": float(training_fidelity), "lime_holdout_fidelity": float(fidelity),
            "sensors": sorted(aggregate, key=lambda r: abs(r["shap"]), reverse=True), "blocks": blocks_report})
    return result
