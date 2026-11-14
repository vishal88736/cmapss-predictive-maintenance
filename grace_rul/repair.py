"""Format-10 post-hoc calibration and reporting helpers.

These functions operate on held-out records only. They never read official test
targets during selection: the caller passes validation/calibration records and
freezes the returned parameters before test evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rmse(error):
    return float(np.sqrt(np.mean(np.square(error))))


def fit_plateau_calibration(records, *, knee=100., knees=None, alphas=None,
                            max_capped_increase=0.0, cap=125.):
    """Choose a one-parameter monotone stretch for the censored RUL plateau.

    The predictor is trained against ``min(RUL, cap)`` and therefore cannot
    distinguish a 120-cycle engine from a 250-cycle engine. A positive stretch
    above ``knee`` is the smallest monotone correction that can recover that
    information. The stretch is accepted only when it improves raw RMSE and
    does not increase capped RMSE by more than ``max_capped_increase``.
    """
    frame = pd.DataFrame(records)
    if frame.empty or not {"prediction", "raw_target"}.issubset(frame.columns):
        raise ValueError("Plateau calibration needs prediction and raw_target records.")
    prediction = frame.prediction.to_numpy(float)
    raw = frame.raw_target.to_numpy(float)
    capped = np.minimum(raw, float(cap))
    if alphas is None:
        alphas = np.linspace(0., 2., 41)
    if knees is not None:
        knee_candidates = [float(k) for k in knees]
    elif isinstance(knee, (list, tuple, np.ndarray)):
        knee_candidates = [float(k) for k in knee]
    else:
        knee_candidates = [float(knee)]
    baseline_capped = _rmse(prediction - capped)
    baseline_raw = _rmse(prediction - raw)
    best = {"alpha": 0.0, "knee": float(knee_candidates[0]), "raw_RMSE": baseline_raw,
            "capped_RMSE": baseline_capped, "N": int(len(frame))}
    for k in knee_candidates:
        for alpha in np.asarray(alphas, float):
            adjusted = prediction + float(alpha) * np.maximum(prediction - float(k), 0.)
            raw_rmse = _rmse(adjusted - raw)
            capped_rmse = _rmse(adjusted - capped)
            row = {"alpha": float(alpha), "knee": float(k),
                   "raw_RMSE": raw_rmse, "capped_RMSE": capped_rmse,
                   "N": int(len(frame))}
            # Prefer the largest raw-error reduction that respects the capped guard.
            if capped_rmse <= baseline_capped + float(max_capped_increase):
                if (row["raw_RMSE"] < best["raw_RMSE"] - 1e-9 or
                        (abs(row["raw_RMSE"] - best["raw_RMSE"]) <= 1e-9 and (row["alpha"] < best["alpha"] or row["knee"] > best["knee"]))):
                    best = row
    best["baseline_raw_RMSE"] = baseline_raw
    best["baseline_capped_RMSE"] = baseline_capped
    best["method"] = "monotone_plateau_stretch"
    best["selection_scope"] = "held validation engines only"
    return best


def apply_plateau_calibration(prediction, calibration):
    prediction = np.asarray(prediction, float)
    if not isinstance(calibration, dict) or float(calibration.get("alpha", 0.)) <= 0:
        return prediction.copy()
    knee = float(calibration.get("knee", 100.))
    return prediction + float(calibration["alpha"]) * np.maximum(prediction - knee, 0.)


def engine_bootstrap(records, *, draws=2000, seed=20261114, cap=125.):
    """Engine-level bootstrap for capped and raw RMSE.

    Prefixes are not independent observations, so resampling rows would
    understate uncertainty. This resamples whole engines with replacement.
    """
    frame = pd.DataFrame(records)
    if frame.empty or "unit" not in frame:
        raise ValueError("Engine bootstrap needs per-engine records.")
    rng = np.random.default_rng(seed)
    units = np.asarray(sorted(frame.unit.unique()))
    grouped = {unit: rows for unit, rows in frame.groupby("unit")}
    capped_values, raw_values = [], []
    for _ in range(int(draws)):
        chosen = rng.choice(units, size=len(units), replace=True)
        rows = pd.concat([grouped[unit] for unit in chosen], ignore_index=True)
        raw = rows.raw_target.to_numpy(float)
        prediction = rows.prediction.to_numpy(float)
        capped_values.append(_rmse(prediction - np.minimum(raw, float(cap))))
        raw_values.append(_rmse(prediction - raw))
    return {
        "engine_N": int(len(units)),
        "draws": int(draws),
        "capped_RMSE": {"mean": float(np.mean(capped_values)),
                        "p05": float(np.quantile(capped_values, .05)),
                        "p95": float(np.quantile(capped_values, .95))},
        "raw_RMSE": {"mean": float(np.mean(raw_values)),
                     "p05": float(np.quantile(raw_values, .05)),
                     "p95": float(np.quantile(raw_values, .95))},
    }


def long_rul_report(records, *, threshold=125., cap=125.):
    frame = pd.DataFrame(records)
    raw = frame.raw_target.to_numpy(float)
    prediction = frame.prediction.to_numpy(float)
    capped = np.minimum(raw, float(cap))
    long = raw > float(threshold)
    return {
        "N": int(len(frame)),
        "long_N": int(long.sum()),
        "long_fraction": float(long.mean()),
        "long_mean_prediction": float(prediction[long].mean()) if long.any() else None,
        "long_mean_target": float(raw[long].mean()) if long.any() else None,
        "long_share_squared_error": float(np.square(prediction[long] - raw[long]).sum() /
                                          max(np.square(prediction - raw).sum(), 1e-12)),
        "raw_RMSE": _rmse(prediction - raw),
        "capped_RMSE": _rmse(prediction - capped),
        "long_raw_RMSE": _rmse(prediction[long] - raw[long]) if long.any() else None,
        "short_raw_RMSE": _rmse(prediction[~long] - raw[~long]) if (~long).any() else None,
    }


def per_engine_tail(records):
    frame = pd.DataFrame(records)
    frame["error"] = frame.prediction - frame.raw_target
    grouped = frame.groupby("unit").agg(
        source_subset=("source_subset", "first") if "source_subset" in frame else ("unit", "first"),
        max_abs_error=("error", lambda values: float(np.abs(values).max())),
        rmse=("error", lambda values: _rmse(values)),
        mean_error=("error", "mean"),
    ).sort_values("max_abs_error", ascending=False)
    return {
        "engines": int(len(grouped)),
        "max_abs_error": float(grouped.max_abs_error.max()),
        "gt20": int((grouped.max_abs_error > 20).sum()),
        "gt40": int((grouped.max_abs_error > 40).sum()),
        "gt100": int((grouped.max_abs_error > 100).sum()),
        "worst": grouped.head(10).reset_index().to_dict("records"),
    }


def interval_band_report(records, *, cap=125.):
    """Coverage and width by observable RUL band, using saved interval columns."""
    frame = pd.DataFrame(records)
    required = {"raw_target", "interval_lower_capped", "interval_upper_capped"}
    if not required.issubset(frame.columns):
        raise ValueError("Interval report needs raw_target and capped bounds.")
    bands = [(0., 35., "critical_eol"), (35., 75., "mid_degradation"),
             (75., 125., "early_degradation"), (125., float("inf"), "plateau")]
    rows = []
    raw = frame.raw_target.to_numpy(float)
    for low, high, name in bands:
        mask = (raw >= low) & (raw < high)
        if not mask.any():
            continue
        covered = ((frame.interval_lower_capped.to_numpy(float)[mask] <= np.minimum(raw[mask], cap)) &
                   (frame.interval_upper_capped.to_numpy(float)[mask] >= np.minimum(raw[mask], cap)))
        rows.append({"band": name, "N": int(mask.sum()),
                     "coverage": float(covered.mean()),
                     "mean_width": float((frame.interval_upper_capped[mask] -
                                          frame.interval_lower_capped[mask]).mean())})
    return rows
