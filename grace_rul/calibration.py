"""Engine-disjoint split-conformal intervals with explicit target contracts.

Calibration engines contain complete run-to-failure trajectories. Their last
rows all have RUL zero, so those rows cannot stand in for censored deployment
endpoints. The default scores each independent engine by its worst absolute
error over *all* observed prefixes. The resulting bound is conservative and
simultaneous over prefixes, subject to exchangeability of whole trajectories.
"""

import math
from decimal import Decimal, ROUND_CEILING

import numpy as np
import pandas as pd

from .data import WindowDataset


METHOD = "split_conformal_absolute"
POLICIES = {"engine_prefix_max", "random_prefix", "observable_stratified", "observable_scaled_engine_max"}
DEFAULT_OBSERVABLE_BINS = [
    (0., 35., "critical_eol"),
    (35., 75., "mid_degradation"),
    (75., 125., "early_degradation"),
    (125., float("inf"), "plateau"),
]


def _validate_alpha(alpha):
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("Interval alpha must be a number strictly between zero and one.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Interval alpha must be strictly between zero and one.")
    return alpha


def conformal_rank(n, alpha):
    """Return ceil((n + 1) * (1 - alpha)), without binary rounding at integers."""
    alpha = _validate_alpha(alpha)
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Conformal calibration needs at least one independent engine.")
    return int((Decimal(int(n) + 1) * (Decimal(1) - Decimal(str(alpha))))
               .to_integral_value(rounding=ROUND_CEILING))


def split_conformal_quantile(scores, alpha):
    """Finite-sample order statistic; return infinity when its rank exceeds n.

    Scores must be one nonnegative value per independent engine. No percentile
    interpolation or replacement by the largest finite score is permitted.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Conformal scores must be a nonempty finite nonnegative vector.")
    rank = conformal_rank(len(scores), alpha)
    return float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else math.inf


def _checked_records(records, dataset):
    """Ensure evaluation covered exactly the declared calibration prefixes."""
    records = pd.DataFrame(records).copy()
    columns = ["unit", "cycle", "prediction", "target", "raw_target"]
    if any(name not in records for name in columns):
        raise ValueError("Calibration evaluation must return unit/cycle, predictions, and both target conventions.")
    if not len(records) or not np.isfinite(records[columns].to_numpy(float)).all():
        raise ValueError("Calibration predictions and targets must be nonempty and finite.")
    if records.duplicated(["unit", "cycle"]).any():
        raise ValueError("A calibration engine prefix was evaluated more than once.")
    expected = {}
    for engine_index, end in dataset.samples:
        engine = dataset.engines[engine_index]
        expected[(engine["unit"], int(engine["cycles"][end]))] = float(engine["targets"][end])
    observed = list(zip(records.unit, records.cycle))
    if len(records) != len(expected) or set(observed) != set(expected):
        raise ValueError("Calibration evaluation must cover every declared prefix of every held engine exactly once.")
    raw = np.asarray([expected[key] for key in observed])
    if not np.allclose(records.raw_target, raw, rtol=0, atol=1e-5) or not np.allclose(
            records.target, np.minimum(raw, dataset.config["train_rul_cap"]), rtol=0, atol=1e-5):
        raise ValueError("Calibration evaluation changed the capped or raw target convention.")
    if "altered_entries" in records and (records.altered_entries != 0).any():
        raise ValueError("These interval contracts require unmodified calibration references.")
    return records


def _observable_scales(records, target_column, floor):
    """Fit a positive error scale on selection records, before calibration."""
    error = records[target_column].to_numpy() - records.prediction.to_numpy()
    fallback = max(floor, float(np.sqrt(np.mean(error ** 2))))
    bins = []
    for lo, hi, name in DEFAULT_OBSERVABLE_BINS:
        mask = (records.prediction >= lo) & (records.prediction < hi)
        local = error[mask]
        scale = max(floor, float(np.sqrt(np.mean(local ** 2)))) if len(local) else fallback
        bins.append({"lo": float(lo), "hi": float(hi) if np.isfinite(hi) else None,
                     "name": name, "scale": scale, "scale_prefixes": int(mask.sum()),
                     "scale_engines": int(records.loc[mask, "unit"].nunique())})
    return bins


def _scale_for_prediction(predictions, bins):
    values = np.asarray(predictions, dtype=float)
    scales = np.full(values.shape, np.nan)
    for item in bins:
        high = math.inf if item["hi"] is None else item["hi"]
        scales[(values >= item["lo"]) & (values < high)] = item["scale"]
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Observable scales must cover every nonnegative prediction.")
    return scales


def calibrate_endpoint_intervals(model, frame, scaler, config, seed, *, evaluator=None, scale_frame=None):
    """Calibrate frozen predictions on held training engines, without fitting.

    ``evaluator(model, dataset, config)`` returns ``(report, records)`` and may be
    injected by training to avoid a circular import. The only data supplied here
    must be complete run-to-failure calibration trajectories, disjoint from all
    fitting and model-selection engines. No official test files are opened.

    ``engine_prefix_max`` evaluates every prefix and uses a maximum per engine;
    ``random_prefix`` retains one seeded prefix per engine and has only a
    matched-observation-policy marginal interpretation. The latter does not
    solve unknown endpoint selection and is not the default.
    """
    alpha = _validate_alpha(config["interval_alpha"])
    cap = float(config["train_rul_cap"])
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Training RUL cap must be finite and positive.")
    policy = config.get("calibration_policy", "engine_prefix_max")
    if policy not in POLICIES:
        raise ValueError(f"Unknown calibration policy: {policy!r}. Expected one of {sorted(POLICIES)}.")
    if frame.empty or not {"unit", "cycle"}.issubset(frame.columns):
        raise ValueError("Calibration needs nonempty engine trajectories with unit and cycle columns.")
    identifiers = frame[["unit", "cycle"]].to_numpy(float)
    if not np.isfinite(identifiers).all() or not (identifiers == identifiers.astype(np.int64)).all():
        raise ValueError("Calibration engine identifiers and cycles must be finite integers.")
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError("Calibration trajectories contain duplicate engine cycles.")
    units = sorted(int(unit) for unit in frame.unit.unique())
    fitted = (getattr(scaler, "state", None) or {}).get("fit_units", [])
    if set(units) & set(fitted):
        raise ValueError("Calibration engines overlap the scaler's fitting engines.")
    # Do not inherit quick-run sampling limits: they would omit prefixes or
    # entire engines and invalidate the declared simultaneous score.
    evaluation_config = {**config, "maximum_windows": None, "window_stride": 1,
                         "maximum_batches": None, "prefix_consistency_loss_weight": 0.}
    dataset = WindowDataset(frame.sort_values(["unit", "cycle"]), scaler, evaluation_config,
                            mode="train" if policy != "random_prefix" else "calibration",
                            seed=seed, family="clean")
    if evaluator is None:
        from .training import evaluate
        evaluator = evaluate
    _, records = evaluator(model, dataset, evaluation_config)
    records = _checked_records(records, dataset)

    scale_records, scale_units = None, []
    if policy == "observable_scaled_engine_max":
        if scale_frame is None or scale_frame.empty or not {"unit", "cycle"}.issubset(scale_frame):
            raise ValueError("Adaptive simultaneous intervals need separate selection engines for scale fitting.")
        scale_units = sorted(int(unit) for unit in scale_frame.unit.unique())
        if set(scale_units) & set(units):
            raise ValueError("Observable scale fitting must exclude every calibration engine.")
        if scale_frame.duplicated(["unit", "cycle"]).any():
            raise ValueError("Observable scale fitting contains duplicate engine cycles.")
        scale_dataset = WindowDataset(scale_frame.sort_values(["unit", "cycle"]), scaler, evaluation_config,
                                      mode="train", seed=seed, family="clean")
        _, scale_records = evaluator(model, scale_dataset, evaluation_config)
        scale_records = _checked_records(scale_records, scale_dataset)

    primary = f"capped_{cap:g}"
    assumption = ("The fitted model and preprocessing are fixed independently of calibration engines; "
                  "complete calibration and future engine trajectories are exchangeable. "
                  "This assumption is not established by C-MAPSS benchmark membership.")
    if policy == "observable_scaled_engine_max":
        scope = ("One maximum normalized residual per independent calibration engine over all prefixes and observable bins; "
                 "positive bin scales are fixed using selection engines only. Simultaneous clean-trajectory coverage "
                 "at 1-alpha requires exchangeable complete trajectories and independence from calibration. "
                 "No guarantee under fleet/censoring shift, corrupted inputs, or OOD conditions.")
        coverage_unit = "engine_trajectory_simultaneous_observable_scaled"
    elif policy == "observable_stratified":
        scope = ("Finite-sample split conformal with observable-prediction-stratified engine-maximum residual scores; "
                 "per-bin interpretation only; alpha is not allocated jointly across bins. "
                 "No guarantee under fleet/censoring shift, corrupted inputs, or OOD conditions.")
        coverage_unit = "engine_observable_stratified_prefix"
    elif policy == "engine_prefix_max":
        scope = ("Finite-sample split conformal with one maximum-prefix residual score per held engine; "
                 "simultaneous clean-prefix coverage requires exchangeable complete engine trajectories. "
                 "No guarantee under fleet/censoring shift, corrupted inputs, or OOD conditions.")
        coverage_unit = "engine_trajectory_simultaneous_prefix"
    else:
        scope = ("Finite-sample split conformal with one seeded clean prefix per held engine; "
                 "marginal endpoint coverage requires exchangeable engines and the same prefix-sampling policy. "
                 "Official test censoring is not established to match; no corrupted/OOD guarantee.")
        coverage_unit = "engine_random_prefix_marginal"
    result = {"method": METHOD, "policy": policy, "endpoint_policy": policy,
              "alpha": alpha, "nominal_coverage": 1 - alpha, "engines": len(units),
              "endpoints": len(records), "units": units, "seed": int(seed),
              "score_unit": "engine", "coverage_unit": coverage_unit,
              "primary_protocol": primary, "protocols": {}, "scope": scope,
              "exchangeability_assumption": assumption,
              "target_contract": "Separate capped and raw residual scores; a capped interval is not an uncapped guarantee."}
    if policy == "observable_scaled_engine_max":
        result.update(contract_revision=2, scale_units=scale_units,
                      scale_scope="Held selection engines; independent of calibration engines",
                      simultaneous_across_observable_bins=True)
    for protocol, column, convention in [
            (primary, "target", f"min(raw_RUL, {cap:g})"),
            ("official_uncapped", "raw_target", "raw_RUL")]:
        errors = np.abs(records[column].to_numpy() - records.prediction.to_numpy())
        scale_bins = None
        if policy == "observable_scaled_engine_max":
            floor = float(config.get("observable_scale_floor", 1.))
            if not np.isfinite(floor) or floor <= 0:
                raise ValueError("Observable scale floor must be finite and positive.")
            scale_bins = _observable_scales(scale_records, column, floor)
            errors = errors / _scale_for_prediction(records.prediction, scale_bins)
        scores = pd.Series(errors, index=records.index).groupby(records.unit).max().sort_index()
        if len(scores) != len(units):
            raise ValueError("Calibration scores lost an independent engine.")
        radius = split_conformal_quantile(scores.to_numpy(), alpha)
        rank = conformal_rank(len(scores), alpha)
        normalized_quantile = float(radius) if np.isfinite(radius) else None
        if scale_bins is not None:
            radius = radius * max(item["scale"] for item in scale_bins)
        finite_radius = float(radius) if np.isfinite(radius) else None
        contract = {"method": METHOD, "policy": policy, "target_convention": convention,
                    "target_column": column, "alpha": alpha, "engines": len(scores), "n": len(scores),
                    "rank": rank, "radius": finite_radius, "unbounded": finite_radius is None,
                    "lower_offset": -finite_radius if finite_radius is not None else None,
                    "upper_offset": finite_radius, "lower_bound": 0., "scope": scope,
                    "engine_scores": [{"unit": int(unit), "score": float(score)}
                                      for unit, score in scores.items()]}
        if column == "target":
            contract["target_cap"] = cap
        if scale_bins is not None:
            contract.update(normalized_quantile=normalized_quantile,
                            score_transform="absolute_error/selection_bin_RMSE",
                            scale_units=scale_units, scale_floor=floor,
                            simultaneous_across_observable_bins=True,
                            stratified_bins=[{**item,
                                "radius": normalized_quantile * item["scale"] if normalized_quantile is not None else None,
                                "unbounded": normalized_quantile is None} for item in scale_bins])
        if policy == "observable_stratified":
            stratified_bins = []
            for lo, hi, name in DEFAULT_OBSERVABLE_BINS:
                mask = (records.prediction >= lo) & (records.prediction < hi)
                sub_records = records[mask]
                if not sub_records.empty:
                    sub_errors = np.abs(sub_records[column].to_numpy() - sub_records.prediction.to_numpy())
                    sub_scores = pd.Series(sub_errors, index=sub_records.index).groupby(sub_records.unit).max().sort_index()
                    bin_radius = split_conformal_quantile(sub_scores.to_numpy(), alpha)
                    bin_finite_radius = float(bin_radius) if np.isfinite(bin_radius) else None
                    stratified_bins.append({
                        "lo": float(lo), "hi": float(hi) if np.isfinite(hi) else 1e9,
                        "name": name, "radius": bin_finite_radius, "engines": len(sub_scores),
                        "unbounded": bin_finite_radius is None
                    })
                else:
                    stratified_bins.append({
                        "lo": float(lo), "hi": float(hi) if np.isfinite(hi) else 1e9,
                        "name": name, "radius": None, "engines": 0, "unbounded": True
                    })
            contract["stratified_bins"] = stratified_bins
        result["protocols"][protocol] = contract
    # Preserve the primary-contract field names used by existing website code;
    # None explicitly means unbounded, not a measured zero-width interval.
    for key in ["rank", "n", "radius", "unbounded", "lower_offset", "upper_offset", "target_convention"]:
        result[key] = result["protocols"][primary][key]
    return result, records


def interval_bounds(predictions, calibration, protocol=None):
    """Apply the requested target contract, or historical signed offsets.

    Legacy artifacts keep their original offset behavior. Their capped-target
    offsets cannot be requested as an uncapped coverage contract. New artifacts
    require a separately calibrated protocol for each target convention.
    """
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Interval point predictions must be finite.")
    if "protocols" in calibration:
        protocol = calibration.get("primary_protocol") if protocol is None else protocol
        if protocol not in calibration["protocols"]:
            raise ValueError(f"No interval calibration for target protocol {protocol!r}.")
        contract = calibration["protocols"][protocol]
        if contract.get("method") != METHOD:
            raise ValueError("Unsupported conformal interval method.")
        if contract.get("unbounded"):
            if contract.get("radius") is not None:
                raise ValueError("Unbounded calibration must store a JSON-null radius.")
            return np.zeros_like(predictions), np.full_like(predictions, math.inf)
        radius = contract.get("radius")
        stratified_bins = contract.get("stratified_bins")
        if stratified_bins and contract.get("policy", calibration.get("policy")) in {"observable_stratified", "observable_scaled_engine_max"}:
            # JSON null means unbounded. Never replace an unsupported local
            # quantile with the global radius and claim the local guarantee.
            radii = np.full(predictions.shape, math.inf, dtype=float)
            for b in stratified_bins:
                high = math.inf if b["hi"] is None else b["hi"]
                mask = (predictions >= b["lo"]) & (predictions < high)
                if np.any(mask) and b.get("radius") is not None and np.isfinite(b["radius"]):
                    radii[mask] = b["radius"]
            lower = np.maximum(0., predictions - radii)
            return lower, np.maximum(lower, predictions + radii)
        if radius is None or not np.isfinite(radius) or radius < 0:
            raise ValueError("Conformal radius must be finite and nonnegative, or explicitly unbounded.")
        lower = np.maximum(0., predictions - radius)
        return lower, np.maximum(lower, predictions + radius)
    if protocol == "official_uncapped":
        raise ValueError("Legacy capped-target intervals have no uncapped calibration contract.")
    lower_offset, upper_offset = calibration["lower_offset"], calibration["upper_offset"]
    if not np.isfinite([lower_offset, upper_offset]).all() or lower_offset > upper_offset:
        raise ValueError("Legacy interval offsets must be finite and ordered.")
    lower = np.maximum(0., predictions + lower_offset)
    return lower, np.maximum(lower, predictions + upper_offset)
