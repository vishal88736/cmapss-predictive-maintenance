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
POLICIES = {"engine_prefix_max", "random_prefix"}


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


def calibrate_endpoint_intervals(model, frame, scaler, config, seed, *, evaluator=None):
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
                            mode="train" if policy == "engine_prefix_max" else "calibration",
                            seed=seed, family="clean")
    if evaluator is None:
        from .training import evaluate
        evaluator = evaluate
    _, records = evaluator(model, dataset, evaluation_config)
    records = _checked_records(records, dataset)

    primary = f"capped_{cap:g}"
    assumption = ("The fitted model and preprocessing are fixed independently of calibration engines; "
                  "complete calibration and future engine trajectories are exchangeable. "
                  "This assumption is not established by C-MAPSS benchmark membership.")
    if policy == "engine_prefix_max":
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
    for protocol, column, convention in [
            (primary, "target", f"min(raw_RUL, {cap:g})"),
            ("official_uncapped", "raw_target", "raw_RUL")]:
        errors = np.abs(records[column].to_numpy() - records.prediction.to_numpy())
        scores = pd.Series(errors, index=records.index).groupby(records.unit).max().sort_index()
        if len(scores) != len(units):
            raise ValueError("Calibration scores lost an independent engine.")
        radius = split_conformal_quantile(scores.to_numpy(), alpha)
        rank = conformal_rank(len(scores), alpha)
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
