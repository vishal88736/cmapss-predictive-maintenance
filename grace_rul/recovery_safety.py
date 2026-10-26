"""Fitting-only support and directed bounds for finite sensor repair.

Inside the fitting support, a proposed finite update has a small original-linear
budget. An observation outside that support can return anywhere in the support,
while movement farther away remains budgeted. This is a conditional safeguard;
an unmodified, unseen observation can itself fall outside the fitting support.
"""

import numpy as np
import torch

from .normalization import transform_spec


def fit_recovery_envelope(fit_frame, scaler):
    """Estimate per-regime encoded min/max from exactly the scaler's fit units.

    No target, validation frame, calibration frame, or official test input enters
    this operation. The caller supplies complete fitting-engine observations.
    """
    state = getattr(scaler, "state", None)
    if not isinstance(state, dict) or "fit_units" not in state:
        raise ValueError("Recovery support requires a fitted scaler with fitting-engine provenance.")
    if len(fit_frame) == 0 or "unit" not in fit_frame:
        raise ValueError("Recovery support requires nonempty fitting-engine data.")
    try:
        raw_units = np.asarray(fit_frame.unit.unique(), dtype=float)
        if not np.isfinite(raw_units).all() or np.any(raw_units != np.floor(raw_units)):
            raise ValueError("Non-integer engine identifier.")
        units = sorted(int(unit) for unit in raw_units)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Recovery support requires finite integer engine identifiers.") from error
    if units != sorted(state["fit_units"]):
        raise ValueError("Recovery support must use exactly the scaler's fitting engines.")
    method, scale = transform_spec(state)
    centers = np.asarray(state.get("centers"), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or not 1 <= len(centers) <= 6 or not np.isfinite(centers).all():
        raise ValueError("Recovery support requires one to six finite three-setting regime centers.")
    values, regimes = scaler.transform(fit_frame)
    values, regimes = np.asarray(values, dtype=np.float32), np.asarray(regimes)
    if (values.shape != (len(fit_frame), 24) or not np.isfinite(values).all() or
            regimes.shape != (len(fit_frame),) or regimes.dtype.kind not in "iu" or
            np.any(regimes < 0) or np.any(regimes >= len(centers))):
        raise ValueError("Recovery support requires finite encoded measurements and valid fitting regimes.")
    lower, upper, counts = [], [], []
    for regime in range(len(centers)):
        sensors = values[regimes == regime, 3:]
        if not len(sensors):
            raise ValueError(f"Recovery regime {regime} has no fitting observations.")
        lower.append(sensors.min(axis=0).tolist())
        upper.append(sensors.max(axis=0).tolist())
        counts.append(len(sensors))
    return {"method": "fitting_engine_regime_encoded_minmax", "fit_units": units,
            "lower": lower, "upper": upper, "regime_counts": counts,
            "centers": centers.tolist(), "sensor_transform": method,
            "sensor_transform_scale": scale,
            "scope": "provided fitting-engine measurements only; no targets or held-out engines"}


def _binary_mask(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.device != device:
        raise ValueError(f"{name} must match its observation shape and device.")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must contain only zero/one mask values.")
    return value.bool()


def bounded_finite_repair(observed, proposal, finite_mask, valid, encoded_lower, encoded_upper,
                          config, *, return_diagnostics=False):
    """Bound an already-gated repair proposal; leave imputation/padding unchanged.

    ``encoded_lower`` and ``encoded_upper`` broadcast to ``observed`` and must
    already select the appropriate fitting regime for each time step. For an
    inlier ``x``, the allowed linear interval is ``[x-b, x+b]``. A high outlier
    may use ``[L-b, x+b]``; a low outlier may use ``[x-b, U+b]``. Thus arbitrarily
    large outliers can return to the full fitting support, rather than taking a
    globally capped step. ``b`` is ``max_intact_repair_rmse`` in the original,
    uncompressed regime-standardized representation, not raw physical units.

    Bounds are evaluated in float64 and rounded inward to representable output
    values. The actual proposal is clamped: gradients are zero when saturated
    and unchanged in the interior. Bounds describe fixed observed inputs and
    fitting statistics; they are not learned, and no straight-through estimator
    is used. Identical proposals preserve the observed bit pattern exactly.
    """
    if (not isinstance(observed, torch.Tensor) or not observed.is_floating_point() or
            observed.ndim < 2 or observed.shape[-1] != 21 or
            not isinstance(proposal, torch.Tensor) or proposal.shape != observed.shape or
            proposal.dtype != observed.dtype or proposal.device != observed.device):
        raise ValueError("Observed and proposed repair must be floating tensors with identical shapes ending in 21 sensors.")
    finite = _binary_mask(finite_mask, observed.shape, "finite_mask", observed.device)
    validity = _binary_mask(valid, observed.shape[:-1], "valid", observed.device)
    active = finite & validity.unsqueeze(-1)
    if not bool(torch.isfinite(observed[active]).all()):
        raise ValueError("Finite valid observations must not contain NaN or infinity.")
    if not bool(torch.isfinite(proposal[validity]).all()):
        raise ValueError("Valid repair proposals must not contain NaN or infinity.")
    try:
        lower = torch.broadcast_to(torch.as_tensor(encoded_lower, device=observed.device, dtype=torch.float64), observed.shape)
        upper = torch.broadcast_to(torch.as_tensor(encoded_upper, device=observed.device, dtype=torch.float64), observed.shape)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Encoded recovery bounds must broadcast to the sensor observations.") from error
    if not bool((torch.isfinite(lower) & torch.isfinite(upper) & (lower <= upper)).all()):
        raise ValueError("Recovery bounds must be finite, ordered encoded intervals.")
    method, scale = transform_spec(config)
    try:
        budget = float(config["max_intact_repair_rmse"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("A finite nonnegative max_intact_repair_rmse budget is required.") from error
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("max_intact_repair_rmse must be finite and nonnegative.")

    # Observations and fitted support are fixed data for generator optimization.
    # Keeping bound construction outside autograd also avoids differentiating
    # nextafter, which only corrects floating-point representation of limits.
    with torch.no_grad():
        safe_observed = torch.where(active, observed, torch.zeros_like(observed))
        x = safe_observed.double()
        high, low = x > upper, x < lower
        lower_base = torch.where(high, lower, x)
        upper_base = torch.where(low, upper, x)
        if method == "asinh":
            low_linear = scale * torch.sinh(lower_base / scale) - budget
            high_linear = scale * torch.sinh(upper_base / scale) + budget
            bound_lower = scale * torch.asinh(low_linear / scale)
            bound_upper = scale * torch.asinh(high_linear / scale)
        else:
            bound_lower, bound_upper = lower_base - budget, upper_base + budget
        if not bool((torch.isfinite(bound_lower[active]) & torch.isfinite(bound_upper[active])).all()):
            raise ValueError("Recovery support exceeds the finite inverse-transform range.")
        cast_lower, cast_upper = bound_lower.to(observed.dtype), bound_upper.to(observed.dtype)
        cast_lower = torch.where(cast_lower.double() < bound_lower,
                                 torch.nextafter(cast_lower, torch.full_like(cast_lower, float("inf"))), cast_lower)
        cast_upper = torch.where(cast_upper.double() > bound_upper,
                                 torch.nextafter(cast_upper, torch.full_like(cast_upper, -float("inf"))), cast_upper)
        # The representable observed point belongs to every directed interval.
        # Include it explicitly to neutralize inverse-transform roundoff at b=0.
        cast_lower = torch.minimum(cast_lower, safe_observed)
        cast_upper = torch.maximum(cast_upper, safe_observed)
    clipped = torch.clamp(proposal, min=cast_lower, max=cast_upper)
    repaired = torch.where(active, clipped, proposal)
    if not return_diagnostics:
        return repaired
    diagnostics = {"finite_valid_mask": active, "inlier_mask": active & ~high & ~low,
                   "outlier_mask": active & (high | low),
                   "clipped_mask": active & (repaired != proposal)}
    return repaired, diagnostics
