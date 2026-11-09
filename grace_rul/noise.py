"""Broadband measurement noise and channel dropouts on top of the fault protocol.

The structured fault protocol models *failures* confined to one sensor domain:
bursts, outages, blocks, bias, drift and stuck readings. Real acquisition data
also carries broadband noise on every channel at once, plus the occasional
missing cell. This module perturbs raw sensor readings before the frozen
fitting-only scaler, so a deployed bundle sees exactly the normalization it was
trained with.

Scope limits are deliberate:

* Operating settings and speed commands are untouched. They describe the
  condition the engine is run at, not a sensor reading, and the regime
  assignment depends on them.
* ``sigma_fraction`` is a multiple of the fitting-only within-regime sensor
  scale, so one level means the same physical perturbation on every channel.
  It is an empirical sensitivity scale, not a sensor specification.
* Dropouts are NaN, which the model's availability mask already handles.
"""

import numpy as np
import pandas as pd

from .config import SENSORS

NOISE_PROTOCOL = "additive-gaussian-dropout-v1"
DEFAULT_LEVELS = (0., .05, .1, .2)
DEFAULT_DROPOUTS = (0., .02, .05)


def validate_noise(sigma_fraction, dropout_rate):
    if isinstance(sigma_fraction, bool) or not np.isfinite(sigma_fraction) or not 0 <= sigma_fraction <= 1:
        raise ValueError("sigma_fraction must be finite and in [0, 1].")
    if isinstance(dropout_rate, bool) or not np.isfinite(dropout_rate) or not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be finite and in [0, 1).")


def apply_measurement_noise(frame, state, *, sigma_fraction=0., dropout_rate=0., seed=0):
    """Return a copy of ``frame`` with noisy sensor readings.

    The draw is keyed on ``(seed, unit)``, so an engine receives identical noise
    regardless of row order or of which other engines share the frame. Nested
    levels reuse the same innovations, which makes a severity sweep comparable
    across levels instead of resampling a fresh realization each time.
    """
    validate_noise(sigma_fraction, dropout_rate)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not {"unit", *SENSORS}.issubset(frame.columns):
        raise ValueError("Expected unit and sensor columns on the trajectory frame.")
    scales = np.asarray(state["sensor_fault_scale"], float)
    if scales.shape != (len(SENSORS),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("A fitting-only sensor fault scale is required.")
    out = frame.copy()
    values = out[SENSORS].to_numpy(float)
    for unit, positions in frame.groupby("unit", sort=True).indices.items():
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(unit)]))
        if sigma_fraction > 0:
            # One fixed draw per engine: a level is a scaled perturbation, not a
            # new random realization, so level sweeps stay paired.
            innovations = rng.normal(size=(len(positions), len(SENSORS)))
            values[positions] += sigma_fraction * scales * innovations
        if dropout_rate > 0:
            missing = rng.random(size=(len(positions), len(SENSORS))) < dropout_rate
            values[positions] = np.where(missing, np.nan, values[positions])
    out[SENSORS] = values
    return out
