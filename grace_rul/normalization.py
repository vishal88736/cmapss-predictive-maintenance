"""Invertible model-space compression; physical sensor faults stay in raw units."""

import numpy as np


def transform_spec(state):
    # Missing metadata identifies the original format-5 linear representation.
    state = state or {}
    method = state.get("sensor_transform", "linear")
    scale = float(state.get("sensor_transform_scale", 1.))
    if method not in {"linear", "asinh"} or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sensor transform must be linear/asinh with a finite positive scale.")
    return method, scale


def encode_sensors(linear, state):
    method, scale = transform_spec(state)
    return scale * np.arcsinh(np.asarray(linear) / scale) if method == "asinh" else np.asarray(linear)


def decode_sensors(encoded, state):
    method, scale = transform_spec(state)
    # Float64 keeps inverse diagnostics usable far beyond float32's sinh range.
    values = np.asarray(encoded, dtype=np.float64)
    return scale * np.sinh(values / scale) if method == "asinh" else values
