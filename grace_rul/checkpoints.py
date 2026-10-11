"""Atomic epoch checkpoints, including random state for interrupted training."""

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ARCHITECTURE


def pipeline_identity(train_frame, validation_frame, subset, config, seed):
    def fingerprint(frame):
        values = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
        return hashlib.sha256(values + repr(list(frame.columns)).encode()).hexdigest()

    return {"version": 1, "architecture": ARCHITECTURE, "subset": subset, "seed": int(seed),
            "config": {k: v for k, v in config.items() if k != "progress_interval_seconds"},
            "fit": fingerprint(train_frame), "validation": fingerprint(validation_frame)}


def random_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state):
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_cpu(state), temporary)
    temporary.replace(path)


def load_checkpoint(path, identity):
    if path is None or not Path(path).exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("identity") != identity:
        raise ValueError(f"Checkpoint does not match this configuration or engine data: {path}. Start a new run directory.")
    return state
