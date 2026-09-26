"""
Utility functions: seed, device, parameter count, memory measurement.
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    """Return CUDA device if available, else CPU."""
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(module: nn.Module) -> int:
    """Count total trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_peak_memory_mb() -> float:
    """Return peak CUDA memory in MB (0 if no CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def measure_latency(
    model: nn.Module,
    input_shape: tuple = (1, 30, 14),
    device: Optional[torch.device] = None,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> float:
    """Measure average per-sample inference latency in milliseconds."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return float(np.mean(times))
