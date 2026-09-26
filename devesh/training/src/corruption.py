"""
Sensor-corruption augmentation for robustness training and evaluation.

All modes preserve input shape [B, W, F].
"""

import torch
import numpy as np
from typing import Optional, Tuple


def apply_corruption(
    x: torch.Tensor,
    mode: str = "gaussian_noise",
    seed: Optional[int] = None,
    p_mask: float = 0.20,
    noise_std: float = 0.15,
    spike_prob: float = 0.05,
    drift_scale: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a corruption to input tensor x and return (corrupted_x, mask).

    Parameters
    ----------
    x        : [B, W, F]  normalised sensor windows
    mode     : one of 'gaussian_noise', 'sensor_masking', 'random_dropout',
                       'spike_noise', 'drift_corruption', 'dynamic_mixed', 'clean'
    seed     : optional random seed (for reproducible testing)
    p_mask   : fraction of sensors/time-steps to mask
    noise_std: standard deviation of Gaussian noise
    spike_prob: per-element probability of spike noise
    drift_scale: maximum drift amplitude

    Returns
    -------
    corrupted_x : [B, W, F]  corrupted tensor (same shape as x)
    mask        : [B, W, F]  binary mask — 1 = corrupted element, 0 = clean
    """
    assert x.ndim == 3, f"Expected [B, W, F] tensor, got shape {x.shape}"
    device = x.device
    B, W, F = x.shape

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    mask = torch.zeros_like(x)
    corrupted = x.clone()

    if mode == "clean":
        return corrupted, mask

    elif mode == "gaussian_noise":
        noise = torch.randn_like(x) * noise_std
        corrupted = x + noise
        mask = torch.ones_like(x)  # all elements "corrupted" but softly

    elif mode == "sensor_masking":
        # Mask entire sensor channels (columns) — realistic total sensor failure
        n_mask = max(1, int(p_mask * F))
        for b in range(B):
            cols = torch.randperm(F)[:n_mask]
            corrupted[b, :, cols] = 0.0
            mask[b, :, cols] = 1.0

    elif mode == "random_dropout":
        # Randomly zero individual elements
        drop_mask = (torch.rand(B, W, F, device=device) < p_mask)
        corrupted = x.clone()
        corrupted[drop_mask] = 0.0
        mask[drop_mask] = 1.0

    elif mode == "spike_noise":
        # Sparse large spikes simulating transient faults
        spike_mask = (torch.rand(B, W, F, device=device) < spike_prob)
        spikes = (torch.randn(B, W, F, device=device) * 5.0) * spike_mask.float()
        corrupted = x + spikes
        mask[spike_mask] = 1.0

    elif mode == "drift_corruption":
        # Time-varying linear drift applied to random sensors
        n_drift = max(1, int(p_mask * F))
        for b in range(B):
            cols = torch.randperm(F)[:n_drift]
            drift = torch.linspace(0, drift_scale, W, device=device)
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            corrupted[b, :, cols] += sign * drift.unsqueeze(-1).expand(W, len(cols))
            mask[b, :, cols] = 1.0

    elif mode == "dynamic_mixed":
        # Randomly selects one of the five failure modes per batch item
        modes = ["gaussian_noise", "sensor_masking", "random_dropout", "spike_noise", "drift_corruption"]
        for b in range(B):
            chosen = modes[torch.randint(len(modes), (1,)).item()]
            c_b, m_b = apply_corruption(
                x[b : b + 1],
                mode=chosen,
                p_mask=p_mask,
                noise_std=noise_std,
                spike_prob=spike_prob,
                drift_scale=drift_scale,
            )
            corrupted[b] = c_b[0]
            mask[b] = m_b[0]

    else:
        raise ValueError(f"Unknown corruption mode: '{mode}'")

    assert corrupted.shape == x.shape, "Corruption changed tensor shape!"
    assert mask.shape == x.shape, "Mask shape mismatch!"
    return corrupted, mask
