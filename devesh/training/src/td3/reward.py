"""
TD3 reward function for offline single-step residual correction.

This is NOT a physical engine controller. The "environment" is the
C-MAPSS data; each step corrects a single RUL prediction.

Reward:
  R = -|y_base + a - y_true|          (accuracy)
    - lambda_mag  * |a|               (correction magnitude penalty)
    - lambda_rate * |a - a_prev|      (correction rate penalty)
    - bounds_penalty                  (hard penalty if corrected > max_rul)
"""

import torch
from typing import Optional


def compute_residual_reward(
    y_base: torch.Tensor,       # [B, 1]  baseline prediction
    action: torch.Tensor,       # [B, 1]  TD3 correction
    y_true: torch.Tensor,       # [B, 1]  ground-truth RUL
    prev_action: Optional[torch.Tensor] = None,
    max_rul: float = 125.0,
    lambda_mag: float = 0.05,
    lambda_rate: float = 0.02,
    bounds_penalty: float = 5.0,
) -> torch.Tensor:
    """Returns reward tensor [B, 1]."""
    if y_base.ndim == 1:
        y_base = y_base.unsqueeze(-1)
    if action.ndim == 1:
        action = action.unsqueeze(-1)
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)

    y_corrected = y_base + action
    accuracy = -(y_corrected - y_true).abs()
    mag_pen = lambda_mag * action.abs()

    if prev_action is not None:
        if isinstance(prev_action, (int, float)):
            prev_action = torch.full_like(action, float(prev_action))
        elif prev_action.ndim == 0:
            prev_action = prev_action.unsqueeze(0).expand_as(action)
        rate_pen = lambda_rate * (action - prev_action).abs()
    else:
        rate_pen = torch.zeros_like(action)

    out_of_bounds = ((y_corrected < 0) | (y_corrected > max_rul)).float()
    bound_pen = bounds_penalty * out_of_bounds

    reward = accuracy - mag_pen - rate_pen - bound_pen
    return reward.float()
