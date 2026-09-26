"""
TD3 Actor and Critic network definitions.

State dim : 66  = [h(64), y_base/rul_scale(1), t/t_max(1)]
Action dim: 1   = residual correction ∈ [-6, +6]
"""

import torch
import torch.nn as nn
from typing import Tuple


class Actor(nn.Module):
    """
    TD3 Actor: state → action.
    Output is scaled by max_action via tanh.
    """

    def __init__(self, state_dim: int = 66, action_dim: int = 1, max_action: float = 6.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.max_action


class Critic(nn.Module):
    """
    Twin-critic network Q1, Q2.
    Takes (state, action) and returns two Q-value estimates.
    """

    def __init__(self, state_dim: int = 66, action_dim: int = 1):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
