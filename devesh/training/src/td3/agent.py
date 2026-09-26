"""
TD3 Agent: Twin Delayed Deep Deterministic Policy Gradient.

Offline formulation: replay buffer is pre-populated from C-MAPSS training
sequences before gradient updates begin. No online environment interaction.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient agent.

    State  : [h(64), y_base/rul_scale(1), t/t_max(1)]  → dim 66
    Action : residual correction ∈ [-max_action, +max_action] → dim 1
    """

    def __init__(
        self,
        state_dim: int = 66,
        action_dim: int = 1,
        max_action: float = 6.0,
        device: Optional[torch.device] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
    ):
        self.device = device or torch.device("cpu")
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self._total_it = 0

        # Actor and target actor
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Twin critics and target critics
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(
        self,
        state: np.ndarray,
        noise: float = 0.0,
    ) -> np.ndarray:
        """
        Select action(s) for a state or batch of states.

        Parameters
        ----------
        state : [state_dim] or [B, state_dim]
        noise : exploration noise std (0 = deterministic)

        Returns
        -------
        action : [1] or [B, 1]
        """
        if state.ndim == 1:
            state = state[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False

        state_t = torch.FloatTensor(state).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()
        self.actor.train()

        if noise > 0.0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)

        return action[0] if squeeze else action

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> Dict:
        self._total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.FloatTensor(action).to(self.device)
        reward_t = torch.FloatTensor(reward).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        done_t = torch.FloatTensor(done).to(self.device)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action_t) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state_t) + noise).clamp(
                -self.max_action, self.max_action
            )
            # Twin critic target Q
            q1_t, q2_t = self.critic_target(next_state_t, next_action)
            target_q = reward_t + (1.0 - done_t) * self.discount * torch.min(q1_t, q2_t)

        # Critic update
        q1, q2 = self.critic(state_t, action_t)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = 0.0
        if self._total_it % self.policy_freq == 0:
            # Delayed actor update
            actor_loss = -self.critic.Q1(state_t, self.actor(state_t)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = float(actor_loss.item())

            # Soft target updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_val,
            "q1_mean": float(q1.mean().item()),
        }

    def save(self, actor_path: str, critic_path: str) -> None:
        os.makedirs(os.path.dirname(actor_path), exist_ok=True)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path: str, critic_path: str) -> None:
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
