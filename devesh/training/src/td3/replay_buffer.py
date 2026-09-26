"""
Replay Buffer for offline TD3 training.
"""

import numpy as np
from typing import Dict, Tuple


class ReplayBuffer:
    """
    Circular replay buffer for offline TD3.

    Stores: state, action, reward, next_state, done
    """

    def __init__(self, state_dim: int = 66, action_dim: int = 1, max_size: int = 100_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.state[self.ptr] = np.ravel(state)[: self.state.shape[1]]
        self.action[self.ptr] = np.ravel(action)[: self.action.shape[1]]
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = np.ravel(next_state)[: self.next_state.shape[1]]
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx],
        )

    def __len__(self) -> int:
        return self.size
