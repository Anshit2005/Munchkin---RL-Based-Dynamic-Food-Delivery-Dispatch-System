"""Uniform experience replay storing next-state action masks (needed for masked targets)."""
from __future__ import annotations

import numpy as np

from environment.constants import N_ACTIONS, OBS_DIM


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.size = 0
        self.ptr = 0

    def add(self, obs, act, rew, next_obs, done, next_mask):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i], self.next_mask[i] = next_obs, float(done), next_mask
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx],
                self.done[idx], self.next_mask[idx])

    def __len__(self):
        return self.size
