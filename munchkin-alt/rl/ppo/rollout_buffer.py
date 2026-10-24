"""On-policy rollout storage with GAE advantage estimation (action masks are stored so that
log-probabilities are recomputed under the same mask during the update)."""
from __future__ import annotations

import numpy as np

from environment.constants import N_ACTIONS, OBS_DIM


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int):
        self.T, self.N = n_steps, n_envs
        self.obs = np.zeros((n_steps, n_envs, OBS_DIM), dtype=np.float32)
        self.masks = np.zeros((n_steps, n_envs, N_ACTIONS), dtype=bool)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.logp = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.adv = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.ret = np.zeros((n_steps, n_envs), dtype=np.float32)

    def add(self, t, obs, masks, actions, logp, values, rewards, dones):
        self.obs[t], self.masks[t], self.actions[t] = obs, masks, actions
        self.logp[t], self.values[t], self.rewards[t], self.dones[t] = logp, values, rewards, dones

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float):
        """dones[t] = 1 means the episode terminated after step t (true terminal at t=180, so no
        bootstrapping across it)."""
        last = np.zeros(self.N, dtype=np.float32)
        for t in reversed(range(self.T)):
            nonterm = 1.0 - self.dones[t]
            next_v = last_values if t == self.T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_v * nonterm - self.values[t]
            last = delta + gamma * lam * nonterm * last
            self.adv[t] = last
        self.ret = self.adv + self.values

    def flat(self):
        f = lambda x: x.reshape(self.T * self.N, *x.shape[2:])
        return (f(self.obs), f(self.masks), f(self.actions), f(self.logp), f(self.adv), f(self.ret),
                f(self.values))
