"""Masked Double DQN agent used for Munchkin dispatch."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .networks import make_network, masked_values
from .state import N_ACTIONS, OBS_SIZE


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.size = self.position = 0

    def add(self, obs, action, reward, next_obs, done, next_mask):
        i = self.position
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.done[i], self.next_mask[i] = next_obs, float(done), next_mask
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, self.size, size=batch_size)
        return (self.obs[indices], self.action[indices], self.reward[indices],
                self.next_obs[indices], self.done[indices], self.next_mask[indices])

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, hidden: int = 128, arch: str = "pairwise", learning_rate: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.0, seed: int = 0):
        torch.manual_seed(seed)
        self.gamma, self.tau = gamma, tau
        self.q = make_network(arch, hidden)
        self.target = copy.deepcopy(self.q).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs, mask, epsilon: float = 0.0) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("DQN received an empty legal-action mask")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        values = self.q(tensor)[0].cpu().numpy()
        return int(np.argmax(np.where(mask, values, -np.inf)))

    def learn(self, batch, grad_clip: float = 10.0) -> float:
        obs, actions, rewards, next_obs, dones, next_masks = batch
        obs = torch.as_tensor(obs, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)
        next_masks = torch.as_tensor(next_masks, dtype=torch.bool)

        chosen = self.q(obs).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            best_next = masked_values(self.q(next_obs), next_masks).argmax(1, keepdim=True)
            next_values = self.target(next_obs).gather(1, best_next).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * next_values
        loss = F.smooth_l1_loss(chosen, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), grad_clip)
        self.optimizer.step()
        self.update_target()
        return float(loss.item())

    @torch.no_grad()
    def update_target(self):
        if self.tau:
            for source, target in zip(self.q.parameters(), self.target.parameters()):
                target.mul_(1.0 - self.tau).add_(self.tau * source)

    def state_dict(self):
        return self.q.state_dict()

