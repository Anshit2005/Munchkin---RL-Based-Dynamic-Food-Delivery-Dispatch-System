"""Clipped PPO agent with GAE and legal-action masking."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .networks import make_network, masked_values
from .state import N_ACTIONS, OBS_SIZE


class PPOAgent:
    def __init__(self, hidden: int = 128, value_hidden: int = 256, arch: str = "pairwise",
                 learning_rate: float = 3e-4, seed: int = 0):
        torch.manual_seed(seed)
        self.policy = make_network(arch, hidden, output_gain=0.01)
        self.value = nn.Sequential(nn.Linear(OBS_SIZE, value_hidden), nn.Tanh(),
                                   nn.Linear(value_hidden, value_hidden), nn.Tanh(),
                                   nn.Linear(value_hidden, 1))
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()),
                                          lr=learning_rate, eps=1e-5)
        self.generator = torch.Generator().manual_seed(seed)

    def distribution(self, obs, mask):
        logits = masked_values(self.policy(obs), mask)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs, mask, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        dist = self.distribution(obs_t, mask_t)
        if deterministic:
            action = dist.probs.argmax(-1)
        else:
            action = torch.multinomial(dist.probs, 1, generator=self.generator).squeeze(-1)
        return int(action.item()), float(dist.log_prob(action).item()), float(self.value(obs_t).item())

    def update(self, rollout: list[dict], gamma: float = 0.99, gae_lambda: float = 0.95,
               clip: float = 0.2, epochs: int = 8, minibatch_size: int = 128,
               value_coef: float = 0.5, entropy_coef: float = 0.01, grad_clip: float = 0.5):
        if not rollout:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        advantages = np.zeros(len(rollout), dtype=np.float32)
        last_advantage = 0.0
        next_value = 0.0
        for i in range(len(rollout) - 1, -1, -1):
            transition = rollout[i]
            continuation = 0.0 if transition["done"] else 1.0
            delta = transition["reward"] + gamma * next_value * continuation - transition["value"]
            last_advantage = delta + gamma * gae_lambda * continuation * last_advantage
            advantages[i] = last_advantage
            next_value = transition["value"]
        returns = advantages + np.asarray([x["value"] for x in rollout], dtype=np.float32)

        obs = torch.as_tensor(np.stack([x["obs"] for x in rollout]), dtype=torch.float32)
        masks = torch.as_tensor(np.stack([x["mask"] for x in rollout]), dtype=torch.bool)
        actions = torch.as_tensor([x["action"] for x in rollout], dtype=torch.int64)
        old_logp = torch.as_tensor([x["logp"] for x in rollout], dtype=torch.float32)
        adv = torch.as_tensor(advantages, dtype=torch.float32)
        ret = torch.as_tensor(returns, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        stats = {"policy_loss": [], "value_loss": [], "entropy": []}
        for _ in range(epochs):
            indices = np.random.permutation(len(rollout))
            for start in range(0, len(rollout), minibatch_size):
                batch = torch.as_tensor(indices[start:start + minibatch_size], dtype=torch.int64)
                dist = self.distribution(obs[batch], masks[batch])
                logp = dist.log_prob(actions[batch])
                ratio = (logp - old_logp[batch]).exp()
                batch_adv = adv[batch]
                policy_loss = -torch.minimum(
                    ratio * batch_adv,
                    ratio.clamp(1.0 - clip, 1.0 + clip) * batch_adv,
                ).mean()
                value_loss = 0.5 * (self.value(obs[batch]).squeeze(-1) - ret[batch]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), grad_clip)
                self.optimizer.step()
                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy.item()))
        return {key: float(np.mean(values)) for key, values in stats.items()}

    def state_dict(self):
        return {"policy": self.policy.state_dict(), "value": self.value.state_dict()}

