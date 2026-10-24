"""Double DQN agent with action masking, plus the Policy wrapper used for evaluation/UI."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from environment.constants import WAIT_ACTION
from rl.common.checkpoint import load_checkpoint
from rl.common.networks import masked_fill
from rl.dqn.model import QNetwork


class DQNAgent:
    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.gamma = cfg.get("gamma", 0.99)
        self.q = QNetwork(cfg.get("arch", "pairwise"), cfg.get("hidden", 128))
        self.q_target = copy.deepcopy(self.q)
        self.q_target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.get("lr", 1e-4))
        self.rng = np.random.default_rng(seed)
        self.tau = cfg.get("tau", 0.0)

    # ------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, eps: float) -> int:
        valid = np.flatnonzero(mask)
        if self.rng.random() < eps:
            return int(self.rng.choice(valid))
        q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    # ------------------------------------------------------------ learning
    def learn(self, batch) -> float:
        obs, act, rew, next_obs, done, next_mask = [torch.as_tensor(x) for x in batch]
        q_sa = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online net selects among VALID next actions, target net evaluates.
            next_online = masked_fill(self.q(next_obs), next_mask)
            next_a = next_online.argmax(1, keepdim=True)
            next_q = self.q_target(next_obs).gather(1, next_a).squeeze(1)
            target = rew + self.gamma * (1.0 - done) * next_q
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.get("grad_clip", 10.0))
        self.opt.step()
        if self.tau > 0:
            self.soft_update()
        return float(loss.item())

    def hard_update(self):
        self.q_target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def soft_update(self):
        for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
            tp.mul_(1 - self.tau).add_(self.tau * p)

    def state(self) -> dict:
        return {"q": self.q.state_dict()}


class DQNPolicy:
    """Greedy (epsilon = 0) masked policy. Loads a checkpoint or wraps an agent."""
    name = "dqn"
    label = "DQN"

    def __init__(self, agent: DQNAgent | None = None, checkpoint: str | None = None):
        if agent is None:
            ck = load_checkpoint(checkpoint)
            agent = DQNAgent(ck["meta"]["agent_cfg"])
            agent.q.load_state_dict(ck["state"]["q"])
        self.agent = agent
        self.agent.q.eval()

    def reset(self, env_config=None):
        pass

    def act(self, obs, info, view) -> int:
        a = self.agent.act(obs, info["action_mask"], 0.0)
        return a if info["action_mask"][a] else WAIT_ACTION
