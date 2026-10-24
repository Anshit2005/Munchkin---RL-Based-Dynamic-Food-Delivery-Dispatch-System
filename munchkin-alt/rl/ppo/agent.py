"""PPO agent (clipped objective, GAE, action masking) and the Policy wrapper."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.constants import WAIT_ACTION
from rl.common.checkpoint import load_checkpoint
from rl.common.networks import masked_fill
from rl.ppo.model import PolicyNetwork, ValueNetwork


class PPOAgent:
    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.policy = PolicyNetwork(cfg.get("arch", "pairwise"), cfg.get("hidden", 128))
        self.value = ValueNetwork(cfg.get("value_hidden", 256))
        self.opt = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()),
                                    lr=cfg.get("lr", 3e-4), eps=1e-5)
        self.gen = torch.Generator().manual_seed(seed)

    def _dist(self, obs, mask):
        return Categorical(logits=masked_fill(self.policy(obs), mask))

    @torch.no_grad()
    def act(self, obs: np.ndarray, masks: np.ndarray, deterministic: bool = False):
        o = torch.as_tensor(obs, dtype=torch.float32)
        m = torch.as_tensor(masks, dtype=torch.bool)
        dist = self._dist(o, m)
        a = dist.probs.argmax(-1) if deterministic else torch.multinomial(dist.probs, 1, generator=self.gen).squeeze(-1)
        return a.numpy(), dist.log_prob(a).numpy(), self.value(o).numpy()

    @torch.no_grad()
    def values(self, obs: np.ndarray) -> np.ndarray:
        return self.value(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    def update(self, buf, ent_coef: float, rng: np.random.Generator) -> dict:
        c = self.cfg
        obs, masks, actions, old_logp, adv, ret, _ = [torch.as_tensor(x) for x in buf.flat()]
        n = obs.shape[0]
        stats = {"pi_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": []}
        stop = False
        for _ in range(c["epochs"]):
            perm = rng.permutation(n)
            for s in range(0, n, c["minibatch"]):
                idx = torch.as_tensor(perm[s:s + c["minibatch"]])
                dist = self._dist(obs[idx], masks[idx])
                logp = dist.log_prob(actions[idx])
                ratio = (logp - old_logp[idx]).exp()
                a = adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                v_loss = 0.5 * (self.value(obs[idx]) - ret[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + c["vf_coef"] * v_loss - ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()),
                                         c["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (ratio.log())).mean().item()
                    stats["pi_loss"].append(pi_loss.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["entropy"].append(ent.item())
                    stats["kl"].append(kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > c["clip"]).float().mean().item())
                if c.get("target_kl") and kl > c["target_kl"]:
                    stop = True
                    break
            if stop:
                break
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def state(self) -> dict:
        return {"policy": self.policy.state_dict(), "value": self.value.state_dict()}


class PPOPolicy:
    """Deterministic (argmax of masked logits) policy for evaluation / UI."""
    name = "ppo"
    label = "PPO"

    def __init__(self, agent: PPOAgent | None = None, checkpoint: str | None = None,
                 deterministic: bool = True):
        if agent is None:
            ck = load_checkpoint(checkpoint)
            agent = PPOAgent(ck["meta"]["agent_cfg"])
            agent.policy.load_state_dict(ck["state"]["policy"])
            agent.value.load_state_dict(ck["state"]["value"])
        self.agent = agent
        self.deterministic = deterministic
        self.agent.policy.eval()

    def reset(self, env_config=None):
        pass

    def act(self, obs, info, view) -> int:
        a, _, _ = self.agent.act(obs[None], info["action_mask"][None], self.deterministic)
        a = int(a[0])
        return a if info["action_mask"][a] else WAIT_ACTION
