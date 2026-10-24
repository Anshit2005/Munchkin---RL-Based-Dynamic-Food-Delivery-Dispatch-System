"""Training/validation environment factories and a small vectorised env (with action masks)."""
from __future__ import annotations

import numpy as np

from environment.config import load_reward_config, load_scenarios
from environment.delivery_env import DeliveryEnv
from environment.sampler import ScenarioSampler
from simulation.runner import run_episode
from utils.seeding import VAL_SEEDS


def make_train_env(batching: bool = False, reward_config=None, p_named: float = 0.6) -> DeliveryEnv:
    return DeliveryEnv(sampler=ScenarioSampler(p_named=p_named, batching=batching),
                       reward_config=reward_config or load_reward_config())


class VecEnv:
    """Synchronous vector env with auto-reset. Keeps the action mask per env."""

    def __init__(self, n_envs: int, base_seed: int = 0, batching: bool = False, reward_config=None):
        self.n = n_envs
        self.envs = [make_train_env(batching, reward_config) for _ in range(n_envs)]
        self.obs = np.zeros((n_envs,) + self.envs[0].observation_space.shape, dtype=np.float32)
        self.masks = np.zeros((n_envs, self.envs[0].action_space.n), dtype=bool)
        for i, e in enumerate(self.envs):
            o, info = e.reset(seed=(base_seed * 1009 + i * 7919) % 1_000_000)
            self.obs[i], self.masks[i] = o, info["action_mask"]

    def step(self, actions):
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)
        finished = []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, term, _, info = e.step(int(a))
            rewards[i] = r
            dones[i] = term
            if term:
                finished.append({"return": info["episode_return"], "metrics": info["episode_metrics"],
                                 "scenario": info["scenario_name"], "breakdown": info["episode_breakdown"]})
                o, info = e.reset()
            self.obs[i], self.masks[i] = o, info["action_mask"]
        return self.obs.copy(), self.masks.copy(), rewards, dones, finished


def validation_score(policy, n_seeds_per_scenario: int = 4, scenarios=None, reward_config=None) -> dict:
    """Mean episode reward of `policy` on fixed validation seeds across the named scenarios."""
    scenarios = scenarios or load_scenarios()
    rewards, per = [], {}
    for name, cfg in scenarios.items():
        rs = [run_episode(policy, cfg, s, reward_config).reward_total for s in VAL_SEEDS[:n_seeds_per_scenario]]
        per[name] = float(np.mean(rs))
        rewards += rs
    return {"mean_reward": float(np.mean(rewards)), "per_scenario": per}
