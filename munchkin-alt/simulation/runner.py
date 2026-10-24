"""Episode runner shared by evaluation, training-time validation and the UI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from environment.config import EnvConfig, RewardConfig
from environment.constants import N_TICKS
from environment.delivery_env import DeliveryEnv


@dataclass
class EpisodeResult:
    policy: str
    scenario: str
    seed: int
    fingerprint: str
    metrics: dict
    reward_total: float
    reward_discounted: float
    reward_breakdown: dict
    n_decisions: int
    n_invalid: int
    wall_time: float
    final_tick: int
    frames: Optional[list] = field(default=None, repr=False)

    def row(self) -> dict:
        r = {"policy": self.policy, "scenario": self.scenario, "seed": self.seed,
             "fingerprint": self.fingerprint, "reward": self.reward_total,
             "reward_discounted": self.reward_discounted, "n_decisions": self.n_decisions,
             "n_invalid": self.n_invalid, "final_tick": self.final_tick}
        r.update(self.metrics)
        r.update({"rw_" + k: v for k, v in self.reward_breakdown.items()})
        return r


def run_episode(policy, env_config: EnvConfig, seed: int, reward_config: Optional[RewardConfig] = None,
                record: bool = False) -> EpisodeResult:
    env = DeliveryEnv(env_config, reward_config, record=record)
    policy.reset(env_config)
    obs, info = env.reset(seed=seed)
    t0 = time.perf_counter()
    terminated = False
    while not terminated:
        action = policy.act(obs, info, env.get_view())
        obs, _, terminated, _, info = env.step(action)
    assert env.engine.t == N_TICKS, "episode must end at exactly 180 ticks"
    return EpisodeResult(
        policy=policy.name, scenario=env_config.name, seed=seed, fingerprint=env.scenario.fingerprint,
        metrics=info["episode_metrics"], reward_total=info["episode_return"],
        reward_discounted=info["episode_return_discounted"], reward_breakdown=info["episode_breakdown"],
        n_decisions=info["n_decisions"], n_invalid=info["n_invalid"],
        wall_time=time.perf_counter() - t0, final_tick=env.engine.t,
        frames=env.frames if record else None)


def run_all(policies: list, env_config: EnvConfig, seed: int, reward_config=None, record: bool = False):
    """Same scenario and seed for every policy ("Run All")."""
    return [run_episode(p, env_config, seed, reward_config, record) for p in policies]
