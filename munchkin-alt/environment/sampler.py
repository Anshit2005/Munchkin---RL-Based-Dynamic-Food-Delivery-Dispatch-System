"""Scenario sampler used for training: the config is a deterministic function of the seed."""
from __future__ import annotations

import numpy as np

from environment.config import EnvConfig, load_env_defaults, load_scenarios
from environment.constants import MAX_DRIVERS, MAX_RESTAURANTS


class ScenarioSampler:
    """With probability p_named pick one of the named scenarios, otherwise randomise the config."""

    def __init__(self, p_named: float = 0.6, batching: bool = False, names=None):
        self.named = load_scenarios()
        if names:
            self.named = {k: v for k, v in self.named.items() if k in names}
        self.names = list(self.named)
        self.p_named = p_named
        self.batching = batching
        self.base = load_env_defaults()

    def __call__(self, seed: int) -> EnvConfig:
        rng = np.random.default_rng([int(seed), 777])
        if rng.random() < self.p_named:
            cfg = self.named[self.names[int(rng.integers(len(self.names)))]]
        else:
            cfg = self.base.replace(
                name="random",
                n_restaurants=int(rng.integers(3, MAX_RESTAURANTS + 1)),
                n_drivers=int(rng.integers(3, MAX_DRIVERS + 1)),
                orders_per_min=float(np.round(rng.uniform(1.0, 10.0), 1)),
                traffic=str(rng.choice(["light", "normal", "heavy"])),
            )
        return cfg.replace(batching=self.batching)
