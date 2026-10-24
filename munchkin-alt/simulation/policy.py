"""Policy protocol shared by heuristics and RL agents, plus two sanity-check policies."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from environment.constants import WAIT_ACTION


class Policy(Protocol):
    name: str

    def reset(self, env_config) -> None: ...

    def act(self, obs: np.ndarray, info: dict, view) -> int: ...


class RandomValidPolicy:
    """Uniformly random over valid actions (sanity floor)."""
    name = "random_valid"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env_config=None):
        pass

    def act(self, obs, info, view) -> int:
        valid = np.flatnonzero(info["action_mask"])
        return int(self.rng.choice(valid))


class NeverAssignPolicy:
    """Always waits (sanity floor)."""
    name = "never_assign"

    def reset(self, env_config=None):
        pass

    def act(self, obs, info, view) -> int:
        return WAIT_ACTION
