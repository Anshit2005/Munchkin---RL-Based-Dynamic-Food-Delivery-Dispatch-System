"""EnvConfig / RewardConfig dataclasses and YAML loading."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml

from environment.constants import MAX_DRIVERS, MAX_RESTAURANTS, TRAFFIC_LEVELS

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

_FORBIDDEN_KEYS = {"duration", "episode_duration", "horizon", "n_ticks", "episode_length"}


@dataclass(frozen=True)
class EnvConfig:
    """World configuration. Deliberately has NO duration field (fixed 3-minute horizon)."""
    name: str = "custom"
    n_restaurants: int = 5
    n_drivers: int = 8
    orders_per_min: float = 5.0
    traffic: str = "normal"
    batching: bool = False
    # physics / calibration
    base_speed: float = 0.2            # grid units per second at traffic multiplier 1.0
    customer_radius: float = 4.0
    prep_mean: float = 30.0
    prep_std: float = 10.0
    prep_min: float = 10.0
    prep_max: float = 60.0
    prep_est_noise: float = 0.2        # lognormal sigma on the *observable* prep estimate
    deadline_slack: float = 30.0
    backlog_frac: float = 0.5          # orders already placed at t=0 = round(frac * orders_per_min)
    traffic_noise: float = 0.1
    min_restaurant_spacing: float = 2.0

    def __post_init__(self):
        if not 1 <= self.n_restaurants <= MAX_RESTAURANTS:
            raise ValueError(f"n_restaurants must be in [1, {MAX_RESTAURANTS}]")
        if not 1 <= self.n_drivers <= MAX_DRIVERS:
            raise ValueError(f"n_drivers must be in [1, {MAX_DRIVERS}]")
        if self.orders_per_min < 0:
            raise ValueError("orders_per_min must be >= 0")
        if self.traffic not in TRAFFIC_LEVELS:
            raise ValueError(f"traffic must be one of {list(TRAFFIC_LEVELS)}")

    @property
    def capacity(self) -> int:
        return 2 if self.batching else 1

    @property
    def nominal_speed(self) -> float:
        return self.base_speed * TRAFFIC_LEVELS[self.traffic]

    def replace(self, **kw) -> "EnvConfig":
        return replace(self, **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        bad = _FORBIDDEN_KEYS & set(d)
        if bad:
            raise ValueError(f"Episode duration is fixed at 3 simulated minutes and not configurable: {sorted(bad)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RewardConfig:
    w_delivery: float = 10.0     # w1
    w_wait: float = 2.0          # w2
    w_late: float = 4.0          # w3
    w_travel: float = 0.1        # w4
    w_batch: float = 0.5         # w5
    w_idle: float = 1.0          # w6
    undelivered_penalty: float = 5.0
    invalid_penalty: float = 1.0
    gamma: float = 0.99

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown reward keys: {sorted(unknown)}")
        return cls(**d)


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_env_defaults() -> EnvConfig:
    return EnvConfig.from_dict(load_yaml("env_default.yaml"))


def load_scenarios() -> dict:
    """Named scenarios (spec Table 1) layered over env_default.yaml."""
    base = load_yaml("env_default.yaml")
    out = {}
    for name, over in load_yaml("scenarios.yaml").items():
        out[name] = EnvConfig.from_dict({**base, **over, "name": name})
    return out


def load_reward_config() -> RewardConfig:
    return RewardConfig.from_dict(load_yaml("reward.yaml"))
