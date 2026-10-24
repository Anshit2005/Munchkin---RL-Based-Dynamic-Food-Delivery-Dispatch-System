"""Test helpers: hand-built scenarios with exact, known dynamics."""
from __future__ import annotations

import numpy as np

from environment.config import EnvConfig, load_env_defaults
from environment.constants import N_TICKS
from environment.scenario import OrderSpec, Scenario, _fingerprint


def make_scenario(restaurants, drivers, orders, cfg: EnvConfig | None = None, traffic_mult: float = 1.0,
                  seed: int = 0) -> Scenario:
    """orders: list of dicts with created_at, restaurant_id, cx, cy, prep, [deadline]."""
    cfg = (cfg or load_env_defaults()).replace(n_restaurants=len(restaurants), n_drivers=len(drivers),
                                               traffic_noise=0.0)
    specs = []
    for o in orders:
        specs.append(OrderSpec(o["created_at"], o["restaurant_id"], o["cx"], o["cy"], o["prep"],
                               o.get("prep_est", o["prep"]), o.get("deadline", 1e9)))
    traffic = np.full(N_TICKS, traffic_mult * 1.0)
    fp = _fingerprint(restaurants, drivers, specs, traffic)
    return Scenario(cfg, seed, tuple(restaurants), tuple(drivers), tuple(specs), traffic, fp)
