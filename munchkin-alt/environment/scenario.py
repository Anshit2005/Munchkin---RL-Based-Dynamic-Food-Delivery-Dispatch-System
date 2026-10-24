"""Pre-sampled exogenous scenario trace.

All randomness (restaurants, driver starts, arrivals, customers, prep times, traffic noise)
is drawn here from the seed. The engine consumes no randomness afterwards, so every policy
sees an identical world and only its decisions differ.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from environment.config import EnvConfig
from environment.constants import GRID_SIZE, N_TICKS, TRAFFIC_LEVELS
from environment.city import manhattan


@dataclass(frozen=True)
class OrderSpec:
    created_at: int
    restaurant_id: int
    cx: float
    cy: float
    prep_actual: float
    prep_est: float
    deadline: float


@dataclass(frozen=True)
class Scenario:
    config: EnvConfig
    seed: int
    restaurants: tuple          # ((x, y), ...)
    driver_starts: tuple        # ((x, y), ...)
    orders: tuple               # OrderSpec, sorted by created_at
    traffic: np.ndarray         # multiplier per tick, length N_TICKS
    fingerprint: str = field(default="")


def _fingerprint(restaurants, drivers, orders, traffic) -> str:
    h = hashlib.sha256()
    h.update(np.round(np.array(restaurants, dtype=float), 6).tobytes())
    h.update(np.round(np.array(drivers, dtype=float), 6).tobytes())
    for o in orders:
        h.update(repr((o.created_at, o.restaurant_id, round(o.cx, 6), round(o.cy, 6),
                       round(o.prep_actual, 6), round(o.prep_est, 6), round(o.deadline, 6))).encode())
    h.update(np.round(traffic, 6).tobytes())
    return h.hexdigest()[:16]


def generate_scenario(cfg: EnvConfig, seed: int) -> Scenario:
    rng = np.random.default_rng([int(seed), 12345])
    G = GRID_SIZE

    # restaurants with minimum spacing (rejection sampling; relax if packed)
    rests = []
    spacing = cfg.min_restaurant_spacing
    tries = 0
    while len(rests) < cfg.n_restaurants:
        p = rng.uniform(1.0, G - 1.0, size=2)
        if all(manhattan(p[0], p[1], q[0], q[1]) >= spacing for q in rests):
            rests.append((float(p[0]), float(p[1])))
        tries += 1
        if tries % 200 == 0:
            spacing *= 0.8
    drivers = [tuple(map(float, rng.uniform(0.0, G, size=2))) for _ in range(cfg.n_drivers)]

    # traffic: AR(1) noise around the nominal level multiplier (policies only see the nominal level)
    level = TRAFFIC_LEVELS[cfg.traffic]
    noise = np.empty(N_TICKS)
    z = 0.0
    for i in range(N_TICKS):
        z = 0.9 * z + rng.normal(0.0, cfg.traffic_noise * np.sqrt(1 - 0.81))
        noise[i] = z
    traffic = np.clip(level * (1.0 + noise), 0.5 * level, 1.5 * level)

    # arrivals: backlog at t=0 plus a Poisson process over the horizon
    n_backlog = int(round(cfg.backlog_frac * cfg.orders_per_min))
    n_arrivals = int(rng.poisson(cfg.orders_per_min * N_TICKS / 60.0))
    times = np.concatenate([np.zeros(n_backlog), np.sort(rng.uniform(0, N_TICKS, n_arrivals))])
    times = np.floor(times).astype(int)
    times = np.minimum(times, N_TICKS - 1)

    speed = cfg.nominal_speed
    specs = []
    for ct in times:
        rid = int(rng.integers(0, cfg.n_restaurants))
        rx, ry = rests[rid]
        off = rng.uniform(-cfg.customer_radius, cfg.customer_radius, size=2)
        cx = float(np.clip(rx + off[0], 0.0, G))
        cy = float(np.clip(ry + off[1], 0.0, G))
        prep = float(np.clip(rng.normal(cfg.prep_mean, cfg.prep_std), cfg.prep_min, cfg.prep_max))
        est = float(prep * rng.lognormal(0.0, cfg.prep_est_noise))
        deadline = float(ct + est + manhattan(rx, ry, cx, cy) / speed + cfg.deadline_slack)
        specs.append(OrderSpec(int(ct), rid, cx, cy, prep, est, deadline))
    specs.sort(key=lambda s: s.created_at)

    fp = _fingerprint(rests, drivers, specs, traffic)
    return Scenario(cfg, int(seed), tuple(rests), tuple(drivers), tuple(specs), traffic, fp)
