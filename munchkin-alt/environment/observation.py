"""Fixed-size observation vector (OBS_DIM = 481), all features hand-normalised to ~[-1, 2].

Layout (OBS_VERSION = 1):
  [0:120)    12 driver slots x 10: present, x, y, available, load/cap, status one-hot(4), time_until_free
  [120:280)  16 order slots  x 10: present, rest_x, rest_y, cust_x, cust_y, age, est_prep_remaining,
                                   ready, time_to_deadline, restaurant_load
  [280:472)  12x16 pair features: estimated completion time of (driver, order), floored at 1e-3
             (exactly 0 means the pair is invalid)
  [472:481)  globals (9)
Hidden quantities (true prep time, traffic noise) are never included.
"""
from __future__ import annotations

import numpy as np

from environment.constants import (DRIVER_FEATS, GLOBAL_FEATS, GRID_SIZE, MAX_DRIVERS, MAX_ORDERS,
                                   N_TICKS, OBS_DIM, ORDER_FEATS, TRAFFIC_LEVELS, DriverStatus)
from environment.estimates import est_completion, est_time_until_free

DRV_OFF = 0
ORD_OFF = MAX_DRIVERS * DRIVER_FEATS
PAIR_OFF = ORD_OFF + MAX_ORDERS * ORDER_FEATS
GLB_OFF = PAIR_OFF + MAX_DRIVERS * MAX_ORDERS


def build_observation(engine, slots, mask, assign_count: int) -> np.ndarray:
    cfg = engine.cfg
    t = engine.t
    speed = cfg.nominal_speed
    G = GRID_SIZE
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    n_avail = 0
    for i, d in enumerate(engine.drivers):
        b = DRV_OFF + i * DRIVER_FEATS
        obs[b] = 1.0
        obs[b + 1] = d.x / G
        obs[b + 2] = d.y / G
        avail = d.status == DriverStatus.AVAILABLE
        n_avail += avail
        obs[b + 3] = 1.0 if avail else 0.0
        obs[b + 4] = d.load / d.capacity
        obs[b + 5 + int(d.status)] = 1.0
        obs[b + 9] = min(est_time_until_free(d, t, speed) / N_TICKS, 1.0)

    rest_count = {}
    for o in slots:
        rest_count[o.restaurant_id] = rest_count.get(o.restaurant_id, 0) + 1
    for j, o in enumerate(slots):
        b = ORD_OFF + j * ORDER_FEATS
        obs[b] = 1.0
        obs[b + 1] = o.rx / G
        obs[b + 2] = o.ry / G
        obs[b + 3] = o.cx / G
        obs[b + 4] = o.cy / G
        obs[b + 5] = min((t - o.created_at) / N_TICKS, 1.0)
        obs[b + 6] = min(o.est_ready_remaining(t) / 60.0, 1.0)
        obs[b + 7] = 1.0 if o.is_ready else 0.0
        obs[b + 8] = max(-1.0, min((o.deadline - t) / N_TICKS, 1.0))
        obs[b + 9] = (rest_count[o.restaurant_id] - 1) / MAX_ORDERS

    for i, d in enumerate(engine.drivers):
        for j, o in enumerate(slots):
            a = i * MAX_ORDERS + j
            if mask[a]:
                obs[PAIR_OFF + a] = max(min(est_completion(d, o, t, speed) / 90.0, 2.0), 1e-3)

    overflow = max(0, len(engine.pending) - MAX_ORDERS)
    g = GLB_OFF
    obs[g] = t / N_TICKS
    obs[g + 1] = (N_TICKS - t) / N_TICKS
    obs[g + 2] = min(len(engine.pending) / MAX_ORDERS, 1.0)
    obs[g + 3] = n_avail / MAX_DRIVERS
    obs[g + 4] = min(len(engine.preparing) / MAX_ORDERS, 1.0)
    obs[g + 5] = cfg.orders_per_min / 10.0
    obs[g + 6] = TRAFFIC_LEVELS[cfg.traffic] - 1.0
    obs[g + 7] = min(overflow / MAX_ORDERS, 1.0)
    obs[g + 8] = assign_count / MAX_DRIVERS
    return np.clip(obs, -2.0, 2.0)


assert GLB_OFF + GLOBAL_FEATS == OBS_DIM
