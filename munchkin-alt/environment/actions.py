"""Action space helpers and the read-only DispatchView given to heuristic policies."""
from __future__ import annotations

import numpy as np

from environment.constants import (MAX_DRIVERS, MAX_ORDERS, N_ACTIONS, WAIT_ACTION,
                                   encode_action)
from environment.estimates import est_completion


def compute_mask(engine, slots) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[WAIT_ACTION] = True
    for i, d in enumerate(engine.drivers):
        for j, o in enumerate(slots):
            if engine.can_assign(d, o):
                mask[encode_action(i, j)] = True
    return mask


class DispatchView:
    """Observable, structured snapshot for baselines. Same information the RL agent gets:
    estimated (not true) prep times and the nominal traffic level only."""

    def __init__(self, engine, slots, mask):
        self.t = engine.t
        self.cfg = engine.cfg
        self.speed = engine.cfg.nominal_speed
        self.drivers = engine.drivers
        self.orders = slots                 # unassigned orders, slot order (max MAX_ORDERS)
        self.mask = mask
        self.max_drivers = MAX_DRIVERS
        self.max_orders = MAX_ORDERS

    def valid_pairs(self):
        """List of (driver, order, action_index) for every valid assignment."""
        out = []
        for i, d in enumerate(self.drivers):
            for j, o in enumerate(self.orders):
                a = encode_action(i, j)
                if self.mask[a]:
                    out.append((d, o, a))
        return out

    def est(self, driver, order) -> float:
        return est_completion(driver, order, self.t, self.speed)
