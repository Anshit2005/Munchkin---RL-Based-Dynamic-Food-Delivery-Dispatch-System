"""Baseline 3 - Minimum Estimated Delivery Time: greedy argmin of the shared ETA over all
feasible (driver, order) pairs."""
from __future__ import annotations

from environment.constants import WAIT_ACTION


class MinETAPolicy:
    name = "baseline_3"
    label = "Baseline 3 (Min Est. Delivery Time)"

    def reset(self, env_config=None):
        pass

    def act(self, obs, info, view) -> int:
        pairs = view.valid_pairs()
        if not pairs:
            return WAIT_ACTION
        best = min(pairs, key=lambda p: (view.est(p[0], p[1]), p[1].created_at, p[0].id, p[1].id))
        return best[2]
