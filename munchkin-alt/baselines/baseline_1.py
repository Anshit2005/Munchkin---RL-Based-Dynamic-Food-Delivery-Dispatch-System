"""Baseline 1 - Nearest Driver: serve the oldest order with the closest feasible driver."""
from __future__ import annotations

from environment.city import manhattan
from environment.constants import WAIT_ACTION


class NearestDriverPolicy:
    name = "baseline_1"
    label = "Baseline 1 (Nearest Driver)"

    def reset(self, env_config=None):
        pass

    def act(self, obs, info, view) -> int:
        pairs = view.valid_pairs()
        if not pairs:
            return WAIT_ACTION
        oldest = min({id(o): o for _, o, _ in pairs}.values(), key=lambda o: (o.created_at, o.id))
        best = min((p for p in pairs if p[1] is oldest),
                   key=lambda p: (manhattan(p[0].x, p[0].y, oldest.rx, oldest.ry), p[0].id))
        return best[2]
