"""Baseline 2 - Oldest / Most Urgent Order: serve the order with least slack first.

Urgency slack = deadline - now - (best estimated completion time over feasible drivers).
Ties go to the oldest order. The order is then given to the nearest feasible driver
(kept distinct from Baseline 3, which minimises estimated delivery time).
"""
from __future__ import annotations

from environment.city import manhattan
from environment.constants import WAIT_ACTION


class MostUrgentOrderPolicy:
    name = "baseline_2"
    label = "Baseline 2 (Most Urgent Order)"

    def reset(self, env_config=None):
        pass

    def act(self, obs, info, view) -> int:
        pairs = view.valid_pairs()
        if not pairs:
            return WAIT_ACTION
        best_eta = {}
        for d, o, _ in pairs:
            e = view.est(d, o)
            if o.id not in best_eta or e < best_eta[o.id]:
                best_eta[o.id] = e
        orders = {p[1].id: p[1] for p in pairs}
        urgent = min(orders.values(),
                     key=lambda o: (o.deadline - view.t - best_eta[o.id], o.created_at, o.id))
        best = min((p for p in pairs if p[1] is urgent),
                   key=lambda p: (manhattan(p[0].x, p[0].y, urgent.rx, urgent.ry), p[0].id))
        return best[2]
