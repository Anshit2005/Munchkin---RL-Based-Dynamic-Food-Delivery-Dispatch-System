"""Episode metrics, computed from the final order/driver logs (independent of the reward code)."""
from __future__ import annotations

import math

from environment.constants import N_TICKS

METRIC_NAMES = ("avg_delivery_time", "avg_wait_time", "on_time_pct", "late_pct", "total_distance",
                "utilization", "batch_count", "completion_rate", "undelivered", "n_orders",
                "n_delivered")

LIFECYCLE_KEYS = ("prep", "wait_for_assignment", "to_restaurant", "wait_at_restaurant", "to_customer")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else math.nan


def compute_metrics(engine) -> dict:
    orders = engine.orders
    delivered = [o for o in orders if o.delivered_at is not None]
    n_created, n_deliv = len(orders), len(delivered)
    on_time = sum(1 for o in delivered if o.delivered_at <= o.deadline)
    late = n_deliv - on_time

    m = {
        "n_orders": n_created,
        "n_delivered": n_deliv,
        "undelivered": n_created - n_deliv,
        # 18.1: mean (delivered_at - created_at) over delivered orders
        "avg_delivery_time": _mean([o.delivered_at - o.created_at for o in delivered]),
        # 18.2: customer waiting, censored at the horizon for undelivered orders
        "avg_wait_time": _mean([min(o.delivered_at if o.delivered_at is not None else N_TICKS, N_TICKS)
                                - o.created_at for o in orders]),
        "on_time_pct": 100.0 * on_time / n_deliv if n_deliv else math.nan,
        "late_pct": 100.0 * late / n_deliv if n_deliv else math.nan,
        "total_distance": float(sum(d.distance for d in engine.drivers)),
        "utilization": 100.0 * sum(d.busy_ticks for d in engine.drivers) / (len(engine.drivers) * N_TICKS),
        "batch_count": sum(1 for b in engine.batches
                           if b["saved"] > 0 and all(o.delivered_at is not None for o in b["orders"])),
        "completion_rate": 100.0 * n_deliv / n_created if n_created else math.nan,
    }
    segs = [o.segments() for o in orders]
    for k in LIFECYCLE_KEYS:
        m["lc_" + k] = _mean([s[k] for s in segs])
    return m
