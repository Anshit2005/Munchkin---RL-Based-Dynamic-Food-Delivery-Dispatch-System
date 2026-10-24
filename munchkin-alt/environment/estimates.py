"""Shared ETA estimator, built only from observable quantities.

Used by Baseline 3, the observation builder (pair features) and the UI so the RL agents and
the heuristics have the same information. It uses the nominal traffic level (the per-tick
traffic noise is unobservable) and the estimated prep time (the true prep time is hidden).
"""
from __future__ import annotations

from environment.city import manhattan
from environment.driver import PICKUP
from environment.constants import DriverStatus


def dropoff_order(rx: float, ry: float, orders: list) -> list:
    """Order in which to drop off 1-2 orders picked up at (rx, ry), minimising route length."""
    if len(orders) < 2:
        return list(orders)
    a, b = orders
    len_ab = manhattan(rx, ry, a.cx, a.cy) + manhattan(a.cx, a.cy, b.cx, b.cy)
    len_ba = manhattan(rx, ry, b.cx, b.cy) + manhattan(b.cx, b.cy, a.cx, a.cy)
    return [a, b] if len_ab <= len_ba else [b, a]


def est_completion(driver, order, t: float, speed: float) -> float:
    """Estimated seconds from now until `order` is delivered if `driver` is assigned to it now."""
    rx, ry = order.rx, order.ry
    arrive = manhattan(driver.x, driver.y, rx, ry) / speed
    if driver.status == DriverStatus.AVAILABLE or not driver.orders:
        depart = max(arrive, order.est_ready_remaining(t))
        return depart + manhattan(rx, ry, order.cx, order.cy) / speed
    # batching onto a driver already heading to the same restaurant
    o1 = driver.orders[0]
    depart = max(arrive, o1.est_ready_remaining(t), order.est_ready_remaining(t))
    seq = dropoff_order(rx, ry, [o1, order])
    pos = (rx, ry)
    clock = depart
    for o in seq:
        clock += manhattan(pos[0], pos[1], o.cx, o.cy) / speed
        pos = (o.cx, o.cy)
        if o is order:
            return clock
    return clock


def est_time_until_free(driver, t: float, speed: float) -> float:
    """Estimated seconds until the driver finishes its current route (0 if available)."""
    if not driver.route:
        return 0.0
    cx, cy = driver.x, driver.y
    clock = 0.0
    for wp in driver.route:
        clock += manhattan(cx, cy, wp.x, wp.y) / speed
        cx, cy = wp.x, wp.y
        if wp.kind == PICKUP:
            clock = max(clock, wp.order.est_ready_remaining(t))
    return clock
