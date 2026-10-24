"""Delivery partner entity."""
from __future__ import annotations

from environment.constants import DriverStatus

PICKUP = 0
DROPOFF = 1


class Waypoint:
    __slots__ = ("x", "y", "kind", "order")

    def __init__(self, x: float, y: float, kind: int, order):
        self.x, self.y, self.kind, self.order = x, y, kind, order


class Driver:
    __slots__ = ("id", "x", "y", "status", "orders", "route", "distance", "busy_ticks", "capacity")

    def __init__(self, id: int, x: float, y: float, capacity: int = 1):
        self.id = id
        self.x, self.y = x, y
        self.status = DriverStatus.AVAILABLE
        self.orders: list = []          # assigned, not yet delivered
        self.route: list = []           # remaining waypoints
        self.distance = 0.0
        self.busy_ticks = 0
        self.capacity = capacity

    @property
    def load(self) -> int:
        return len(self.orders)

    @property
    def available(self) -> bool:
        return self.status == DriverStatus.AVAILABLE
