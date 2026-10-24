"""Order entity. Status is derived from timestamps (single source of truth)."""
from __future__ import annotations

from typing import Optional

from environment.constants import OrderStatus


class Order:
    __slots__ = ("id", "restaurant_id", "rx", "ry", "cx", "cy", "created_at", "prep_actual",
                 "prep_est", "deadline", "ready_at", "assigned_at", "driver_id",
                 "arrived_at_rest", "picked_up_at", "delivered_at")

    def __init__(self, id: int, restaurant_id: int, rx: float, ry: float, cx: float, cy: float,
                 created_at: int, prep_actual: float, prep_est: float, deadline: float):
        self.id = id
        self.restaurant_id = restaurant_id
        self.rx, self.ry, self.cx, self.cy = rx, ry, cx, cy
        self.created_at = created_at
        self.prep_actual = prep_actual      # hidden from policies
        self.prep_est = prep_est            # observable estimate
        self.deadline = deadline
        self.ready_at: Optional[int] = None
        self.assigned_at: Optional[int] = None
        self.driver_id: Optional[int] = None
        self.arrived_at_rest: Optional[int] = None
        self.picked_up_at: Optional[int] = None
        self.delivered_at: Optional[int] = None

    @property
    def status(self) -> OrderStatus:
        if self.delivered_at is not None:
            return OrderStatus.DELIVERED
        if self.picked_up_at is not None:
            return OrderStatus.PICKED_UP
        if self.assigned_at is not None:
            return OrderStatus.ASSIGNED
        if self.ready_at is not None:
            return OrderStatus.READY
        return OrderStatus.PREPARING

    @property
    def is_ready(self) -> bool:
        return self.ready_at is not None

    @property
    def unassigned(self) -> bool:
        return self.assigned_at is None

    def age(self, t: int) -> int:
        return t - self.created_at

    def est_ready_remaining(self, t: float) -> float:
        """Observable estimate of seconds until the food is ready (0 if ready)."""
        if self.ready_at is not None:
            return 0.0
        return max(0.0, self.created_at + self.prep_est - t)

    def is_late(self) -> bool:
        return self.delivered_at is not None and self.delivered_at > self.deadline

    # --- lifecycle segments (None until both endpoints exist) ---
    def segments(self) -> dict:
        def d(a, b):
            return None if a is None or b is None else float(a - b)
        return {
            "prep": d(self.ready_at, self.created_at),
            "wait_for_assignment": d(self.assigned_at, self.created_at),
            "to_restaurant": d(self.arrived_at_rest, self.assigned_at),
            "wait_at_restaurant": d(self.picked_up_at, self.arrived_at_rest),
            "to_customer": d(self.delivered_at, self.picked_up_at),
        }
