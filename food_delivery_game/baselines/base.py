"""Shared interfaces for dispatch policies."""

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment import DeliveryEnvironment


@dataclass(frozen=True)
class Assignment:
    driver_id: int
    order_id: int
    explanation: str


class DispatchPolicy(Protocol):
    name: str

    def select_assignment(self, env: "DeliveryEnvironment") -> Assignment | None:
        """Choose one valid ready-order/available-driver pair, or no action."""


def estimated_completion_minutes(env: "DeliveryEnvironment", driver, order) -> float:
    """Estimate trip time using public prep estimates and nominal traffic only."""
    speed = env.config.nominal_speed
    pickup_minutes = driver.location.distance_to(order.restaurant_location) / speed
    prep_remaining = max(order.estimated_ready_at - env.time, 0.0)
    delivery_minutes = order.restaurant_location.distance_to(order.customer_location) / speed
    return max(pickup_minutes, prep_remaining) + delivery_minutes
