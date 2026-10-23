"""Data objects used by the delivery environment."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class OrderStatus(str, Enum):
    PREPARING = "Preparing"
    READY = "Ready"
    TO_RESTAURANT = "To restaurant"
    TO_CUSTOMER = "To customer"
    DELIVERED = "Delivered"


class DriverStatus(str, Enum):
    AVAILABLE = "Available"
    TO_RESTAURANT = "To restaurant"
    TO_CUSTOMER = "To customer"


@dataclass
class Restaurant:
    restaurant_id: int
    location: Point


@dataclass
class Driver:
    driver_id: int
    location: Point
    status: DriverStatus = DriverStatus.AVAILABLE
    order_id: int | None = None
    destination: Point | None = None
    remaining_distance: float = 0.0


@dataclass
class Order:
    order_id: int
    restaurant_id: int
    restaurant_location: Point
    customer_location: Point
    created_at: int
    ready_at: int
    status: OrderStatus
    assigned_driver_id: int | None = None
    delivered_at: int | None = None
