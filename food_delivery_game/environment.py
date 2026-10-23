"""A small discrete-time food-delivery simulation.

The environment deliberately exposes reset/step/state methods that can later be
used by baseline policies and RL agents. The temporary built-in dispatch rule
only keeps the first version of the game runnable; it is not an evaluated model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import SimulationConfig
from .entities import Driver, DriverStatus, Order, OrderStatus, Point, Restaurant


@dataclass(frozen=True)
class StepResult:
    time: int
    done: bool
    reward: float
    delivered_count: int
    pending_count: int


class DeliveryEnvironment:
    """Simulate orders and drivers for a fixed three-minute episode."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(self.config.seed if seed is None else seed)
        self.time = 0
        self.next_order_id = 1
        self.orders: list[Order] = []
        self.restaurants = [
            Restaurant(i, self._random_point()) for i in range(self.config.restaurant_count)
        ]
        self.drivers = [
            Driver(i, self._random_point()) for i in range(self.config.driver_count)
        ]
        self._arrival_credit = 0.0

    @property
    def done(self) -> bool:
        return self.time >= self.config.episode_minutes

    @property
    def delivered_orders(self) -> list[Order]:
        return [o for o in self.orders if o.status == OrderStatus.DELIVERED]

    @property
    def pending_orders(self) -> list[Order]:
        return [o for o in self.orders if o.status != OrderStatus.DELIVERED]

    def step(self) -> StepResult:
        """Advance one simulated minute and return a small reward and summary."""
        if self.done:
            return StepResult(self.time, True, 0.0, len(self.delivered_orders), len(self.pending_orders))

        self.time += 1
        self._generate_orders()
        for order in self.orders:
            if order.status == OrderStatus.PREPARING and order.ready_at <= self.time:
                order.status = OrderStatus.READY
        self._dispatch_ready_orders()
        delivered_before = len(self.delivered_orders)
        self._advance_drivers()
        reward = float(len(self.delivered_orders) - delivered_before)
        return StepResult(self.time, self.done, reward, len(self.delivered_orders), len(self.pending_orders))

    def state_summary(self) -> dict[str, int | float | bool]:
        return {
            "time": self.time,
            "episode_minutes": self.config.episode_minutes,
            "done": self.done,
            "orders": len(self.orders),
            "preparing": sum(o.status == OrderStatus.PREPARING for o in self.orders),
            "ready": sum(o.status == OrderStatus.READY for o in self.orders),
            "in_transit": sum(o.status in (OrderStatus.TO_RESTAURANT, OrderStatus.TO_CUSTOMER) for o in self.orders),
            "delivered": len(self.delivered_orders),
            "available_drivers": sum(d.status == DriverStatus.AVAILABLE for d in self.drivers),
        }

    def _random_point(self) -> Point:
        return Point(self.rng.uniform(3, self.config.width - 3), self.rng.uniform(3, self.config.height - 3))

    def _generate_orders(self) -> None:
        self._arrival_credit += self.config.orders_per_minute
        count = int(self._arrival_credit)
        self._arrival_credit -= count
        for _ in range(count):
            restaurant = self.rng.choice(self.restaurants)
            prep_time = self.rng.randint(
                self.config.preparation_min_minutes, self.config.preparation_max_minutes
            )
            self.orders.append(Order(
                order_id=self.next_order_id,
                restaurant_id=restaurant.restaurant_id,
                restaurant_location=restaurant.location,
                customer_location=self._random_point(),
                created_at=self.time,
                ready_at=self.time + prep_time,
                status=OrderStatus.PREPARING if prep_time else OrderStatus.READY,
            ))
            self.next_order_id += 1

    def _dispatch_ready_orders(self) -> None:
        # Temporary deterministic setup rule: oldest ready order to nearest free driver.
        available = [d for d in self.drivers if d.status == DriverStatus.AVAILABLE]
        ready = sorted((o for o in self.orders if o.status == OrderStatus.READY), key=lambda o: o.created_at)
        for order in ready:
            if not available:
                break
            driver = min(available, key=lambda d: d.location.distance_to(order.restaurant_location))
            available.remove(driver)
            order.assigned_driver_id = driver.driver_id
            order.status = OrderStatus.TO_RESTAURANT
            driver.order_id = order.order_id
            driver.status = DriverStatus.TO_RESTAURANT
            driver.destination = order.restaurant_location
            driver.remaining_distance = driver.location.distance_to(order.restaurant_location)

    def _advance_drivers(self) -> None:
        for driver in self.drivers:
            if driver.status == DriverStatus.AVAILABLE or driver.destination is None:
                continue
            driver.remaining_distance -= self.config.travel_speed
            if driver.remaining_distance > 0:
                continue
            driver.location = driver.destination
            order = next((o for o in self.orders if o.order_id == driver.order_id), None)
            if order is None:
                self._release_driver(driver)
            elif driver.status == DriverStatus.TO_RESTAURANT:
                order.status = OrderStatus.TO_CUSTOMER
                driver.status = DriverStatus.TO_CUSTOMER
                driver.destination = order.customer_location
                driver.remaining_distance = order.restaurant_location.distance_to(order.customer_location)
                if driver.remaining_distance <= 0:
                    self._deliver(order, driver)
            else:
                self._deliver(order, driver)

    def _deliver(self, order: Order, driver: Driver) -> None:
        order.status = OrderStatus.DELIVERED
        order.delivered_at = self.time
        self._release_driver(driver)

    @staticmethod
    def _release_driver(driver: Driver) -> None:
        driver.status = DriverStatus.AVAILABLE
        driver.order_id = None
        driver.destination = None
        driver.remaining_distance = 0.0
