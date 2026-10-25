"""A small discrete-time food-delivery simulation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import SimulationConfig
from .entities import Driver, DriverStatus, Order, OrderStatus, Point, Restaurant
from .baselines.base import DispatchPolicy


@dataclass(frozen=True)
class StepResult:
    time: int
    done: bool
    reward: float
    delivered_count: int
    pending_count: int


@dataclass(frozen=True)
class DispatchEvent:
    time: int
    strategy: str
    driver_id: int
    order_id: int
    explanation: str


class DeliveryEnvironment:
    """Simulate orders and drivers for the configured episode horizon."""

    DELIVERY_REWARD_POINTS = 10.0

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(self.config.seed if seed is None else seed)
        self.time = 0
        self.next_order_id = 1
        self.orders: list[Order] = []
        self.dispatch_log: list[DispatchEvent] = []
        self.total_travel_distance = 0.0
        self.total_reward = 0.0
        self._tick_prepared = False
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

    def step(self, policy: DispatchPolicy | None = None) -> StepResult:
        """Advance one simulated minute and return a small reward and summary."""
        if self.done:
            return StepResult(self.time, True, 0.0, len(self.delivered_orders), len(self.pending_orders))

        self.prepare_minute()
        if policy is not None:
            while True:
                assignment = policy.select_assignment(self)
                if assignment is None:
                    break
                if not self.assign_order(assignment.driver_id, assignment.order_id):
                    raise ValueError(f"{policy.name} selected an invalid assignment: {assignment}")
                self.dispatch_log.append(DispatchEvent(
                    time=self.time,
                    strategy=policy.name,
                    driver_id=assignment.driver_id,
                    order_id=assignment.order_id,
                    explanation=assignment.explanation,
                ))
        return self.advance_minute()

    def prepare_minute(self) -> None:
        """Create this minute's orders and mark food ready; safe to call only once per minute."""
        if self.done:
            return
        if self._tick_prepared:
            return
        self._generate_orders()
        for order in self.orders:
            if order.status == OrderStatus.PREPARING and order.ready_at <= self.time:
                order.status = OrderStatus.READY
        self._tick_prepared = True

    def advance_minute(self) -> StepResult:
        """Move drivers for one minute and advance the episode clock."""
        if self.done:
            return StepResult(self.time, True, 0.0, len(self.delivered_orders), len(self.pending_orders))
        self.prepare_minute()
        delivered_before = len(self.delivered_orders)
        self._advance_drivers()
        self.time += 1
        self._tick_prepared = False
        reward = self.DELIVERY_REWARD_POINTS * (len(self.delivered_orders) - delivered_before)
        self.total_reward += reward
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
            "total_travel_distance": self.total_travel_distance,
            "total_reward": self.total_reward,
        }

    def assign_order(self, driver_id: int, order_id: int) -> bool:
        """Assign a ready order to an available driver; return False if invalid."""
        driver = next((d for d in self.drivers if d.driver_id == driver_id), None)
        order = next((o for o in self.orders if o.order_id == order_id), None)
        if driver is None or order is None:
            return False
        if driver.status != DriverStatus.AVAILABLE or order.status != OrderStatus.READY:
            return False

        order.assigned_driver_id = driver.driver_id
        order.status = OrderStatus.TO_RESTAURANT
        driver.order_id = order.order_id
        driver.status = DriverStatus.TO_RESTAURANT
        driver.destination = order.restaurant_location
        driver.remaining_distance = driver.location.distance_to(order.restaurant_location)
        return True

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

    def _advance_drivers(self) -> None:
        for driver in self.drivers:
            if driver.status == DriverStatus.AVAILABLE or driver.destination is None:
                continue
            travel_budget = self.config.travel_speed
            while travel_budget > 0 and driver.status != DriverStatus.AVAILABLE:
                if driver.destination is None:
                    break
                distance_to_destination = driver.remaining_distance
                distance_travelled = min(travel_budget, distance_to_destination)
                self.total_travel_distance += distance_travelled
                travel_budget -= distance_travelled
                if distance_travelled < distance_to_destination:
                    fraction = distance_travelled / distance_to_destination
                    driver.location = Point(
                        driver.location.x + (driver.destination.x - driver.location.x) * fraction,
                        driver.location.y + (driver.destination.y - driver.location.y) * fraction,
                    )
                    driver.remaining_distance -= distance_travelled
                    break

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
        order.delivered_at = self.time + 1
        self._release_driver(driver)

    @staticmethod
    def _release_driver(driver: Driver) -> None:
        driver.status = DriverStatus.AVAILABLE
        driver.order_id = None
        driver.destination = None
        driver.remaining_distance = 0.0
