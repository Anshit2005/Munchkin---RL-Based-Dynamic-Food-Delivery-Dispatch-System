"""A small discrete-time food-delivery simulation."""

from __future__ import annotations

import random
import math
from dataclasses import dataclass

from .config import SimulationConfig, TRAFFIC_MULTIPLIERS
from .entities import Driver, DriverStatus, Order, OrderStatus, Point, Restaurant
from .reward import RewardConfig, minute_reward
from .baselines.base import DispatchPolicy



@dataclass(frozen=True)
class StepResult:
    time: int
    done: bool
    reward: float
    delivered_count: int
    pending_count: int
    score_reward: float = 0.0


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

    def __init__(self, config: SimulationConfig | None = None,
                 reward_config: RewardConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.reward_config = reward_config or RewardConfig()
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(self.config.seed if seed is None else seed)
        self.time = 0
        self.next_order_id = 1
        self.orders: list[Order] = []
        self.dispatch_log: list[DispatchEvent] = []
        self.total_travel_distance = 0.0
        self.total_reward = 0.0
        self.total_training_reward = 0.0
        self.reward_breakdown = {key: 0.0 for key in
                                 ("delivery", "wait", "late", "travel", "batch", "idle", "terminal", "invalid")}
        self._tick_prepared = False
        restaurant_points = self._restaurant_points()
        self.restaurants = [Restaurant(i, point) for i, point in enumerate(restaurant_points)]
        self.drivers = [
            Driver(i, self._random_point()) for i in range(self.config.driver_count)
        ]
        self._sample_scenario()

    @property
    def done(self) -> bool:
        return self.time >= self.config.episode_minutes

    @property
    def delivered_orders(self) -> list[Order]:
        return [o for o in self.orders if o.status == OrderStatus.DELIVERED]

    @property
    def pending_orders(self) -> list[Order]:
        return [o for o in self.orders if o.status != OrderStatus.DELIVERED]

    @property
    def dispatchable_orders(self) -> list[Order]:
        return [o for o in self.orders if o.status in (OrderStatus.PREPARING, OrderStatus.READY)]

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
        distance_before = self.total_travel_distance
        self._advance_drivers()
        self.time += 1
        self._tick_prepared = False
        newly_delivered = len(self.delivered_orders) - delivered_before
        score_reward = self.DELIVERY_REWARD_POINTS * newly_delivered
        overdue = sum(order.deadline_at < self.time for order in self.pending_orders)
        idle = min(sum(d.status == DriverStatus.AVAILABLE for d in self.drivers),
                   len(self.dispatchable_orders))
        parts = minute_reward(
            self.reward_config,
            delivered=newly_delivered,
            waiting=len(self.pending_orders),
            overdue=overdue,
            distance=self.total_travel_distance - distance_before,
            idle=idle,
            unfinished=len(self.pending_orders) if self.done else 0,
        )
        reward = sum(parts.values())
        self.total_reward += score_reward
        self.total_training_reward += reward
        for key, value in parts.items():
            self.reward_breakdown[key] += value
        return StepResult(self.time, self.done, reward, len(self.delivered_orders),
                          len(self.pending_orders), score_reward)

    def state_summary(self) -> dict[str, int | float | bool]:
        return {
            "time": self.time,
            "episode_minutes": self.config.episode_minutes,
            "done": self.done,
            "orders": len(self.orders),
            "preparing": sum(o.status == OrderStatus.PREPARING for o in self.orders),
            "ready": sum(o.status == OrderStatus.READY for o in self.orders),
            "in_transit": sum(o.status in (OrderStatus.TO_RESTAURANT, OrderStatus.TO_CUSTOMER) for o in self.orders),
            "drivers_waiting_for_food": sum(d.status == DriverStatus.WAITING_FOR_FOOD for d in self.drivers),
            "delivered": len(self.delivered_orders),
            "available_drivers": sum(d.status == DriverStatus.AVAILABLE for d in self.drivers),
            "total_travel_distance": self.total_travel_distance,
            "total_reward": self.total_reward,
            "total_training_reward": self.total_training_reward,
            "overdue": sum(o.deadline_at < self.time for o in self.pending_orders),
            "traffic_multiplier": self.traffic_profile[min(self.time, len(self.traffic_profile) - 1)],
        }

    def assign_order(self, driver_id: int, order_id: int) -> bool:
        """Assign a preparing or ready order to an available driver."""
        driver = next((d for d in self.drivers if d.driver_id == driver_id), None)
        order = next((o for o in self.orders if o.order_id == order_id), None)
        if driver is None or order is None:
            return False
        if driver.status != DriverStatus.AVAILABLE or order.status not in (OrderStatus.PREPARING, OrderStatus.READY):
            return False

        order.assigned_driver_id = driver.driver_id
        order.assigned_at = self.time
        order.status = OrderStatus.TO_RESTAURANT
        driver.order_id = order.order_id
        driver.status = DriverStatus.TO_RESTAURANT
        driver.destination = order.restaurant_location
        driver.remaining_distance = driver.location.distance_to(order.restaurant_location)
        return True

    def _random_point(self) -> Point:
        return Point(self.rng.uniform(3, self.config.width - 3), self.rng.uniform(3, self.config.height - 3))

    def _restaurant_points(self) -> list[Point]:
        """Place restaurants with minimum spacing, relaxing it if the map is crowded."""
        points: list[Point] = []
        spacing = self.config.min_restaurant_spacing
        attempts = 0
        while len(points) < self.config.restaurant_count:
            candidate = self._random_point()
            if all(candidate.distance_to(existing) >= spacing for existing in points):
                points.append(candidate)
                continue
            attempts += 1
            if attempts % 200 == 0:
                spacing *= 0.8
        return points

    def _sample_scenario(self) -> None:
        """Pre-sample all exogenous events so every strategy sees the same seeded city."""
        cfg = self.config
        base_traffic = TRAFFIC_MULTIPLIERS[cfg.traffic_level]
        z = 0.0
        self.traffic_profile = []
        innovation_scale = cfg.traffic_noise * (1.0 - 0.9 ** 2) ** 0.5
        for _ in range(cfg.episode_minutes):
            z = 0.9 * z + self.rng.gauss(0.0, innovation_scale)
            self.traffic_profile.append(base_traffic * max(0.5, min(1.5, 1.0 + z)))

        self.scenario_orders: dict[int, list[Order]] = {minute: [] for minute in range(cfg.episode_minutes)}
        initial_backlog = int(round(cfg.backlog_fraction * cfg.orders_per_minute))
        nominal_speed = cfg.travel_speed * base_traffic
        for minute in range(cfg.episode_minutes):
            count = self._poisson(cfg.orders_per_minute)
            if minute == 0:
                count += initial_backlog
            for _ in range(count):
                restaurant = self.rng.choice(self.restaurants)
                radius = cfg.customer_radius
                customer = Point(
                    max(3.0, min(cfg.width - 3.0,
                                 restaurant.location.x + self.rng.uniform(-radius, radius))),
                    max(3.0, min(cfg.height - 3.0,
                                 restaurant.location.y + self.rng.uniform(-radius, radius))),
                )
                prep_actual = max(cfg.preparation_min_minutes,
                                  min(cfg.preparation_max_minutes,
                                      self.rng.gauss(cfg.preparation_mean_minutes,
                                                     cfg.preparation_std_minutes)))
                prep_estimate = prep_actual * self.rng.lognormvariate(
                    0.0, cfg.preparation_estimate_noise
                )
                # The deadline estimate is scenario-level (prep + restaurant-to-customer
                # travel), matching munchkin-alt. Actual pickup travel still affects lateness.
                estimated_trip = restaurant.location.distance_to(customer) / nominal_speed
                order = Order(
                    order_id=self.next_order_id,
                    restaurant_id=restaurant.restaurant_id,
                    restaurant_location=restaurant.location,
                    customer_location=customer,
                    created_at=minute,
                    ready_at=minute + prep_actual,
                    status=OrderStatus.PREPARING if prep_actual > 0 else OrderStatus.READY,
                    preparation_minutes=prep_actual,
                    preparation_estimate_minutes=prep_estimate,
                    deadline_at=(minute + prep_estimate + estimated_trip + cfg.deadline_slack_minutes),
                )
                self.scenario_orders[minute].append(order)
                self.next_order_id += 1

    def _generate_orders(self) -> None:
        self.orders.extend(self.scenario_orders.get(self.time, ()))

    def _poisson(self, rate: float) -> int:
        """Sample a Poisson arrival count with the standard-library RNG."""
        if rate <= 0:
            return 0
        threshold = math.exp(-rate)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= self.rng.random()
        return count - 1

    def _advance_drivers(self) -> None:
        for driver in self.drivers:
            if driver.status == DriverStatus.AVAILABLE or driver.destination is None:
                continue
            speed = self.config.travel_speed * self.traffic_profile[min(self.time, len(self.traffic_profile) - 1)]
            travel_budget = speed
            elapsed = 0.0
            if driver.status == DriverStatus.WAITING_FOR_FOOD:
                order = next((o for o in self.orders if o.order_id == driver.order_id), None)
                if order is None:
                    self._release_driver(driver)
                    continue
                prep_wait = max(0.0, order.ready_at - self.time)
                if prep_wait >= travel_budget / speed:
                    continue
                travel_budget -= prep_wait * speed
                elapsed += prep_wait
                self._send_driver_to_customer(driver, order)
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
                    elapsed += distance_travelled / speed
                    prep_wait = max(0.0, order.ready_at - (self.time + elapsed))
                    if prep_wait >= travel_budget / speed:
                        driver.status = DriverStatus.WAITING_FOR_FOOD
                        driver.destination = order.restaurant_location
                        driver.remaining_distance = 0.0
                        break
                    travel_budget -= prep_wait * speed
                    elapsed += prep_wait
                    self._send_driver_to_customer(driver, order)
                    if driver.remaining_distance <= 0:
                        self._deliver(order, driver, self.time + elapsed)
                else:
                    self._deliver(order, driver, self.time + elapsed)

    def _deliver(self, order: Order, driver: Driver, delivered_at: float | None = None) -> None:
        order.status = OrderStatus.DELIVERED
        order.delivered_at = self.time + 1 if delivered_at is None else delivered_at
        self._release_driver(driver)

    @staticmethod
    def _send_driver_to_customer(driver: Driver, order: Order) -> None:
        order.status = OrderStatus.TO_CUSTOMER
        driver.status = DriverStatus.TO_CUSTOMER
        driver.destination = order.customer_location
        driver.remaining_distance = order.restaurant_location.distance_to(order.customer_location)

    @staticmethod
    def _release_driver(driver: Driver) -> None:
        driver.status = DriverStatus.AVAILABLE
        driver.order_id = None
        driver.destination = None
        driver.remaining_distance = 0.0
