"""Fixed-size observation and masked action encoding for learned dispatch policies."""

from __future__ import annotations

import numpy as np

from ..entities import DriverStatus, OrderStatus
from .version import OBS_VERSION

MAX_DRIVERS = 16
MAX_ORDERS = 64
DRIVER_FEATURES = 11
ORDER_FEATURES = 10
GLOBAL_FEATURES = 9
WAIT_ACTION = MAX_DRIVERS * MAX_ORDERS
N_ACTIONS = WAIT_ACTION + 1
DRIVER_OFFSET = 0
ORDER_OFFSET = MAX_DRIVERS * DRIVER_FEATURES
PAIR_OFFSET = ORDER_OFFSET + MAX_ORDERS * ORDER_FEATURES
GLOBAL_OFFSET = PAIR_OFFSET + MAX_DRIVERS * MAX_ORDERS
OBS_SIZE = GLOBAL_OFFSET + GLOBAL_FEATURES


def dispatchable_orders(env):
    return sorted(
        (order for order in env.dispatchable_orders),
        key=lambda order: (order.created_at, order.order_id),
    )


def encode_state(env):
    """Return the fixed observation and action mask for the current prepared minute."""
    if len(env.drivers) > MAX_DRIVERS:
        raise ValueError(f"DQN/PPO support at most {MAX_DRIVERS} delivery partners")
    cfg = env.config
    horizon = max(cfg.episode_minutes, 1)
    all_dispatchable = dispatchable_orders(env)
    orders = all_dispatchable[:MAX_ORDERS]
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[WAIT_ACTION] = True

    available_count = 0
    available_drivers = []
    for i, driver in enumerate(env.drivers):
        base = DRIVER_OFFSET + i * DRIVER_FEATURES
        obs[base] = 1.0
        obs[base + 1] = driver.location.x / cfg.width
        obs[base + 2] = driver.location.y / cfg.height
        available = driver.status == DriverStatus.AVAILABLE
        available_count += int(available)
        if available:
            available_drivers.append(driver)
        obs[base + 3] = float(available)
        obs[base + 4] = float(driver.status == DriverStatus.TO_RESTAURANT)
        obs[base + 5] = float(driver.status == DriverStatus.TO_CUSTOMER)
        if driver.destination is not None:
            obs[base + 6] = driver.destination.x / cfg.width
            obs[base + 7] = driver.destination.y / cfg.height
        obs[base + 8] = min(driver.remaining_distance / (cfg.width + cfg.height), 1.0)
        obs[base + 10] = float(driver.status == DriverStatus.WAITING_FOR_FOOD)
        speed = cfg.nominal_speed
        time_until_free = driver.remaining_distance / speed
        if driver.status in (DriverStatus.TO_RESTAURANT, DriverStatus.WAITING_FOR_FOOD) and driver.order_id is not None:
            assigned = next((item for item in env.orders if item.order_id == driver.order_id), None)
            if assigned is not None:
                pickup_eta = (driver.remaining_distance / speed
                              if driver.status == DriverStatus.TO_RESTAURANT else 0.0)
                prep_remaining = max(assigned.estimated_ready_at - env.time, 0.0)
                delivery_eta = assigned.restaurant_location.distance_to(assigned.customer_location) / speed
                time_until_free = pickup_eta + max(prep_remaining - pickup_eta, 0.0) + delivery_eta
        max_route_time = max(2.0 * (cfg.width + cfg.height) / speed, 1e-6)
        obs[base + 9] = min(time_until_free / max_route_time, 2.0)

    restaurant_load: dict[int, int] = {}
    for order in orders:
        restaurant_load[order.restaurant_id] = restaurant_load.get(order.restaurant_id, 0) + 1
    for j, order in enumerate(orders):
        base = ORDER_OFFSET + j * ORDER_FEATURES
        obs[base] = 1.0
        obs[base + 1] = order.restaurant_location.x / cfg.width
        obs[base + 2] = order.restaurant_location.y / cfg.height
        obs[base + 3] = order.customer_location.x / cfg.width
        obs[base + 4] = order.customer_location.y / cfg.height
        obs[base + 5] = min(max(env.time - order.created_at, 0) / horizon, 1.0)
        obs[base + 7] = restaurant_load[order.restaurant_id] / MAX_ORDERS
        obs[base + 8] = order.restaurant_location.distance_to(order.customer_location) / (cfg.width + cfg.height)
        prep_remaining = max(order.estimated_ready_at - env.time, 0.0)
        best_eta = min(
            max(driver.location.distance_to(order.restaurant_location) / cfg.nominal_speed,
                prep_remaining)
            + order.restaurant_location.distance_to(order.customer_location) / cfg.nominal_speed
            for driver in available_drivers
        ) if available_drivers else horizon
        prep_remaining = max(order.estimated_ready_at - env.time, 0.0)
        obs[base + 6] = min(prep_remaining / max(cfg.preparation_max_minutes, 1e-6), 2.0)
        obs[base + 9] = max(min((order.deadline_at - env.time - best_eta) / horizon, 1.0), -1.0)

    for i, driver in enumerate(env.drivers):
        if driver.status != DriverStatus.AVAILABLE:
            continue
        for j, order in enumerate(orders):
            pickup_eta = driver.location.distance_to(order.restaurant_location) / cfg.nominal_speed
            prep_remaining = max(order.estimated_ready_at - env.time, 0.0)
            delivery_eta = order.restaurant_location.distance_to(order.customer_location) / cfg.nominal_speed
            eta = max(pickup_eta, prep_remaining) + delivery_eta
            action = i * MAX_ORDERS + j
            mask[action] = True
            eta_scale = max(2.0 * (cfg.width + cfg.height) / cfg.nominal_speed, 1.0)
            obs[PAIR_OFFSET + action] = max(min(eta / eta_scale, 2.0), 1e-4)

    globals_ = GLOBAL_OFFSET
    obs[globals_] = env.time / horizon
    obs[globals_ + 1] = max(horizon - env.time, 0) / horizon
    obs[globals_ + 2] = min(len(orders) / MAX_ORDERS, 1.0)
    obs[globals_ + 3] = available_count / MAX_DRIVERS
    obs[globals_ + 4] = min(len(env.pending_orders) / MAX_ORDERS, 1.0)
    obs[globals_ + 5] = min(cfg.orders_per_minute / 8.0, 2.0)
    obs[globals_ + 6] = min(cfg.restaurant_count / 10.0, 1.0)
    obs[globals_ + 7] = min(max(len(all_dispatchable) - len(orders), 0) / MAX_ORDERS, 2.0)
    obs[globals_ + 8] = cfg.nominal_speed / cfg.travel_speed
    return np.clip(obs, -2.0, 2.0), mask, orders

