"""Fixed-size observation and masked action encoding for learned dispatch policies."""

from __future__ import annotations

import numpy as np

from ..entities import DriverStatus, OrderStatus

MAX_DRIVERS = 16
MAX_ORDERS = 64
DRIVER_FEATURES = 10
ORDER_FEATURES = 10
GLOBAL_FEATURES = 8
WAIT_ACTION = MAX_DRIVERS * MAX_ORDERS
N_ACTIONS = WAIT_ACTION + 1
DRIVER_OFFSET = 0
ORDER_OFFSET = MAX_DRIVERS * DRIVER_FEATURES
PAIR_OFFSET = ORDER_OFFSET + MAX_ORDERS * ORDER_FEATURES
GLOBAL_OFFSET = PAIR_OFFSET + MAX_DRIVERS * MAX_ORDERS
OBS_SIZE = GLOBAL_OFFSET + GLOBAL_FEATURES


def ready_orders(env):
    return sorted(
        (order for order in env.orders if order.status == OrderStatus.READY),
        key=lambda order: (order.created_at, order.order_id),
    )


def encode_state(env):
    """Return the fixed observation and action mask for the current prepared minute."""
    if len(env.drivers) > MAX_DRIVERS:
        raise ValueError(f"DQN/PPO support at most {MAX_DRIVERS} delivery partners")
    cfg = env.config
    horizon = max(cfg.episode_minutes, 1)
    all_ready = ready_orders(env)
    orders = all_ready[:MAX_ORDERS]
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[WAIT_ACTION] = True

    available_count = 0
    for i, driver in enumerate(env.drivers):
        base = DRIVER_OFFSET + i * DRIVER_FEATURES
        obs[base] = 1.0
        obs[base + 1] = driver.location.x / cfg.width
        obs[base + 2] = driver.location.y / cfg.height
        available = driver.status == DriverStatus.AVAILABLE
        available_count += int(available)
        obs[base + 3] = float(available)
        obs[base + 4] = float(driver.status == DriverStatus.TO_RESTAURANT)
        obs[base + 5] = float(driver.status == DriverStatus.TO_CUSTOMER)
        if driver.destination is not None:
            obs[base + 6] = driver.destination.x / cfg.width
            obs[base + 7] = driver.destination.y / cfg.height
        obs[base + 8] = min(driver.remaining_distance / (cfg.width + cfg.height), 1.0)
        obs[base + 9] = float(driver.order_id is not None)

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
        obs[base + 6] = 1.0
        obs[base + 7] = restaurant_load[order.restaurant_id] / MAX_ORDERS
        obs[base + 8] = order.restaurant_location.distance_to(order.customer_location) / (cfg.width + cfg.height)
        obs[base + 9] = min(max(horizon - env.time, 0) / horizon, 1.0)

    for i, driver in enumerate(env.drivers):
        if driver.status != DriverStatus.AVAILABLE:
            continue
        for j, order in enumerate(orders):
            eta = (driver.location.distance_to(order.restaurant_location)
                   + order.restaurant_location.distance_to(order.customer_location)) / cfg.travel_speed
            action = i * MAX_ORDERS + j
            mask[action] = True
            eta_scale = max(2.0 * (cfg.width + cfg.height) / cfg.travel_speed, 1.0)
            obs[PAIR_OFFSET + action] = max(min(eta / eta_scale, 2.0), 1e-4)

    globals_ = GLOBAL_OFFSET
    obs[globals_] = env.time / horizon
    obs[globals_ + 1] = max(horizon - env.time, 0) / horizon
    obs[globals_ + 2] = min(len(orders) / MAX_ORDERS, 1.0)
    obs[globals_ + 3] = available_count / MAX_DRIVERS
    obs[globals_ + 4] = min(len(env.pending_orders) / MAX_ORDERS, 1.0)
    obs[globals_ + 5] = min(cfg.orders_per_minute / 5.0, 2.0)
    obs[globals_ + 6] = min(cfg.restaurant_count / 10.0, 1.0)
    obs[globals_ + 7] = min(max(len(all_ready) - len(orders), 0) / MAX_ORDERS, 2.0)
    return np.clip(obs, -2.0, 2.0), mask, orders

