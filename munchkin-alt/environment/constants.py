"""Fixed experimental constants.

The episode duration is a module constant on purpose: it is NOT part of EnvConfig and is
never exposed to the GUI, so it cannot depend on demand, drivers, restaurants or algorithm.
"""
from enum import IntEnum

EPISODE_DURATION_S = 180          # exactly 3 simulated minutes
TICK_S = 1                        # 1 tick = 1 simulated second
N_TICKS = EPISODE_DURATION_S // TICK_S

MAX_DRIVERS = 12
MAX_ORDERS = 16                   # visible (unassigned) order slots
MAX_RESTAURANTS = 8
GRID_SIZE = 10.0

WAIT_ACTION = MAX_DRIVERS * MAX_ORDERS
N_ACTIONS = WAIT_ACTION + 1
ASSIGN_CAP = MAX_DRIVERS          # max assignments in one tick before time is forced forward

DRIVER_FEATS = 10
ORDER_FEATS = 10
PAIR_FEATS = 1
GLOBAL_FEATS = 9
OBS_DIM = (MAX_DRIVERS * DRIVER_FEATS + MAX_ORDERS * ORDER_FEATS
           + MAX_DRIVERS * MAX_ORDERS * PAIR_FEATS + GLOBAL_FEATS)   # 481
OBS_VERSION = 1

TRAFFIC_LEVELS = {"light": 1.2, "normal": 1.0, "heavy": 0.7}


class DriverStatus(IntEnum):
    AVAILABLE = 0        # no assigned orders; stationary; assignable
    TO_RESTAURANT = 1
    AT_RESTAURANT = 2    # arrived, food not ready yet (waiting for pickup)
    TO_CUSTOMER = 3
    # IDLE is a derived flag (AVAILABLE while unassigned orders exist), not a stored status.


class OrderStatus(IntEnum):
    PREPARING = 0   # created, food being prepared (not yet assigned)
    READY = 1       # food ready, waiting for a driver
    ASSIGNED = 2    # a driver has been assigned (before pickup)
    PICKED_UP = 3
    DELIVERED = 4


def encode_action(driver_idx: int, order_slot: int) -> int:
    return driver_idx * MAX_ORDERS + order_slot


def decode_action(action: int):
    """Return (driver_idx, order_slot), or None for WAIT."""
    if action == WAIT_ACTION:
        return None
    return divmod(int(action), MAX_ORDERS)
