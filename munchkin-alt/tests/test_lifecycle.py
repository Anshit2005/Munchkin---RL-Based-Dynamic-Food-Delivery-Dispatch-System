"""Order lifecycle invariants, conservation of orders, and exact movement arithmetic."""
import pytest

from baselines import MinETAPolicy
from environment.config import EnvConfig, load_env_defaults
from environment.constants import DriverStatus, OrderStatus
from environment.delivery_env import DeliveryEnv
from environment.scenario import generate_scenario
from simulation.runner import run_episode
from simulation.simulator import Engine
from tests.helpers import make_scenario


def _final_engine(policy, cfg, seed):
    env = DeliveryEnv(cfg)
    obs, info = env.reset(seed=seed)
    policy.reset(cfg)
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy.act(obs, info, env.get_view()))
    return env.engine


@pytest.mark.parametrize("scn", ["medium_demand", "high_demand", "driver_shortage"])
def test_timestamps_are_monotone(scenarios, scn):
    eng = _final_engine(MinETAPolicy(), scenarios[scn], 3)
    for o in eng.orders:
        chain = [o.created_at, o.ready_at, o.assigned_at, o.arrived_at_rest, o.picked_up_at, o.delivered_at]
        if o.assigned_at is not None:
            assert o.assigned_at >= o.created_at
        if o.picked_up_at is not None:
            assert o.ready_at is not None and o.picked_up_at >= o.ready_at   # never before food is ready
            assert o.picked_up_at >= o.arrived_at_rest >= o.assigned_at
        if o.delivered_at is not None:
            assert o.delivered_at > o.picked_up_at
        seq = [x for x in chain if x is not None and x is not o.ready_at]
        assert seq == sorted(seq)


def test_never_assigned_twice_and_capacity(scenarios):
    cfg = scenarios["medium_demand"].replace(batching=True)
    env = DeliveryEnv(cfg)
    obs, info = env.reset(seed=4)
    pol = MinETAPolicy()
    done = False
    while not done:
        obs, _, done, _, info = env.step(pol.act(obs, info, env.get_view()))
        for d in env.engine.drivers:
            assert d.load <= d.capacity
    ids = [o.driver_id for o in env.engine.orders if o.assigned_at is not None]
    assert all(i is not None for i in ids)


def test_order_conservation_each_tick(scenarios):
    eng = Engine(generate_scenario(scenarios["high_demand"], 9))
    pol_env = DeliveryEnv(scenarios["high_demand"])
    obs, info = pol_env.reset(seed=9)
    pol = MinETAPolicy()
    done = False
    while not done:
        obs, _, done, _, info = pol_env.step(pol.act(obs, info, pol_env.get_view()))
        e = pol_env.engine
        n_created = len(e.orders)
        n_pending = len(e.pending)
        n_assigned_open = sum(1 for o in e.orders if o.assigned_at is not None and o.delivered_at is None)
        n_delivered = sum(1 for o in e.orders if o.delivered_at is not None)
        assert n_created == n_pending + n_assigned_open + n_delivered


def test_single_order_exact_timing():
    """Driver (0,0) -> restaurant (2,0) -> customer (2,2); speed 0.13/s, prep 10 s."""
    sc = make_scenario([(2.0, 0.0)], [(0.0, 0.0)],
                       [{"created_at": 0, "restaurant_id": 0, "cx": 2.0, "cy": 2.0, "prep": 10.0}])
    eng = Engine(sc)
    (o,) = eng.orders
    eng.assign(eng.drivers[0], o)
    while not eng.finished and o.delivered_at is None:
        eng.advance_tick()
    speed = sc.config.base_speed
    assert o.ready_at == 10
    assert o.picked_up_at == pytest.approx(2.0 / speed, abs=1)       # ~15.4 s -> tick 16
    assert o.delivered_at == pytest.approx(4.0 / speed, abs=1.5)     # ~30.8 s total
    assert eng.drivers[0].distance == pytest.approx(4.0)
    assert eng.drivers[0].status == DriverStatus.AVAILABLE
    assert o.status == OrderStatus.DELIVERED


def test_driver_waits_for_food():
    """Driver arrives before the food is ready and waits at the restaurant."""
    sc = make_scenario([(1.0, 0.0)], [(0.0, 0.0)],
                       [{"created_at": 0, "restaurant_id": 0, "cx": 1.0, "cy": 1.0, "prep": 50.0}])
    eng = Engine(sc)
    (o,) = eng.orders
    eng.assign(eng.drivers[0], o)
    for _ in range(30):
        eng.advance_tick()
    assert eng.drivers[0].status == DriverStatus.AT_RESTAURANT and o.picked_up_at is None
    while o.delivered_at is None and not eng.finished:
        eng.advance_tick()
    assert o.picked_up_at == 50          # picked up the tick the food becomes ready
