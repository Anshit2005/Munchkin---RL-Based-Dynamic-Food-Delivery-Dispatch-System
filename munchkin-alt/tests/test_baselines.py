"""Known-answer dispatch decisions on hand-built worlds."""
from baselines import MinETAPolicy, MostUrgentOrderPolicy, NearestDriverPolicy
from environment.constants import WAIT_ACTION, decode_action
from environment.delivery_env import DeliveryEnv
from simulation.simulator import Engine
from tests.helpers import make_scenario


def _setup(drivers, orders, restaurants=((5.0, 5.0),)):
    env = DeliveryEnv()
    env.engine = Engine(make_scenario(list(restaurants), drivers, orders))
    env.assign_count = 0
    env._refresh()
    return env


def _choice(policy, env):
    a = policy.act(env._obs, {"action_mask": env._mask}, env.get_view())
    if a == WAIT_ACTION:
        return None
    di, oj = decode_action(a)
    return di, env._slots[oj].id


def test_nearest_driver_picks_closest_driver_for_oldest_order():
    env = _setup([(0.0, 0.0), (4.0, 5.0), (9.0, 9.0)],
                 [{"created_at": 0, "restaurant_id": 0, "cx": 6.0, "cy": 6.0, "prep": 30.0}])
    assert _choice(NearestDriverPolicy(), env) == (1, 0)


def test_all_baselines_wait_when_no_driver_available():
    env = _setup([(0.0, 0.0)], [{"created_at": 0, "restaurant_id": 0, "cx": 6.0, "cy": 6.0, "prep": 30.0}])
    env.engine.assign(env.engine.drivers[0], env.engine.orders[0])
    env.engine.orders.append(env.engine.orders[0])   # keep structure valid; pending is now empty
    env._refresh()
    for pol in (NearestDriverPolicy(), MostUrgentOrderPolicy(), MinETAPolicy()):
        assert _choice(pol, env) is None


def test_most_urgent_serves_least_slack_order_first():
    # order 1 has a much tighter deadline than order 0 (older)
    env = _setup([(5.0, 4.0)],
                 [{"created_at": 0, "restaurant_id": 0, "cx": 6.0, "cy": 6.0, "prep": 30.0, "deadline": 500},
                  {"created_at": 1, "restaurant_id": 0, "cx": 6.0, "cy": 7.0, "prep": 30.0, "deadline": 60}])
    env.engine.advance_tick()
    env._refresh()
    assert len(env._slots) == 2
    assert _choice(MostUrgentOrderPolicy(), env)[1] == 1
    assert _choice(NearestDriverPolicy(), env)[1] == 0          # oldest first


def test_min_eta_picks_global_minimum_pair():
    # driver 0 is far; driver 1 is near; the order with the shorter trip should be picked with driver 1
    env = _setup([(0.0, 0.0), (5.0, 4.5)],
                 [{"created_at": 0, "restaurant_id": 0, "cx": 9.0, "cy": 9.0, "prep": 20.0},
                  {"created_at": 0, "restaurant_id": 0, "cx": 5.5, "cy": 5.0, "prep": 20.0}])
    assert _choice(MinETAPolicy(), env) == (1, 1)


def test_baselines_never_emit_invalid_actions(scenarios):
    from simulation.runner import run_episode
    for pol in (NearestDriverPolicy(), MostUrgentOrderPolicy(), MinETAPolicy()):
        for cfg in scenarios.values():
            assert run_episode(pol, cfg, 21).n_invalid == 0
