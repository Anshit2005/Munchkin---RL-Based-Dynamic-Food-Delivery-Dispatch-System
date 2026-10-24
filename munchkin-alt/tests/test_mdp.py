"""Observation, action mask, reward, metrics and the Gymnasium API."""
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from baselines import MinETAPolicy, MostUrgentOrderPolicy, NearestDriverPolicy
from environment.config import RewardConfig
from environment.constants import (MAX_DRIVERS, MAX_ORDERS, N_ACTIONS, OBS_DIM, WAIT_ACTION,
                                   encode_action)
from environment.delivery_env import DeliveryEnv
from environment.reward import tick_reward
from environment.scenario import generate_scenario
from evaluation.metrics import compute_metrics
from simulation.policy import RandomValidPolicy
from simulation.runner import run_episode
from simulation.simulator import Engine, TickEvents
from tests.helpers import make_scenario


def test_dimensions():
    assert OBS_DIM == 481 and N_ACTIONS == MAX_DRIVERS * MAX_ORDERS + 1 == 193


def test_check_env(scenarios):
    env = DeliveryEnv(scenarios["medium_demand"])
    check_env(env, skip_render_check=True)


def test_observation_valid_and_wait_always_legal(scenarios):
    env = DeliveryEnv(scenarios["high_demand"])
    obs, info = env.reset(seed=5)
    pol = RandomValidPolicy(1)
    done = False
    while not done:
        assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
        assert np.isfinite(obs).all() and obs.min() >= -2 and obs.max() <= 2
        assert info["action_mask"][WAIT_ACTION]
        obs, _, done, _, info = env.step(pol.act(obs, info, None))


def test_random_valid_fuzz_has_no_invalid_actions(scenarios):
    for name, cfg in scenarios.items():
        for seed in range(15):
            r = run_episode(RandomValidPolicy(seed), cfg, seed)
            assert r.n_invalid == 0 and r.final_tick == 180


def test_batching_fuzz(scenarios):
    cfg = scenarios["high_demand"].replace(batching=True)
    for seed in range(10):
        r = run_episode(RandomValidPolicy(seed), cfg, seed)
        assert r.n_invalid == 0 and r.final_tick == 180


def test_invalid_action_is_penalised_and_treated_as_wait(scenarios):
    env = DeliveryEnv(scenarios["medium_demand"])
    env.reset(seed=1)
    mask = env.action_masks()
    bad = int(np.flatnonzero(~mask)[0])
    t0 = env.engine.t
    _, r, _, _, info = env.step(bad)
    assert info["invalid_action"] and env.engine.t > t0
    assert info["reward_breakdown"]["invalid"] == -env.rc.invalid_penalty


def test_hidden_prep_time_does_not_leak_into_observation():
    base = dict(restaurants=[(2.0, 2.0)], drivers=[(0.0, 0.0)])
    o = {"created_at": 0, "restaurant_id": 0, "cx": 3.0, "cy": 3.0, "prep_est": 30.0}
    a = make_scenario(orders=[{**o, "prep": 20.0}], **base)
    b = make_scenario(orders=[{**o, "prep": 55.0}], **base)
    ea, eb = DeliveryEnv(), DeliveryEnv()
    from environment.observation import build_observation
    from environment.actions import compute_mask
    for env, sc in ((ea, a), (eb, b)):
        env.engine = Engine(sc)
    sa, sb = ea.engine.pending[:MAX_ORDERS], eb.engine.pending[:MAX_ORDERS]
    oa = build_observation(ea.engine, sa, compute_mask(ea.engine, sa), 0)
    ob = build_observation(eb.engine, sb, compute_mask(eb.engine, sb), 0)
    assert np.array_equal(oa, ob)


def test_reward_components_hand_computed():
    rc = RewardConfig()
    ev = TickEvents()
    ev.n_delivered, ev.n_waiting, ev.n_overdue = 2, 6, 3
    ev.distance, ev.n_idle, ev.batch_saving = 0.5, 4, 1.2
    b = tick_reward(ev, rc)
    assert b.delivery == pytest.approx(2 * rc.w_delivery)
    assert b.wait == pytest.approx(-rc.w_wait * 6 / 60)
    assert b.late == pytest.approx(-rc.w_late * 3 / 60)
    assert b.travel == pytest.approx(-rc.w_travel * 0.5)
    assert b.batch == pytest.approx(rc.w_batch * 1.2)
    assert b.idle == pytest.approx(-rc.w_idle * 4 / 60)
    assert b.total == pytest.approx(sum(b.to_dict()[k] for k in
                                        ("delivery", "wait", "late", "travel", "batch", "idle")))


def test_terminal_penalty_for_undelivered(scenarios):
    from baselines import MinETAPolicy
    r = run_episode(RandomValidPolicy(0), scenarios["high_demand"], 2)
    assert r.reward_breakdown["terminal"] == -RewardConfig().undelivered_penalty * r.metrics["undelivered"]


def test_episode_reward_equals_sum_of_components(scenarios):
    r = run_episode(MinETAPolicy(), scenarios["medium_demand"], 3)
    parts = sum(v for k, v in r.reward_breakdown.items() if k != "total")
    assert r.reward_total == pytest.approx(parts)


def test_metrics_toy_scenario():
    sc = make_scenario([(1.0, 0.0), (5.0, 5.0)], [(0.0, 0.0), (9.0, 9.0)],
                       [{"created_at": 0, "restaurant_id": 0, "cx": 1.0, "cy": 1.0, "prep": 10.0, "deadline": 1000},
                        {"created_at": 0, "restaurant_id": 1, "cx": 5.0, "cy": 6.0, "prep": 10.0, "deadline": 5},
                        {"created_at": 100, "restaurant_id": 0, "cx": 2.0, "cy": 2.0, "prep": 10.0, "deadline": 1000}])
    eng = Engine(sc)
    eng.assign(eng.drivers[0], eng.orders[0])
    eng.assign(eng.drivers[1], eng.orders[1])
    while not eng.finished:
        eng.advance_tick()
    m = compute_metrics(eng)
    o0, o1, o2 = eng.orders
    assert m["n_orders"] == 3 and m["n_delivered"] == 2 and m["undelivered"] == 1
    assert m["avg_delivery_time"] == pytest.approx((o0.delivered_at + o1.delivered_at) / 2)
    assert m["on_time_pct"] == pytest.approx(50.0) and m["late_pct"] == pytest.approx(50.0)
    assert m["completion_rate"] == pytest.approx(200 / 3)
    # waiting is censored at the horizon for the undelivered order created at t=100
    assert m["avg_wait_time"] == pytest.approx((o0.delivered_at + o1.delivered_at + (180 - 100)) / 3)
    assert m["total_distance"] == pytest.approx(sum(d.distance for d in eng.drivers))
    busy = sum(d.busy_ticks for d in eng.drivers)
    assert m["utilization"] == pytest.approx(100 * busy / (2 * 180))


def test_batching_saves_distance_and_counts():
    # two orders at the same restaurant with nearby customers, one driver able to carry both
    cfg = None
    from environment.config import load_env_defaults
    cfg = load_env_defaults().replace(batching=True)
    sc = make_scenario([(1.0, 0.0)], [(0.0, 0.0)],
                       [{"created_at": 0, "restaurant_id": 0, "cx": 3.0, "cy": 0.0, "prep": 15.0},
                        {"created_at": 0, "restaurant_id": 0, "cx": 3.0, "cy": 1.0, "prep": 15.0}], cfg=cfg)
    eng = Engine(sc)
    d = eng.drivers[0]
    eng.assign(d, eng.orders[0])
    assert eng.can_assign(d, eng.orders[1])
    eng.assign(d, eng.orders[1])
    while not eng.finished:
        eng.advance_tick()
    assert all(o.delivered_at is not None for o in eng.orders)
    assert compute_metrics(eng)["batch_count"] == 1
    assert eng.batches[0]["saved"] > 0
