"""Identical seeds -> identical worlds; every policy sees the same scenario (fair comparison)."""
from environment.scenario import generate_scenario
from simulation.runner import run_all, run_episode
from utils.seeding import TEST_SEEDS, TRAIN_SEED_MAX, VAL_SEEDS


def test_scenario_is_deterministic(scenarios):
    a = generate_scenario(scenarios["high_demand"], 42)
    b = generate_scenario(scenarios["high_demand"], 42)
    c = generate_scenario(scenarios["high_demand"], 43)
    assert a.fingerprint == b.fingerprint != c.fingerprint


def test_same_fingerprint_across_all_policies(all_policies, scenarios):
    for cfg in scenarios.values():
        res = run_all(all_policies, cfg, seed=11)
        assert len({r.fingerprint for r in res}) == 1


def test_episode_repeatable(scenarios):
    from baselines import MinETAPolicy
    r1 = run_episode(MinETAPolicy(), scenarios["medium_demand"], 7)
    r2 = run_episode(MinETAPolicy(), scenarios["medium_demand"], 7)
    assert r1.metrics == r2.metrics and r1.reward_total == r2.reward_total


def test_seed_pools_disjoint():
    assert max(range(TRAIN_SEED_MAX)) < min(VAL_SEEDS)
    assert max(VAL_SEEDS) < min(TEST_SEEDS)
    assert not set(VAL_SEEDS) & set(TEST_SEEDS)
