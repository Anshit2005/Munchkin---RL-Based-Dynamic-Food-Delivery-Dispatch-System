"""Every episode must last exactly 3 simulated minutes, regardless of policy or configuration."""
import dataclasses

import pytest

from environment.config import EnvConfig
from environment.constants import EPISODE_DURATION_S, N_TICKS
from environment.scenario import generate_scenario
from simulation.runner import run_episode
from simulation.simulator import Engine


def test_constant_is_three_minutes():
    assert EPISODE_DURATION_S == 180 and N_TICKS == 180


def test_every_policy_and_scenario_ends_at_180(all_policies, scenarios):
    for cfg in scenarios.values():
        for pol in all_policies:
            for seed in (1, 2, 3):
                res = run_episode(pol, cfg, seed)
                assert res.final_tick == 180, (pol.name, cfg.name, seed)


def test_engine_refuses_to_run_past_horizon(scenarios):
    eng = Engine(generate_scenario(scenarios["medium_demand"], 0))
    for _ in range(N_TICKS):
        eng.advance_tick()
    assert eng.finished and eng.t == 180
    with pytest.raises(RuntimeError):
        eng.advance_tick()


def test_duration_is_not_configurable():
    assert "duration" not in {f.name for f in dataclasses.fields(EnvConfig)}
    for key in ("duration", "episode_duration", "horizon", "n_ticks"):
        with pytest.raises(ValueError):
            EnvConfig.from_dict({key: 300})
    with pytest.raises(TypeError):
        EnvConfig(duration=300)


def test_duration_independent_of_config(all_policies):
    for n_drivers in (1, 12):
        for rate in (0.0, 10.0):
            cfg = EnvConfig(n_drivers=n_drivers, orders_per_min=rate, backlog_frac=0.5)
            assert run_episode(all_policies[2], cfg, 5).final_tick == 180
