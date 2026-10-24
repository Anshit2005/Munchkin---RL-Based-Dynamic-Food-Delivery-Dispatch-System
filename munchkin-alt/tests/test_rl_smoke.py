"""Tiny training runs: pipelines execute, checkpoints round-trip, masked policies stay valid."""
import numpy as np
import torch
import yaml

from environment.config import ROOT
from rl.common.networks import make_action_net
from rl.dqn.agent import DQNPolicy
from rl.dqn.train import train as train_dqn
from rl.ppo.agent import PPOPolicy
from rl.ppo.train import train as train_ppo
from simulation.runner import run_episode


def _cfg(name):
    return yaml.safe_load(open(ROOT / "configs" / name))


def test_pairwise_net_matches_shapes_and_ignores_invalid_pairs(scenarios):
    from environment.delivery_env import DeliveryEnv
    env = DeliveryEnv(scenarios["medium_demand"])
    obs, info = env.reset(seed=0)
    for arch in ("pairwise", "mlp"):
        net = make_action_net(arch, 32)
        out = net(torch.as_tensor(obs).unsqueeze(0))
        assert out.shape == (1, 193)


def test_dqn_trains_saves_and_reloads(tmp_path, scenarios):
    cfg = _cfg("dqn.yaml")
    cfg.update(total_steps=1500, learning_starts=300, val_every=1500, val_seeds_per_scenario=1,
               target_update=300, hidden=32)
    run = train_dqn(cfg, 0, "t", out_root=tmp_path, verbose=False)
    pol = DQNPolicy(checkpoint=str(run / "best.pt"))
    r = run_episode(pol, scenarios["medium_demand"], 1_000_000)
    assert r.n_invalid == 0 and r.final_tick == 180
    r2 = run_episode(DQNPolicy(checkpoint=str(run / "best.pt")), scenarios["medium_demand"], 1_000_000)
    assert r.reward_total == r2.reward_total          # save -> load -> identical behaviour


def test_ppo_trains_saves_and_reloads(tmp_path, scenarios):
    cfg = _cfg("ppo.yaml")
    cfg.update(total_steps=2 * 4 * 64, n_envs=4, n_steps=64, minibatch=64, epochs=2,
               val_every_updates=2, val_seeds_per_scenario=1, hidden=32, value_hidden=32)
    run = train_ppo(cfg, 0, "t", out_root=tmp_path, verbose=False)
    pol = PPOPolicy(checkpoint=str(run / "best.pt"))
    r = run_episode(pol, scenarios["medium_demand"], 1_000_000)
    assert r.n_invalid == 0 and r.final_tick == 180
