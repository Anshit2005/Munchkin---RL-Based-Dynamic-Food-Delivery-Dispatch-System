"""PPO training.

    python -m rl.ppo.train --config configs/ppo.yaml --seed 0 --run-name ppo_s0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from environment.config import ROOT, load_reward_config
from rl.common.checkpoint import save_checkpoint
from rl.common.env_factory import VecEnv, validation_score
from rl.common.logger import CSVLogger
from rl.ppo.agent import PPOAgent, PPOPolicy
from rl.ppo.rollout_buffer import RolloutBuffer
from utils.seeding import set_global_seeds


def train(cfg: dict, seed: int, run_name: str, out_root: Path = ROOT / "models" / "ppo",
          batching: bool = False, verbose: bool = True) -> Path:
    torch.set_num_threads(cfg.get("threads", 2))
    set_global_seeds(seed)
    rc = load_reward_config()
    run_dir = out_root / run_name
    log = CSVLogger(run_dir / "train_log.csv")
    val_log = CSVLogger(run_dir / "val_log.csv")

    vec = VecEnv(cfg["n_envs"], base_seed=seed, batching=batching, reward_config=rc)
    agent = PPOAgent(cfg, seed)
    buf = RolloutBuffer(cfg["n_steps"], cfg["n_envs"])
    rng = np.random.default_rng(seed)
    per_update = cfg["n_steps"] * cfg["n_envs"]
    n_updates = max(1, cfg["total_steps"] // per_update)
    scale = cfg.get("reward_scale", 1.0)
    meta = {"algo": "ppo", "agent_cfg": cfg, "reward_hash": rc.hash(), "batching": batching,
            "seed": seed, "run_name": run_name}

    best, t0, ep_count = -np.inf, time.time(), 0
    for u in range(1, n_updates + 1):
        frac = 1.0 - (u - 1) / n_updates
        for g in agent.opt.param_groups:
            g["lr"] = cfg["lr"] * frac
        ent_coef = cfg["ent_coef_end"] + (cfg["ent_coef_start"] - cfg["ent_coef_end"]) * frac

        finished = []
        for t in range(cfg["n_steps"]):
            obs, masks = vec.obs.copy(), vec.masks.copy()
            actions, logp, values = agent.act(obs, masks)
            _, _, rewards, dones, fin = vec.step(actions)
            buf.add(t, obs, masks, actions, logp, values, rewards * scale, dones.astype(np.float32))
            finished += fin
        buf.compute_gae(agent.values(vec.obs), cfg["gamma"], cfg["gae_lambda"])
        stats = agent.update(buf, ent_coef, rng)

        ep_count += len(finished)
        step = u * per_update
        log.log({"update": u, "step": step, "episodes": ep_count,
                 "mean_return": float(np.mean([f["return"] for f in finished])) if finished else float("nan"),
                 "mean_on_time_pct": float(np.nanmean([f["metrics"]["on_time_pct"] for f in finished])) if finished else float("nan"),
                 "ent_coef": ent_coef, **stats})

        if u % cfg["val_every_updates"] == 0 or u == n_updates:
            agent.policy.eval()
            v = validation_score(PPOPolicy(agent), cfg["val_seeds_per_scenario"], reward_config=rc)
            agent.policy.train()
            val_log.log({"step": step, "val_reward": v["mean_reward"], **v["per_scenario"]})
            if verbose:
                print(f"[ppo s{seed}] upd {u}/{n_updates} step {step} train_ret "
                      f"{np.mean([f['return'] for f in finished]) if finished else float('nan'):.1f} "
                      f"ent {stats['entropy']:.2f} kl {stats['kl']:.3f} val {v['mean_reward']:.1f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if v["mean_reward"] > best:
                best = v["mean_reward"]
                save_checkpoint(run_dir / "best.pt", agent.state(), {**meta, "step": step, "val_reward": best})
    save_checkpoint(run_dir / "last.pt", agent.state(), {**meta, "step": n_updates * per_update, "val_reward": best})
    log.close()
    val_log.close()
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--batching", action="store_true")
    args = ap.parse_args()
    with open(ROOT / args.config if not Path(args.config).is_absolute() else args.config) as f:
        cfg = yaml.safe_load(f)
    if args.steps:
        cfg["total_steps"] = args.steps
    if args.arch:
        cfg["arch"] = args.arch
    train(cfg, args.seed, args.run_name or f"ppo_s{args.seed}", batching=args.batching)


if __name__ == "__main__":
    main()
