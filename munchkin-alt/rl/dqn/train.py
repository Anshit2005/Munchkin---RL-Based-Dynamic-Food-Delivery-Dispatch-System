"""DQN training.

    python -m rl.dqn.train --config configs/dqn.yaml --seed 0 --run-name dqn_s0
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
from rl.common.env_factory import make_train_env, validation_score
from rl.common.logger import CSVLogger
from rl.dqn.agent import DQNAgent, DQNPolicy
from rl.dqn.replay_buffer import ReplayBuffer
from utils.seeding import set_global_seeds


def train(cfg: dict, seed: int, run_name: str, out_root: Path = ROOT / "models" / "dqn",
          batching: bool = False, verbose: bool = True) -> Path:
    torch.set_num_threads(cfg.get("threads", 2))
    set_global_seeds(seed)
    rc = load_reward_config()
    run_dir = out_root / run_name
    log = CSVLogger(run_dir / "train_log.csv")
    val_log = CSVLogger(run_dir / "val_log.csv")

    env = make_train_env(batching, rc)
    agent = DQNAgent(cfg, seed)
    buf = ReplayBuffer(cfg["buffer_size"])
    rng = np.random.default_rng(seed)
    total = cfg["total_steps"]
    decay_steps = max(1, int(cfg["eps_decay_frac"] * total))
    scale = cfg.get("reward_scale", 1.0)
    meta = {"algo": "dqn", "agent_cfg": cfg, "reward_hash": rc.hash(), "batching": batching,
            "seed": seed, "run_name": run_name}

    obs, info = env.reset(seed=seed * 7919 % 1_000_000)
    ep, best, t0, loss = 0, -np.inf, time.time(), float("nan")
    for step in range(1, total + 1):
        eps = max(cfg["eps_end"], cfg["eps_start"] - (cfg["eps_start"] - cfg["eps_end"]) * step / decay_steps)
        a = agent.act(obs, info["action_mask"], eps)
        nobs, r, term, _, ninfo = env.step(a)
        buf.add(obs, a, r * scale, nobs, term, ninfo["action_mask"])
        obs, info = nobs, ninfo

        if term:
            ep += 1
            m = info["episode_metrics"]
            log.log({"episode": ep, "step": step, "return": info["episode_return"], "epsilon": eps,
                     "loss": loss, "scenario": info["scenario_name"], "n_delivered": m["n_delivered"],
                     "on_time_pct": m["on_time_pct"], "avg_delivery_time": m["avg_delivery_time"],
                     **{"rw_" + k: v for k, v in info["episode_breakdown"].items()}})
            obs, info = env.reset()

        if step >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            loss = agent.learn(buf.sample(cfg["batch_size"], rng))
        if step % cfg["target_update"] == 0 and cfg.get("tau", 0.0) == 0:
            agent.hard_update()

        if step % cfg["val_every"] == 0 or step == total:
            agent.q.eval()
            v = validation_score(DQNPolicy(agent), cfg["val_seeds_per_scenario"], reward_config=rc)
            agent.q.train()
            val_log.log({"step": step, "val_reward": v["mean_reward"], **v["per_scenario"]})
            if verbose:
                print(f"[dqn s{seed}] step {step}/{total} ep {ep} eps {eps:.2f} loss {loss:.3f} "
                      f"val {v['mean_reward']:.1f} ({time.time() - t0:.0f}s)", flush=True)
            if v["mean_reward"] > best:
                best = v["mean_reward"]
                save_checkpoint(run_dir / "best.pt", agent.state(), {**meta, "step": step, "val_reward": best})
    save_checkpoint(run_dir / "last.pt", agent.state(), {**meta, "step": total, "val_reward": best})
    log.close()
    val_log.close()
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dqn.yaml")
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
    train(cfg, args.seed, args.run_name or f"dqn_s{args.seed}", batching=args.batching)


if __name__ == "__main__":
    main()
