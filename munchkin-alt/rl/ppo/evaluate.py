"""Evaluate a PPO checkpoint on val/test seeds across the named scenarios.

    python -m rl.ppo.evaluate --checkpoint models/ppo/ppo_s0/best.pt --split val
"""
from __future__ import annotations

import argparse

import pandas as pd

from environment.config import ROOT, load_reward_config, load_scenarios
from rl.ppo.agent import PPOPolicy
from simulation.runner import run_episode
from utils.seeding import split_seeds


def evaluate(checkpoint: str, split: str = "val", n_seeds: int | None = None,
             deterministic: bool = True) -> pd.DataFrame:
    policy = PPOPolicy(checkpoint=checkpoint, deterministic=deterministic)
    rc = load_reward_config()
    rows = [run_episode(policy, cfg, s, rc).row() for cfg in load_scenarios().values()
            for s in split_seeds(split, n_seeds)]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()
    df = evaluate(args.checkpoint, args.split, args.n_seeds, not args.stochastic)
    out = ROOT / "results" / "raw" / f"ppo_eval_{args.split}.csv"
    df.to_csv(out, index=False)
    print(df.groupby("scenario")[["reward", "avg_delivery_time", "on_time_pct", "n_delivered"]].mean())
    print("saved", out)


if __name__ == "__main__":
    main()
