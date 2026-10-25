"""Matched-seed comparison of available heuristics, DQN, and PPO checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

from ..baselines import POLICIES, create_policy
from ..config import SimulationConfig, PROJECT_ROOT
from ..simulation.runner import run_policy_episode
from .policies import create_rl_policy
from .registry import available_policies
from .state import MAX_DRIVERS


def evaluate(episodes: int = 20, minutes: int = 60, seed: int = 1000,
             drivers: int = 8, restaurants: int = 5, orders_per_minute: float = 1.5):
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    policy_names = [*POLICIES, *available_policies()]
    if available_policies() and drivers > MAX_DRIVERS:
        raise ValueError(f"DQN/PPO support at most {MAX_DRIVERS} delivery partners")
    policies = {
        name: (create_policy(name) if name in POLICIES else create_rl_policy(name))
        for name in policy_names
    }
    results = defaultdict(list)
    for episode in range(episodes):
        episode_seed = seed + episode
        config = SimulationConfig(
            restaurant_count=restaurants,
            driver_count=drivers,
            orders_per_minute=orders_per_minute,
            episode_minutes=minutes,
            seed=episode_seed,
        )
        for name, policy in policies.items():
            results[name].append(run_policy_episode(policy, config, episode_seed))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Munchkin strategies on matched random seeds")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--drivers", type=int, default=8)
    parser.add_argument("--restaurants", type=int, default=5)
    parser.add_argument("--orders-per-minute", type=float, default=1.5)
    args = parser.parse_args()
    results = evaluate(args.episodes, args.minutes, args.seed, args.drivers,
                       args.restaurants, args.orders_per_minute)
    fields = ("strategy", "episodes", "mean_delivered", "mean_reward", "mean_delivery_minutes", "mean_travel_distance")
    rows = []
    for name, episodes in results.items():
        delivery_times = [item.average_delivery_time for item in episodes if item.delivered]
        rows.append({
            "strategy": name,
            "episodes": len(episodes),
            "mean_delivered": sum(item.delivered for item in episodes) / len(episodes),
            "mean_reward": sum(item.total_reward for item in episodes) / len(episodes),
            "mean_delivery_minutes": sum(delivery_times) / len(delivery_times) if delivery_times else math.nan,
            "mean_travel_distance": sum(item.total_travel_distance for item in episodes) / len(episodes),
        })
    out = PROJECT_ROOT / "results" / "rl_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'Strategy':38} {'Delivered':>10} {'Reward':>10} {'Avg delivery (min)':>20} {'Travel':>12}")
    for row in rows:
        avg = row["mean_delivery_minutes"]
        avg_text = f"{avg:.2f}" if math.isfinite(avg) else "n/a"
        print(f"{row['strategy']:38} {row['mean_delivered']:10.2f} {row['mean_reward']:10.1f} "
              f"{avg_text:>20} {row['mean_travel_distance']:12.1f}")
    print(f"Saved matched-seed results to {out}")


if __name__ == "__main__":
    main()
