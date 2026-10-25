"""Command-line comparison of the three dispatch baselines."""

import argparse

from .baselines import POLICIES
from .config import SimulationConfig
from .simulation.runner import average_metrics, compare_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Munchkin dispatch baselines.")
    parser.add_argument("--episodes", type=int, default=20, help="matched seeds per strategy (default: 20)")
    parser.add_argument("--seed", type=int, default=7, help="first random seed (default: 7)")
    parser.add_argument("--drivers", type=int, default=8)
    parser.add_argument("--restaurants", type=int, default=5)
    parser.add_argument("--orders-per-minute", type=float, default=5)
    parser.add_argument("--traffic", choices=("light", "normal", "heavy"), default="normal")
    parser.add_argument("--minutes", type=int, default=60)
    args = parser.parse_args()

    config = SimulationConfig(
        driver_count=args.drivers,
        restaurant_count=args.restaurants,
        orders_per_minute=args.orders_per_minute,
        traffic_level=args.traffic,
        episode_minutes=args.minutes,
        seed=args.seed,
    )
    results = compare_policies(config, episodes=args.episodes)
    print(f"Matched comparison: {args.episodes} episodes per policy, {args.minutes} simulated minutes each, "
          f"{args.traffic} traffic")
    print(f"{'Strategy':38} {'Delivered':>10} {'Game pts':>10} {'RL reward':>12} "
          f"{'Avg delivery (min)':>20} {'Travel distance':>18}")
    for name in POLICIES:
        delivered, reward, avg_time, distance = average_metrics(results[name])
        rl_reward = sum(r.total_training_reward for r in results[name]) / len(results[name])
        print(f"{name:38} {delivered:10.2f} {reward:10.1f} {rl_reward:12.1f} "
              f"{avg_time:20.2f} {distance:18.2f}")


if __name__ == "__main__":
    main()
