"""Print a matched-seed dispatch strategy report for low/medium/heavy demand.

Run from the project root with:
    python3 -m food_delivery_game.rl.scenario_report
"""

from __future__ import annotations

import argparse
import math

from ..baselines import POLICIES, create_policy
from ..config import SimulationConfig
from ..simulation.runner import run_policy_episode
from .policies import create_rl_policy
from .registry import available_policies
from .state import MAX_DRIVERS

SCENARIOS = (
    ("Low demand", 0.5),
    ("Medium demand", 1.5),
    ("Heavy demand", 4.0),
)
STRATEGIES = (*POLICIES, "DQN", "PPO")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def print_report(episodes: int = 5, minutes: int = 60, seed: int = 2026,
                 drivers: int = 8, restaurants: int = 5) -> None:
    if episodes < 1 or minutes < 1 or drivers < 1 or restaurants < 1:
        raise ValueError("episodes, minutes, drivers, and restaurants must be positive")
    trained = set(available_policies())
    missing = [name for name in ("DQN", "PPO") if name not in trained]
    if missing:
        missing_names = " and ".join(missing)
        raise FileNotFoundError(
            f"Missing trained {missing_names} checkpoint(s). Train them first with "
            "python3 -m food_delivery_game.rl.train --algorithm dqn|ppo --episodes 500"
        )
    if drivers > MAX_DRIVERS:
        raise ValueError(f"DQN and PPO support at most {MAX_DRIVERS} delivery partners")

    print(f"MUNCHKIN STRATEGY REPORT  |  {episodes} matched episode(s) per scenario")
    print(f"Setup: {minutes} simulated minutes, {drivers} delivery partners, "
          f"{restaurants} restaurants; each delivery earns 10 points.\n")

    for scenario_index, (label, arrival_rate) in enumerate(SCENARIOS):
        print(f"{label.upper()} — {arrival_rate:g} orders/minute")
        print(f"{'Strategy':34} {'Delivered':>10} {'Reward pts':>12} "
              f"{'Avg delivery min':>17} {'Travel distance':>17}")
        print("-" * 96)
        for strategy in STRATEGIES:
            metrics = []
            for episode in range(episodes):
                episode_seed = seed + scenario_index * 100_000 + episode
                config = SimulationConfig(
                    restaurant_count=restaurants,
                    driver_count=drivers,
                    orders_per_minute=arrival_rate,
                    episode_minutes=minutes,
                    seed=episode_seed,
                )
                # Recreate per episode so any stochastic PPO sampling is isolated
                # and reproducible for the same strategy/scenario/seed sequence.
                policy = create_policy(strategy) if strategy in POLICIES else create_rl_policy(strategy)
                if strategy == "PPO":
                    policy.agent.generator.manual_seed(episode_seed)
                metrics.append(run_policy_episode(policy, config, seed=episode_seed))
            delivered = _mean([float(item.delivered) for item in metrics])
            reward = _mean([item.total_reward for item in metrics])
            delivery_times = [item.average_delivery_time for item in metrics if item.delivered]
            avg_time = _mean(delivery_times)
            distance = _mean([item.total_travel_distance for item in metrics])
            time_text = f"{avg_time:.2f}" if math.isfinite(avg_time) else "n/a"
            print(f"{strategy:34} {delivered:10.2f} {reward:12.1f} "
                  f"{time_text:>17} {distance:17.1f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print matched-seed results for all Munchkin strategies at three demand levels"
    )
    parser.add_argument("--episodes", type=int, default=5,
                        help="matched random seeds per strategy and demand level (default: 5)")
    parser.add_argument("--minutes", type=int, default=60,
                        help="simulated minutes per episode (default: 60)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--drivers", type=int, default=8)
    parser.add_argument("--restaurants", type=int, default=5)
    args = parser.parse_args()
    try:
        print_report(args.episodes, args.minutes, args.seed, args.drivers, args.restaurants)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
