"""Compare every available Munchkin policy across varied city configurations.

Run from the repository root:
    python3 -m food_delivery_game.rl.parameter_report

The report uses matched seeds and writes raw episode data, an aggregate CSV, and
a human-readable Markdown summary under ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from ..baselines import POLICIES, create_policy
from ..config import PROJECT_ROOT, SimulationConfig
from ..simulation.runner import EpisodeMetrics, run_policy_episode
from .policies import create_rl_policy
from .registry import available_policies
from .state import MAX_DRIVERS


@dataclass(frozen=True)
class Scenario:
    name: str
    orders_per_minute: float
    traffic: str
    restaurants: int
    drivers: int


# Demand sweep holds resources fixed; each of the following sweeps then varies
# one resource or traffic independently. The final two stress cases vary them
# together to expose undersupply and generous-capacity behavior.
SCENARIOS = (
    Scenario("low_demand", 2.0, "normal", 5, 8),
    Scenario("medium_demand", 5.0, "normal", 5, 8),
    Scenario("high_demand", 8.0, "normal", 5, 8),
    Scenario("light_traffic", 5.0, "light", 5, 8),
    Scenario("heavy_traffic", 5.0, "heavy", 5, 8),
    Scenario("few_restaurants", 5.0, "normal", 3, 8),
    Scenario("many_restaurants", 5.0, "normal", 8, 8),
    Scenario("few_drivers", 5.0, "normal", 5, 3),
    Scenario("many_drivers", 5.0, "normal", 5, 16),
    Scenario("undersupplied_high_demand", 8.0, "heavy", 3, 3),
    Scenario("well_resourced_high_demand", 8.0, "heavy", 8, 16),
)

METRICS = (
    ("delivered", lambda item: float(item.delivered)),
    ("game_points", lambda item: item.total_reward),
    ("rl_reward", lambda item: item.total_training_reward),
    ("average_delivery_minutes", lambda item: item.average_delivery_time),
    ("travel_distance", lambda item: item.total_travel_distance),
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _strategy_names() -> list[str]:
    available = set(available_policies())
    missing = [name for name in ("DQN", "PPO") if name not in available]
    if missing:
        joined = " and ".join(missing)
        raise FileNotFoundError(
            f"Missing or incompatible {joined} checkpoint(s). Train them first with "
            "python3 -m food_delivery_game.rl.train --algorithm dqn --episodes 500 "
            "and python3 -m food_delivery_game.rl.train --algorithm ppo --episodes 500."
        )
    return [*POLICIES, "DQN", "PPO"]


def _validate_args(episodes: int, minutes: int) -> None:
    if episodes < 1 or minutes < 1:
        raise ValueError("episodes and minutes must be positive integers")
    if max(scenario.drivers for scenario in SCENARIOS) > MAX_DRIVERS:
        raise ValueError("A report scenario exceeds the RL models' supported driver count")


def _summarize(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for name, value in METRICS:
        # A zero-delivery episode has no defined mean delivery time. Exclude it
        # from that metric while retaining it for all other measurements.
        selected = [value(item) for item in episodes
                    if name != "average_delivery_minutes" or item.delivered > 0]
        mean, std = _mean_std(selected)
        summary[f"{name}_mean"] = mean
        summary[f"{name}_std"] = std
    return summary


def _write_reports(output_dir: Path, rows: list[dict], raw_rows: list[dict],
                   episodes: int, minutes: int, seed: int) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "parameter_report_summary.csv"
    raw_path = output_dir / "parameter_report_episodes.csv"
    markdown_path = output_dir / "parameter_report.md"

    summary_fields = [
        "scenario", "orders_per_minute", "traffic", "restaurants", "drivers",
        "strategy", "episodes",
        *(f"{name}_{stat}" for name, _ in METRICS for stat in ("mean", "std")),
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(rows)

    raw_fields = [
        "scenario", "episode", "seed", "orders_per_minute", "traffic",
        "restaurants", "drivers", "strategy", "delivered", "game_points",
        "rl_reward", "average_delivery_minutes", "travel_distance",
    ]
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    lines = [
        "# Munchkin model parameter report",
        "",
        f"Matched episodes per strategy and configuration: **{episodes}**",
        f"Simulated minutes per episode: **{minutes}**",
        f"Base seed: **{seed}**",
        "",
        "All strategies use the same seed within each configuration. PPO uses deterministic masked action selection. "
        "Means and sample standard deviations are shown; average delivery time excludes episodes with no deliveries.",
        "Game points determine the game score. RL reward is a separate shaped objective.",
        "",
        "## Results by configuration",
        "",
        "| Configuration | Game-points leader(s) | Shaped-reward leader(s) |",
        "|---|---|---|",
    ]
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario.name]
        points_best = max(row["game_points_mean"] for row in scenario_rows)
        reward_best = max(row["rl_reward_mean"] for row in scenario_rows)
        point_leaders = ", ".join(row["strategy"] for row in scenario_rows
                                  if math.isclose(row["game_points_mean"], points_best,
                                                  rel_tol=1e-12, abs_tol=1e-9))
        reward_leaders = ", ".join(row["strategy"] for row in scenario_rows
                                   if math.isclose(row["rl_reward_mean"], reward_best,
                                                   rel_tol=1e-12, abs_tol=1e-9))
        lines.append(f"| {scenario.name} | {point_leaders} | {reward_leaders} |")
    lines.extend([
        "",
        "The two leader columns use different objectives: points track completed deliveries, while shaped reward also penalizes waiting, lateness, travel, idle capacity, and unfinished orders.",
        "",
        "| Configuration | Orders/min | Traffic | Restaurants | Drivers | Strategy | Delivered | Game points | RL reward | Avg delivery (min) | Distance |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        def fmt(metric: str, digits: int = 1) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            if not math.isfinite(mean):
                return "n/a"
            return f"{mean:.{digits}f} ± {std:.{digits}f}"

        lines.append(
            f"| {row['scenario']} | {row['orders_per_minute']:g} | {row['traffic']} "
            f"| {row['restaurants']} | {row['drivers']} | {row['strategy']} "
            f"| {fmt('delivered')} | {fmt('game_points')} | {fmt('rl_reward')} "
            f"| {fmt('average_delivery_minutes', 2)} | {fmt('travel_distance')} |"
        )
    lines.extend([
        "",
        "## Method",
        "",
        "The suite includes low/medium/high demand with resources fixed, an isolated traffic sweep, "
        "restaurant-count and driver-count sweeps, and two combined high-demand capacity stress cases. "
        "Every configuration is evaluated with all three heuristic baselines plus DQN and PPO. "
        "Five seeds are a practical screening sample; increase `--episodes` for more stable estimates.",
        "",
        "The raw per-episode measurements are in `parameter_report_episodes.csv`; aggregate means and "
        "standard deviations are in `parameter_report_summary.csv`.",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path, raw_path, markdown_path


def run_report(episodes: int = 5, minutes: int = 60, seed: int = 2026,
               output_dir: Path | None = None) -> tuple[Path, Path, Path]:
    _validate_args(episodes, minutes)
    strategies = _strategy_names()
    policies = {
        name: create_policy(name) if name in POLICIES else create_rl_policy(name)
        for name in strategies
    }
    summary_rows: list[dict] = []
    raw_rows: list[dict] = []

    print(f"MUNCHKIN PARAMETER REPORT | {len(SCENARIOS)} configurations | "
          f"{episodes} matched episode(s) each | {minutes} minutes per episode")
    for scenario_index, scenario in enumerate(SCENARIOS):
        print(f"Running {scenario.name}: {scenario.orders_per_minute:g} orders/min, "
              f"{scenario.traffic} traffic, {scenario.restaurants} restaurants, "
              f"{scenario.drivers} drivers", flush=True)
        config = SimulationConfig(
            restaurant_count=scenario.restaurants,
            driver_count=scenario.drivers,
            orders_per_minute=scenario.orders_per_minute,
            traffic_level=scenario.traffic,
            episode_minutes=minutes,
            seed=seed,
        )
        for strategy in strategies:
            metrics = []
            for episode in range(episodes):
                episode_seed = seed + scenario_index * 100_000 + episode
                result = run_policy_episode(policies[strategy], config, episode_seed)
                metrics.append(result)
                raw_rows.append({
                    "scenario": scenario.name,
                    "episode": episode + 1,
                    "seed": episode_seed,
                    "orders_per_minute": scenario.orders_per_minute,
                    "traffic": scenario.traffic,
                    "restaurants": scenario.restaurants,
                    "drivers": scenario.drivers,
                    "strategy": strategy,
                    "delivered": result.delivered,
                    "game_points": result.total_reward,
                    "rl_reward": result.total_training_reward,
                    "average_delivery_minutes": result.average_delivery_time if result.delivered else "",
                    "travel_distance": result.total_travel_distance,
                })
            summary = _summarize(metrics)
            summary_rows.append({
                "scenario": scenario.name,
                "orders_per_minute": scenario.orders_per_minute,
                "traffic": scenario.traffic,
                "restaurants": scenario.restaurants,
                "drivers": scenario.drivers,
                "strategy": strategy,
                "episodes": episodes,
                **summary,
            })

    paths = _write_reports(output_dir or PROJECT_ROOT / "results", summary_rows,
                           raw_rows, episodes, minutes, seed)
    print("\nReport files:")
    for path in paths:
        print(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all Munchkin baselines and RL models across city configurations"
    )
    parser.add_argument("--episodes", type=int, default=5,
                        help="matched seeds per strategy/configuration (default: 5)")
    parser.add_argument("--minutes", type=int, default=60,
                        help="simulated minutes per episode (default: 60)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results",
                        help="directory for Markdown and CSV reports")
    args = parser.parse_args()
    try:
        run_report(args.episodes, args.minutes, args.seed, args.output_dir)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
