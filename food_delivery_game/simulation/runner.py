"""Run matched simulation episodes and collect comparison metrics."""

from dataclasses import dataclass

from ..baselines import POLICIES, create_policy
from ..config import SimulationConfig
from ..environment import DeliveryEnvironment


@dataclass(frozen=True)
class EpisodeMetrics:
    strategy: str
    seed: int
    delivered: int
    total_reward: float
    average_delivery_time: float
    total_travel_distance: float


def run_episode(strategy: str, config: SimulationConfig, seed: int | None = None) -> EpisodeMetrics:
    """Run one strategy for one fixed-horizon episode."""
    return run_policy_episode(create_policy(strategy), config, seed)


def run_policy_episode(policy, config: SimulationConfig, seed: int | None = None) -> EpisodeMetrics:
    """Run any policy implementing ``select_assignment`` for one episode."""
    env = DeliveryEnvironment(config)
    env.reset(seed=seed)
    while not env.done:
        env.step(policy)
    delivery_times = [o.delivered_at - o.created_at for o in env.delivered_orders if o.delivered_at is not None]
    return EpisodeMetrics(
        strategy=policy.name,
        seed=config.seed if seed is None else seed,
        delivered=len(env.delivered_orders),
        total_reward=env.total_reward,
        average_delivery_time=sum(delivery_times) / len(delivery_times) if delivery_times else 0.0,
        total_travel_distance=env.total_travel_distance,
    )


def compare_policies(config: SimulationConfig, episodes: int = 20) -> dict[str, list[EpisodeMetrics]]:
    """Compare all baselines using the same seeds for every policy."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    return {
        name: [run_episode(name, config, seed=config.seed + i) for i in range(episodes)]
        for name in POLICIES
    }


def average_metrics(results: list[EpisodeMetrics]) -> tuple[float, float, float, float]:
    count = len(results)
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        sum(r.delivered for r in results) / count,
        sum(r.total_reward for r in results) / count,
        sum(r.average_delivery_time for r in results) / count,
        sum(r.total_travel_distance for r in results) / count,
    )
