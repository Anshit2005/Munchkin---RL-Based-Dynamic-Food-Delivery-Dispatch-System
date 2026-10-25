"""Configuration for one delivery-simulation episode."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SimulationConfig:
    width: int = 100
    height: int = 70
    restaurant_count: int = 5
    driver_count: int = 8
    orders_per_minute: float = 5.0
    travel_speed: float = 60.0
    preparation_min_minutes: int = 0
    preparation_max_minutes: int = 1
    episode_minutes: int = 3
    seed: int = 7

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Map dimensions must be positive.")
        if self.restaurant_count < 1:
            raise ValueError("At least one restaurant is required.")
        if self.driver_count < 1:
            raise ValueError("At least one driver is required.")
        if self.orders_per_minute < 0:
            raise ValueError("Order arrival rate cannot be negative.")
        if self.travel_speed <= 0:
            raise ValueError("Travel speed must be positive.")
        if self.preparation_min_minutes < 0:
            raise ValueError("Preparation time cannot be negative.")
        if self.preparation_max_minutes < self.preparation_min_minutes:
            raise ValueError("Maximum preparation time must be >= minimum.")
        if self.episode_minutes < 1:
            raise ValueError("An episode must last at least one simulated minute.")
