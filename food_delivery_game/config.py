"""Configuration for one delivery-simulation episode."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAFFIC_MULTIPLIERS = {"light": 1.2, "normal": 1.0, "heavy": 0.7}


@dataclass(frozen=True)
class SimulationConfig:
    width: int = 100
    height: int = 70
    restaurant_count: int = 5
    driver_count: int = 8
    orders_per_minute: float = 5.0
    travel_speed: float = 60.0
    preparation_min_minutes: float = 0.15
    preparation_max_minutes: float = 1.0
    preparation_mean_minutes: float = 0.5
    preparation_std_minutes: float = 0.17
    preparation_estimate_noise: float = 0.2
    deadline_slack_minutes: float = 5.0
    backlog_fraction: float = 0.5
    traffic_level: str = "normal"
    traffic_noise: float = 0.1
    customer_radius: float = 28.0
    min_restaurant_spacing: float = 12.0
    episode_minutes: int = 60
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
        if not self.preparation_min_minutes <= self.preparation_mean_minutes <= self.preparation_max_minutes:
            raise ValueError("Mean preparation time must be between minimum and maximum.")
        if self.preparation_std_minutes < 0 or self.preparation_estimate_noise < 0:
            raise ValueError("Preparation variability cannot be negative.")
        if self.deadline_slack_minutes < 0 or self.backlog_fraction < 0 or self.traffic_noise < 0:
            raise ValueError("Deadline slack, initial backlog, and traffic noise cannot be negative.")
        if self.customer_radius < 0 or self.min_restaurant_spacing < 0:
            raise ValueError("Customer radius and restaurant spacing cannot be negative.")
        if self.traffic_level not in TRAFFIC_MULTIPLIERS:
            raise ValueError("Traffic level must be light, normal, or heavy.")
        if self.episode_minutes < 1:
            raise ValueError("An episode must last at least one simulated minute.")

    @property
    def nominal_speed(self) -> float:
        return self.travel_speed * TRAFFIC_MULTIPLIERS[self.traffic_level]
