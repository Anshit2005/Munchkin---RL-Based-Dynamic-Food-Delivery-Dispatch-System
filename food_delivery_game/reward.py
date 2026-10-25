"""Multi-objective dispatch reward, adapted from the Munchkin-alt reward model.

The game scoreboard remains 10 points per completed delivery. This reward is the
training/evaluation objective for RL and is reported separately from scoreboard points.
One main-project step represents one simulated minute, so per-tick waiting/late/idle
terms use order-minutes and driver-minutes directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    delivery: float = 10.0
    waiting: float = 2.0
    late: float = 4.0
    travel: float = 0.1
    batch_saving: float = 0.5
    idle: float = 1.0
    unfinished: float = 5.0
    invalid_action: float = 1.0


def minute_reward(config: RewardConfig, *, delivered: int, waiting: int, overdue: int,
                  distance: float, idle: int, unfinished: int = 0,
                  batch_saving: float = 0.0, invalid: int = 0) -> dict[str, float]:
    """Return per-minute reward components using the alternate project's weights."""
    return {
        "delivery": config.delivery * delivered,
        "wait": -config.waiting * waiting,
        "late": -config.late * overdue,
        "travel": -config.travel * distance,
        "batch": config.batch_saving * max(batch_saving, 0.0),
        "idle": -config.idle * idle,
        "terminal": -config.unfinished * unfinished,
        "invalid": -config.invalid_action * invalid,
    }
