"""Reward: R = w1*D - w2*W - w3*L - w4*T + w5*B - w6*I, plus terminal and invalid-action terms.

Per simulated tick:
  D deliveries completed this tick
  W waiting penalty: created-but-undelivered orders, in order-minutes (n/60 per tick)
  L lateness: undelivered orders past their deadline, in order-minutes (n/60 per tick)
  T distance travelled by all drivers this tick
  B batching benefit: distance saved versus separate trips (credited at pickup)
  I idle penalty: available drivers while unassigned orders exist, driver-minutes (n/60 per tick)
Terminal: each order still undelivered at t=180 costs `undelivered_penalty`.
Every component is tracked separately so the agent cannot silently optimise one metric.
"""
from __future__ import annotations

from environment.config import RewardConfig
from simulation.simulator import TickEvents

COMPONENTS = ("delivery", "wait", "late", "travel", "batch", "idle", "terminal", "invalid")


class RewardBreakdown:
    __slots__ = COMPONENTS

    def __init__(self):
        for c in COMPONENTS:
            setattr(self, c, 0.0)

    def add(self, other: "RewardBreakdown"):
        for c in COMPONENTS:
            setattr(self, c, getattr(self, c) + getattr(other, c))

    @property
    def total(self) -> float:
        return sum(getattr(self, c) for c in COMPONENTS)

    def to_dict(self) -> dict:
        d = {c: getattr(self, c) for c in COMPONENTS}
        d["total"] = self.total
        return d


def tick_reward(ev: TickEvents, rc: RewardConfig) -> RewardBreakdown:
    b = RewardBreakdown()
    b.delivery = rc.w_delivery * ev.n_delivered
    b.wait = -rc.w_wait * ev.n_waiting / 60.0
    b.late = -rc.w_late * ev.n_overdue / 60.0
    b.travel = -rc.w_travel * ev.distance
    b.batch = rc.w_batch * ev.batch_saving
    b.idle = -rc.w_idle * ev.n_idle / 60.0
    return b


def terminal_reward(n_undelivered: int, rc: RewardConfig) -> float:
    return -rc.undelivered_penalty * n_undelivered
