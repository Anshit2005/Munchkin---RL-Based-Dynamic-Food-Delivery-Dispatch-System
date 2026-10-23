"""Shared interfaces for dispatch policies."""

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment import DeliveryEnvironment


@dataclass(frozen=True)
class Assignment:
    driver_id: int
    order_id: int


class DispatchPolicy(Protocol):
    name: str

    def select_assignment(self, env: "DeliveryEnvironment") -> Assignment | None:
        """Choose one valid ready-order/available-driver pair, or no action."""
