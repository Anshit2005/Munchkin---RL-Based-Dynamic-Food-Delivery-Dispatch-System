"""Heuristic dispatch policies used as comparison baselines."""

from .base import Assignment, DispatchPolicy
from .minimum_eta import MinimumETAPolicy
from .nearest_driver import NearestDriverPolicy
from .oldest_order import OldestOrderPolicy

POLICIES: dict[str, type[DispatchPolicy]] = {
    NearestDriverPolicy.name: NearestDriverPolicy,
    OldestOrderPolicy.name: OldestOrderPolicy,
    MinimumETAPolicy.name: MinimumETAPolicy,
}


def create_policy(name: str) -> DispatchPolicy:
    """Create a fresh policy instance by its display name."""
    try:
        return POLICIES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown dispatch policy: {name}") from exc


__all__ = ["Assignment", "DispatchPolicy", "POLICIES", "create_policy"]
