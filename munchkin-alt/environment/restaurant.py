from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Restaurant:
    id: int
    x: float
    y: float
