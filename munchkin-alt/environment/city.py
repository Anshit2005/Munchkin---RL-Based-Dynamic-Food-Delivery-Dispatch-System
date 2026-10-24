"""Grid geometry: Manhattan distance and L-shaped movement (x first, then y)."""
from __future__ import annotations

EPS = 1e-9


def manhattan(ax: float, ay: float, bx: float, by: float) -> float:
    return abs(ax - bx) + abs(ay - by)


def travel_time(dist: float, speed: float) -> float:
    return dist / speed


def move_towards(x: float, y: float, tx: float, ty: float, budget: float):
    """Move along x then y towards (tx, ty) using at most `budget` distance.

    Returns (new_x, new_y, distance_used)."""
    used = 0.0
    dx = tx - x
    step = max(-budget, min(budget, dx))
    x += step
    used += abs(step)
    budget -= abs(step)
    if budget > EPS:
        dy = ty - y
        step = max(-budget, min(budget, dy))
        y += step
        used += abs(step)
    return x, y, used


def at_point(x: float, y: float, tx: float, ty: float) -> bool:
    return abs(x - tx) < EPS and abs(y - ty) < EPS
