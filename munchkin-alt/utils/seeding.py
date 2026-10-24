"""Disjoint seed pools: training / validation / test, plus global seeding."""
from __future__ import annotations

import random

import numpy as np

TRAIN_SEED_MAX = 1_000_000
VAL_SEEDS = list(range(1_000_000, 1_000_200))
TEST_SEEDS = list(range(2_000_000, 2_000_100))


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def split_seeds(split: str, n: int | None = None) -> list:
    seeds = {"val": VAL_SEEDS, "test": TEST_SEEDS}[split]
    return seeds[:n] if n else list(seeds)
