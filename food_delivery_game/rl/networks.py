"""Shared pairwise action-value/policy network for dispatch decisions."""

from __future__ import annotations

import torch
from torch import nn

from .state import (DRIVER_FEATURES, DRIVER_OFFSET, GLOBAL_FEATURES, GLOBAL_OFFSET,
                    MAX_DRIVERS, MAX_ORDERS, N_ACTIONS, OBS_SIZE, ORDER_FEATURES,
                    ORDER_OFFSET, PAIR_OFFSET, WAIT_ACTION)


def _mlp(sizes, output_gain: float | None = None):
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    if output_gain is not None:
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(net[-1].weight, gain=output_gain)
    return net


class PairwiseActionNetwork(nn.Module):
    """Score feasible (driver, order) pairs with shared weights and a separate WAIT head."""

    def __init__(self, hidden: int = 128, output_gain: float | None = None):
        super().__init__()
        context_size = GLOBAL_FEATURES + DRIVER_FEATURES + ORDER_FEATURES
        self.pair_net = _mlp([DRIVER_FEATURES + ORDER_FEATURES + 1 + context_size,
                              hidden, hidden, 1], output_gain)
        self.wait_net = _mlp([context_size, hidden, 1], output_gain)

    def forward(self, observation):
        batch = observation.shape[0]
        drivers = observation[:, DRIVER_OFFSET:ORDER_OFFSET].reshape(batch, MAX_DRIVERS, DRIVER_FEATURES)
        orders = observation[:, ORDER_OFFSET:PAIR_OFFSET].reshape(batch, MAX_ORDERS, ORDER_FEATURES)
        pairs = observation[:, PAIR_OFFSET:GLOBAL_OFFSET].reshape(batch, MAX_DRIVERS, MAX_ORDERS, 1)
        globals_ = observation[:, GLOBAL_OFFSET:GLOBAL_OFFSET + GLOBAL_FEATURES]

        driver_present = drivers[..., 0:1]
        order_present = orders[..., 0:1]
        driver_pool = (drivers * driver_present).sum(1) / driver_present.sum(1).clamp(min=1.0)
        order_pool = (orders * order_present).sum(1) / order_present.sum(1).clamp(min=1.0)
        context = torch.cat([globals_, driver_pool, order_pool], dim=1)

        b, d, o = (pairs.squeeze(-1) > 0).nonzero(as_tuple=True)
        pair_context = context[b]
        pair_inputs = torch.cat([drivers[b, d], orders[b, o], pairs[b, d, o], pair_context], dim=1)
        pair_scores = self.pair_net(pair_inputs).squeeze(-1)
        scores = observation.new_zeros((batch, MAX_DRIVERS, MAX_ORDERS))
        scores = scores.index_put((b, d, o), pair_scores)
        wait_score = self.wait_net(context)
        return torch.cat([scores.reshape(batch, WAIT_ACTION), wait_score], dim=1)


class MLPActionNetwork(nn.Module):
    def __init__(self, hidden: int = 256, output_gain: float | None = None):
        super().__init__()
        self.net = _mlp([OBS_SIZE, hidden, hidden, N_ACTIONS], output_gain)

    def forward(self, observation):
        return self.net(observation)


def make_network(arch: str = "pairwise", hidden: int = 128, output_gain: float | None = None):
    if arch == "pairwise":
        return PairwiseActionNetwork(hidden, output_gain)
    if arch == "mlp":
        return MLPActionNetwork(hidden, output_gain)
    raise ValueError(f"unknown network architecture: {arch}")


def masked_values(values, mask):
    return values.masked_fill(~mask, -1e9)

