"""Networks shared by DQN and PPO.

Two architectures over the fixed 481-d observation:
  * "mlp"      - plain MLP: 481 -> H -> H -> 193.
  * "pairwise" - a shared MLP scores every (driver, order) pair from [driver feats, order feats,
                 pair ETA, global + pooled context]; a separate head scores WAIT. Permutation-aware
                 and far more sample-efficient, since the same weights are reused for all 192 pairs.
Both output one value per action (Q-values for DQN, logits for PPO).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from environment.constants import (DRIVER_FEATS, GLOBAL_FEATS, MAX_DRIVERS, MAX_ORDERS, N_ACTIONS,
                                   OBS_DIM, ORDER_FEATS)

D_END = MAX_DRIVERS * DRIVER_FEATS
O_END = D_END + MAX_ORDERS * ORDER_FEATS
P_END = O_END + MAX_DRIVERS * MAX_ORDERS
NEG_INF = -1e9


def mlp(sizes, out_gain=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    if out_gain is not None:      # orthogonal init (used by PPO)
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(net[-1].weight, gain=out_gain)
    return net


class MLPActionNet(nn.Module):
    def __init__(self, hidden: int = 256, out_gain=None):
        super().__init__()
        self.net = mlp([OBS_DIM, hidden, hidden, N_ACTIONS], out_gain)

    def forward(self, obs):
        return self.net(obs)


class PairwiseActionNet(nn.Module):
    def __init__(self, hidden: int = 128, out_gain=None):
        super().__init__()
        ctx = GLOBAL_FEATS + DRIVER_FEATS + ORDER_FEATS
        self.pair = mlp([DRIVER_FEATS + ORDER_FEATS + 1 + ctx, hidden, hidden, 1], out_gain)
        self.wait = mlp([ctx, hidden, 1], out_gain)

    def forward(self, obs):
        B = obs.shape[0]
        drv = obs[:, :D_END].reshape(B, MAX_DRIVERS, DRIVER_FEATS)
        orr = obs[:, D_END:O_END].reshape(B, MAX_ORDERS, ORDER_FEATS)
        pair = obs[:, O_END:P_END].reshape(B, MAX_DRIVERS, MAX_ORDERS, 1)
        glb = obs[:, P_END:]
        dp = drv[..., 0:1]
        op = orr[..., 0:1]
        drv_pool = (drv * dp).sum(1) / dp.sum(1).clamp(min=1.0)
        ord_pool = (orr * op).sum(1) / op.sum(1).clamp(min=1.0)
        ctx = torch.cat([glb, drv_pool, ord_pool], dim=1)
        # Only valid pairs have a non-zero pair feature (the observation floors it at 1e-3), and
        # invalid actions are always masked, so score just the valid pairs (~10 of 192).
        b, d, o = (pair.squeeze(-1) > 0).nonzero(as_tuple=True)
        x = torch.cat([drv[b, d], orr[b, o], pair[b, d, o], ctx[b]], dim=-1)
        s = self.pair(x).squeeze(-1)
        scores = torch.zeros(B, MAX_DRIVERS, MAX_ORDERS, dtype=s.dtype).index_put((b, d, o), s)
        return torch.cat([scores.reshape(B, MAX_DRIVERS * MAX_ORDERS), self.wait(ctx)], dim=1)


def make_action_net(arch: str, hidden: int, out_gain=None) -> nn.Module:
    if arch == "mlp":
        return MLPActionNet(hidden, out_gain)
    if arch == "pairwise":
        return PairwiseActionNet(hidden, out_gain)
    raise ValueError(f"unknown arch {arch}")


def masked_fill(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~mask, NEG_INF)
