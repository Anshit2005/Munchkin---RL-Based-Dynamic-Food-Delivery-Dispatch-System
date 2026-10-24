"""PPO networks: a policy network (action logits) and a separate value network."""
from __future__ import annotations

import torch.nn as nn

from environment.constants import OBS_DIM
from rl.common.networks import make_action_net, mlp


class PolicyNetwork(nn.Module):
    """pi_theta(a | s): logits over the 193 actions (masking is applied by the agent)."""

    def __init__(self, arch: str = "pairwise", hidden: int = 128):
        super().__init__()
        self.arch, self.hidden = arch, hidden
        self.net = make_action_net(arch, hidden, out_gain=0.01)

    def forward(self, obs):
        return self.net(obs)


class ValueNetwork(nn.Module):
    """V(s): scalar state value."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = mlp([OBS_DIM, hidden, hidden, 1], out_gain=1.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
