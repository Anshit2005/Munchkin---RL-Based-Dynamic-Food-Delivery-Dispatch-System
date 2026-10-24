"""DQN Q-network: maps an observation to one Q-value per action."""
from __future__ import annotations

import torch.nn as nn

from rl.common.networks import make_action_net


class QNetwork(nn.Module):
    def __init__(self, arch: str = "pairwise", hidden: int = 128):
        super().__init__()
        self.arch, self.hidden = arch, hidden
        self.net = make_action_net(arch, hidden)

    def forward(self, obs):
        return self.net(obs)
