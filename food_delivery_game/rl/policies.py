"""Load trained DQN/PPO checkpoints as regular dispatch policies."""

from __future__ import annotations

from ..baselines.base import Assignment
from .registry import MODEL_FILES
from .state import MAX_ORDERS, WAIT_ACTION, encode_state


class RLDispatchPolicy:
    def __init__(self, name: str, agent, algorithm: str):
        self.name = name
        self.agent = agent
        self.algorithm = algorithm

    def select_assignment(self, env):
        observation, mask, orders = encode_state(env)
        if self.algorithm == "DQN":
            action = self.agent.act(observation, mask, epsilon=0.0)
        else:
            # PPO represents a categorical policy. Sampling follows the trained
            # policy; greedy argmax can collapse to WAIT even when dispatch
            # actions collectively carry most of the policy probability.
            action, _, _ = self.agent.act(observation, mask, deterministic=False)
        if action == WAIT_ACTION:
            return None
        driver_index, order_slot = divmod(action, MAX_ORDERS)
        if order_slot >= len(orders) or driver_index >= len(env.drivers):
            return None
        driver, order = env.drivers[driver_index], orders[order_slot]
        return Assignment(driver.driver_id, order.order_id, f"{self.algorithm} selected this legal driver/order pair")


def available_policies() -> list[str]:
    return [name for name, path in MODEL_FILES.items() if path.is_file()]


def create_rl_policy(name: str) -> RLDispatchPolicy:
    import torch

    if name not in MODEL_FILES:
        raise ValueError(f"Unknown RL policy: {name}")
    path = MODEL_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"No trained {name} model at {path}. Train it with python -m food_delivery_game.rl.train.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("observation_size") != 1832 or checkpoint.get("action_count") != 1025:
        raise RuntimeError(f"Checkpoint {path} does not match the current Munchkin RL state/action layout")
    algorithm = checkpoint["algorithm"]
    cfg = checkpoint["agent_config"]
    if algorithm == "DQN":
        from .dqn import DQNAgent
        agent = DQNAgent(**cfg)
        agent.q.load_state_dict(checkpoint["state_dict"])
        agent.target.load_state_dict(checkpoint["state_dict"])
        agent.q.eval()
    elif algorithm == "PPO":
        from .ppo import PPOAgent
        agent = PPOAgent(**cfg)
        agent.policy.load_state_dict(checkpoint["state_dict"]["policy"])
        agent.value.load_state_dict(checkpoint["state_dict"]["value"])
        agent.policy.eval()
    else:
        raise RuntimeError(f"Unsupported RL checkpoint algorithm: {algorithm}")
    return RLDispatchPolicy(name, agent, algorithm)

