"""Load trained DQN/PPO checkpoints as regular dispatch policies."""

from __future__ import annotations

from ..baselines.base import Assignment
from .registry import MODEL_FILES
from .state import MAX_ORDERS, N_ACTIONS, OBS_SIZE, OBS_VERSION, WAIT_ACTION, encode_state


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
            # Use greedy masked action selection so the same state always yields
            # the same dispatch in CLI evaluation and the interactive match.
            action, _, _ = self.agent.act(observation, mask, deterministic=True)
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
    if checkpoint.get("observation_size") != OBS_SIZE or checkpoint.get("action_count") != N_ACTIONS:
        raise RuntimeError(f"Checkpoint {path} does not match the current Munchkin RL state/action layout")
    if checkpoint.get("observation_version") != OBS_VERSION:
        raise RuntimeError(
            f"Checkpoint {path} uses an older observation layout. Retrain {name} with "
            f"python3 -m food_delivery_game.rl.train --algorithm {name.lower()} --episodes 500"
        )
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

