"""Train a DQN or PPO dispatcher on the Munchkin simulator.

Examples from the project root:
    python -m food_delivery_game.rl.train --algorithm dqn --episodes 1000
    python -m food_delivery_game.rl.train --algorithm ppo --episodes 1000
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from ..config import PROJECT_ROOT, SimulationConfig
from .dqn import DQNAgent, ReplayBuffer
from .environment import RLDeliveryEnvironment
from .ppo import PPOAgent
from .state import N_ACTIONS, OBS_SIZE

MODEL_DIR = PROJECT_ROOT / "models"


def _episode_config(seed: int, minutes: int) -> SimulationConfig:
    rng = random.Random(seed)
    return SimulationConfig(
        restaurant_count=rng.randint(1, 10),
        driver_count=rng.randint(3, 16),
        orders_per_minute=round(rng.uniform(0.5, 5.0), 1),
        travel_speed=60.0,
        episode_minutes=minutes,
        seed=seed,
    )


def _save(path: Path, algorithm: str, agent, agent_config: dict, episodes: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "algorithm": algorithm,
        "state_dict": agent.state_dict(),
        "agent_config": agent_config,
        "episodes": episodes,
        "observation_size": OBS_SIZE,
        "action_count": N_ACTIONS,
    }, path)


def train_dqn(episodes: int, minutes: int, seed: int, hidden: int = 128,
              learning_rate: float = 3e-4) -> Path:
    config = {"hidden": hidden, "arch": "pairwise", "learning_rate": learning_rate,
              "gamma": 0.99, "tau": 0.0, "seed": seed}
    agent = DQNAgent(**config)
    replay = ReplayBuffer(20_000)
    rng = np.random.default_rng(seed)
    output = MODEL_DIR / "dqn.pt"
    log_path = MODEL_DIR / "dqn_training.csv"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    global_step = 0
    total_steps = max(1, episodes * minutes * 3)
    target_sync = 1000

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["episode", "return", "steps", "epsilon", "loss"])
        writer.writeheader()
        for episode in range(1, episodes + 1):
            env = RLDeliveryEnvironment(_episode_config(seed + episode, minutes))
            obs, mask, _ = env.reset(seed=seed + episode)
            episode_return, episode_steps, last_loss = 0.0, 0, float("nan")
            done = False
            while not done:
                epsilon = max(0.05, 1.0 - 0.95 * global_step / total_steps)
                action = agent.act(obs, mask, epsilon)
                next_obs, reward, done, info = env.step(action)
                replay.add(obs, action, reward, next_obs, done, info["action_mask"])
                obs, mask = next_obs, info["action_mask"]
                global_step += 1
                episode_steps += 1
                episode_return += reward
                if len(replay) >= 256 and global_step % 4 == 0:
                    last_loss = agent.learn(replay.sample(128, rng))
                if global_step % target_sync == 0:
                    agent.target.load_state_dict(agent.q.state_dict())
            writer.writerow({"episode": episode, "return": episode_return,
                             "steps": episode_steps, "epsilon": epsilon, "loss": last_loss})
            if episode % 25 == 0 or episode == episodes:
                _save(output, "DQN", agent, config, episode)
                print(f"DQN episode {episode}/{episodes}: return={episode_return:.1f}, "
                      f"steps={episode_steps}, epsilon={epsilon:.3f}", flush=True)
    return output


def train_ppo(episodes: int, minutes: int, seed: int, hidden: int = 128,
              learning_rate: float = 3e-4) -> Path:
    np.random.seed(seed)
    config = {"hidden": hidden, "value_hidden": 256, "arch": "pairwise",
              "learning_rate": learning_rate, "seed": seed}
    agent = PPOAgent(**config)
    output = MODEL_DIR / "ppo.pt"
    log_path = MODEL_DIR / "ppo_training.csv"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    gamma, gae_lambda = 0.99, 0.95

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["episode", "return", "steps", "policy_loss", "value_loss"])
        writer.writeheader()
        for episode in range(1, episodes + 1):
            env = RLDeliveryEnvironment(_episode_config(seed + episode, minutes))
            obs, mask, _ = env.reset(seed=seed + episode)
            rollout = []
            episode_return = 0.0
            done = False
            while not done:
                action, logp, value = agent.act(obs, mask)
                next_obs, reward, done, info = env.step(action)
                rollout.append({"obs": obs, "mask": mask, "action": action, "logp": logp,
                                "value": value, "reward": reward, "done": done})
                obs, mask = next_obs, info["action_mask"]
                episode_return += reward
            stats = agent.update(rollout, gamma=gamma, gae_lambda=gae_lambda)
            writer.writerow({"episode": episode, "return": episode_return, "steps": len(rollout),
                             "policy_loss": stats["policy_loss"], "value_loss": stats["value_loss"]})
            if episode % 25 == 0 or episode == episodes:
                _save(output, "PPO", agent, config, episode)
                print(f"PPO episode {episode}/{episodes}: return={episode_return:.1f}, "
                      f"steps={len(rollout)}, policy_loss={stats['policy_loss']:.4f}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Train a Munchkin DQN or PPO dispatcher")
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.episodes < 1 or args.minutes < 1 or args.hidden < 1 or args.learning_rate <= 0:
        parser.error("episodes, minutes, hidden must be positive and learning-rate must be > 0")
    torch.set_num_threads(2)
    train = train_dqn if args.algorithm == "dqn" else train_ppo
    output = train(args.episodes, args.minutes, args.seed, args.hidden, args.learning_rate)
    print(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
