"""Train a DQN or PPO dispatcher on the Munchkin simulator.

Examples from the project root:
    python -m food_delivery_game.rl.train --algorithm dqn --episodes 1000
    python -m food_delivery_game.rl.train --algorithm ppo --episodes 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..config import PROJECT_ROOT, SimulationConfig
from ..simulation.runner import run_policy_episode
from .dqn import DQNAgent, ReplayBuffer
from .environment import RLDeliveryEnvironment
from .policies import RLDispatchPolicy
from .ppo import PPOAgent
from .state import N_ACTIONS, OBS_SIZE, OBS_VERSION

MODEL_DIR = PROJECT_ROOT / "models"
DQN_MIN_REPLAY = 32
DQN_BATCH_SIZE = 128


def _episode_config(seed: int, minutes: int) -> SimulationConfig:
    rng = random.Random(seed)
    return SimulationConfig(
        restaurant_count=rng.randint(3, 8),
        driver_count=rng.randint(3, 16),
        orders_per_minute=round(rng.uniform(1.0, 8.0), 1),
        traffic_level=rng.choice(("light", "normal", "heavy")),
        travel_speed=60.0,
        episode_minutes=minutes,
        seed=seed,
    )


def _save(path: Path, algorithm: str, agent, agent_config: dict, episodes: int,
          reward_scale: float, validation_score: float, training_episodes: int,
          gradient_updates: int | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "algorithm": algorithm,
        "state_dict": agent.state_dict(),
        "agent_config": agent_config,
        "checkpoint_episode": episodes,
        "training_episodes": training_episodes,
        "gradient_updates": gradient_updates,
        "observation_size": OBS_SIZE,
        "action_count": N_ACTIONS,
        "observation_version": OBS_VERSION,
        "reward_scale": reward_scale,
        "validation_score": validation_score,
    }
    torch.save(checkpoint, path)
    metadata_path = path.with_name(f"{path.stem}.meta.json")
    metadata_path.write_text(json.dumps({
        key: checkpoint[key] for key in (
            "algorithm", "checkpoint_episode", "training_episodes", "gradient_updates",
            "observation_size", "action_count", "observation_version", "reward_scale",
            "validation_score",
        )
    }, indent=2) + "\n", encoding="utf-8")


def _validate(agent, algorithm: str, episodes: int, minutes: int, seed: int) -> float:
    """Average alternate-model reward across fixed low/medium/heavy conditions."""
    scenarios = ((2.0, "light"), (5.0, "normal"), (8.0, "heavy"))
    policy = RLDispatchPolicy(algorithm, agent, algorithm)
    rewards = []
    for scenario_index, (rate, traffic_level) in enumerate(scenarios):
        for episode in range(episodes):
            episode_seed = seed + scenario_index * 100_000 + episode
            config = SimulationConfig(
                restaurant_count=5,
                driver_count=8,
                orders_per_minute=rate,
                traffic_level=traffic_level,
                travel_speed=60.0,
                episode_minutes=minutes,
                seed=episode_seed,
            )
            rewards.append(run_policy_episode(policy, config, episode_seed).total_training_reward)
    return sum(rewards) / len(rewards)


def train_dqn(episodes: int, minutes: int, seed: int, hidden: int = 128,
              learning_rate: float = 3e-4, eval_every: int = 25,
              validation_episodes: int = 3, reward_scale: float = 0.1) -> Path:
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
    best_validation = float("-inf")
    gradient_updates = 0

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["episode", "score", "shaped_return", "steps", "epsilon", "loss", "gradient_updates", "validation_score"])
        writer.writeheader()
        for episode in range(1, episodes + 1):
            env = RLDeliveryEnvironment(_episode_config(seed + episode, minutes))
            obs, mask, _ = env.reset(seed=seed + episode)
            episode_return, shaped_return, episode_steps, last_loss = 0.0, 0.0, 0, float("nan")
            done = False
            while not done:
                epsilon = max(0.05, 1.0 - 0.95 * global_step / total_steps)
                action = agent.act(obs, mask, epsilon)
                next_obs, reward, done, info = env.step(action)
                replay.add(obs, action, reward * reward_scale, next_obs, done, info["action_mask"])
                obs, mask = next_obs, info["action_mask"]
                global_step += 1
                episode_steps += 1
                episode_return += info["score_reward"]
                shaped_return += reward
                if len(replay) >= DQN_MIN_REPLAY and global_step % 4 == 0:
                    last_loss = agent.learn(replay.sample(DQN_BATCH_SIZE, rng))
                    gradient_updates += 1
                if global_step % target_sync == 0:
                    agent.target.load_state_dict(agent.q.state_dict())
            validation_text = ""
            if episode % eval_every == 0 or episode == episodes:
                validation = _validate(agent, "DQN", validation_episodes, minutes, seed + 900_000)
                validation_text = validation
                if validation > best_validation and gradient_updates > 0:
                    best_validation = validation
                    _save(output, "DQN", agent, config, episode, reward_scale, validation,
                          episodes, gradient_updates)
                print(f"DQN episode {episode}/{episodes}: score={env.sim.total_reward:.1f}, "
                      f"validation={validation:.1f}, best={best_validation:.1f}, "
                      f"epsilon={epsilon:.3f}", flush=True)
            writer.writerow({"episode": episode, "score": episode_return,
                             "shaped_return": shaped_return, "steps": episode_steps,
                             "epsilon": epsilon, "loss": last_loss,
                             "gradient_updates": gradient_updates,
                             "validation_score": validation_text})
    if gradient_updates == 0:
        raise RuntimeError("DQN training completed without a gradient update; no checkpoint was saved")
    return output


def train_ppo(episodes: int, minutes: int, seed: int, hidden: int = 128,
              learning_rate: float = 3e-4, eval_every: int = 25,
              validation_episodes: int = 3, reward_scale: float = 0.1) -> Path:
    np.random.seed(seed)
    config = {"hidden": hidden, "value_hidden": 256, "arch": "pairwise",
              "learning_rate": learning_rate, "seed": seed}
    agent = PPOAgent(**config)
    output = MODEL_DIR / "ppo.pt"
    log_path = MODEL_DIR / "ppo_training.csv"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    gamma, gae_lambda = 0.99, 0.95
    best_validation = float("-inf")

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["episode", "score", "shaped_return", "steps", "policy_loss", "value_loss", "validation_score"])
        writer.writeheader()
        for episode in range(1, episodes + 1):
            env = RLDeliveryEnvironment(_episode_config(seed + episode, minutes))
            obs, mask, _ = env.reset(seed=seed + episode)
            rollout = []
            episode_return = 0.0
            shaped_return = 0.0
            done = False
            while not done:
                action, logp, value = agent.act(obs, mask)
                next_obs, reward, done, info = env.step(action)
                rollout.append({"obs": obs, "mask": mask, "action": action, "logp": logp,
                                "value": value, "reward": reward * reward_scale, "done": done})
                obs, mask = next_obs, info["action_mask"]
                episode_return += info["score_reward"]
                shaped_return += reward
            stats = agent.update(rollout, gamma=gamma, gae_lambda=gae_lambda)
            validation_text = ""
            if episode % eval_every == 0 or episode == episodes:
                validation = _validate(agent, "PPO", validation_episodes, minutes, seed + 900_000)
                validation_text = validation
                if validation > best_validation:
                    best_validation = validation
                    _save(output, "PPO", agent, config, episode, reward_scale, validation,
                          episodes)
                print(f"PPO episode {episode}/{episodes}: score={env.sim.total_reward:.1f}, "
                      f"validation={validation:.1f}, best={best_validation:.1f}, "
                      f"policy_loss={stats['policy_loss']:.4f}", flush=True)
            writer.writerow({"episode": episode, "score": episode_return,
                             "shaped_return": shaped_return, "steps": len(rollout),
                             "policy_loss": stats["policy_loss"], "value_loss": stats["value_loss"],
                             "validation_score": validation_text})
    return output


def main():
    parser = argparse.ArgumentParser(description="Train a Munchkin DQN or PPO dispatcher")
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=25,
                        help="validate across low/medium/heavy demand every N episodes")
    parser.add_argument("--validation-episodes", type=int, default=3,
                        help="validation episodes per demand level")
    parser.add_argument("--reward-scale", type=float, default=0.1,
                        help="scale shaped rewards for stable neural-network updates")
    args = parser.parse_args()
    if args.episodes < 1 or args.minutes < 1 or args.hidden < 1 or args.learning_rate <= 0:
        parser.error("episodes, minutes, hidden must be positive and learning-rate must be > 0")
    if args.eval_every < 1 or args.validation_episodes < 1 or args.reward_scale <= 0:
        parser.error("eval-every, validation-episodes, and reward-scale must be positive")
    torch.set_num_threads(2)
    train = train_dqn if args.algorithm == "dqn" else train_ppo
    output = train(args.episodes, args.minutes, args.seed, args.hidden, args.learning_rate,
                   args.eval_every, args.validation_episodes, args.reward_scale)
    print(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
