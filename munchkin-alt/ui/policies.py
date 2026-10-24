"""Policy registry for the UI: baselines always available, RL only if a compatible checkpoint exists."""
from __future__ import annotations

import json
from pathlib import Path

from baselines import MinETAPolicy, MostUrgentOrderPolicy, NearestDriverPolicy
from environment.config import ROOT

KEYS = ["baseline_1", "baseline_2", "baseline_3", "dqn", "ppo"]
LABELS = {"baseline_1": "Baseline 1", "baseline_2": "Baseline 2", "baseline_3": "Baseline 3",
          "dqn": "DQN", "ppo": "PPO"}
SUBTITLES = {"baseline_1": "Nearest driver", "baseline_2": "Most urgent order", "baseline_3": "Min est. delivery time",
             "dqn": "Deep Q-Network", "ppo": "Proximal Policy Opt."}


def find_checkpoint(algo: str):
    """models/<algo>/final/best.pt if present, else the run with the best validation reward."""
    base = ROOT / "models" / algo
    final = base / "final" / "best.pt"
    if final.exists():
        return final
    best, best_val = None, -float("inf")
    for meta in base.glob("*/best.json"):
        try:
            v = json.loads(meta.read_text())["val_reward"]
        except Exception:
            continue
        if v > best_val and meta.with_suffix(".pt").exists():
            best, best_val = meta.with_suffix(".pt"), v
    return best


def make_policy(key: str):
    if key == "baseline_1":
        return NearestDriverPolicy()
    if key == "baseline_2":
        return MostUrgentOrderPolicy()
    if key == "baseline_3":
        return MinETAPolicy()
    ck = find_checkpoint(key)
    if ck is None:
        raise FileNotFoundError(f"no trained {key} checkpoint under models/{key}/")
    if key == "dqn":
        from rl.dqn.agent import DQNPolicy
        return DQNPolicy(checkpoint=str(ck))
    from rl.ppo.agent import PPOPolicy
    return PPOPolicy(checkpoint=str(ck))


def availability() -> dict:
    out = {}
    for k in KEYS:
        if k.startswith("baseline"):
            out[k] = (True, "")
        else:
            out[k] = (find_checkpoint(k) is not None, "train it first: python -m rl.%s.train" % k)
    return out
