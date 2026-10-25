"""Lightweight model registry that does not import PyTorch until a model is selected."""

import json

from ..config import PROJECT_ROOT
from .version import OBS_VERSION

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILES = {"DQN": MODEL_DIR / "dqn.pt", "PPO": MODEL_DIR / "ppo.pt"}


def available_policies() -> list[str]:
    available = []
    for name, path in MODEL_FILES.items():
        metadata_path = path.with_name(f"{path.stem}.meta.json")
        if not path.is_file() or not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("observation_version") == OBS_VERSION:
            available.append(name)
    return available
