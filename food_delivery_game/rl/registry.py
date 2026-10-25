"""Lightweight model registry that does not import PyTorch until a model is selected."""

from ..config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILES = {"DQN": MODEL_DIR / "dqn.pt", "PPO": MODEL_DIR / "ppo.pt"}


def available_policies() -> list[str]:
    return [name for name, path in MODEL_FILES.items() if path.is_file()]
