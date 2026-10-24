"""Checkpoints with compatibility metadata (obs layout, action space, reward hash...)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from environment.constants import N_ACTIONS, OBS_DIM, OBS_VERSION


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def save_checkpoint(path, state: dict, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "obs_version": OBS_VERSION, "obs_dim": OBS_DIM, "n_actions": N_ACTIONS,
            "git_commit": _git_commit()}
    torch.save({"state": state, "meta": meta}, path)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    if m["obs_version"] != OBS_VERSION or m["obs_dim"] != OBS_DIM or m["n_actions"] != N_ACTIONS:
        raise RuntimeError(f"checkpoint {path} is incompatible with the current environment "
                           f"(obs_version {m['obs_version']} vs {OBS_VERSION})")
    return ck
