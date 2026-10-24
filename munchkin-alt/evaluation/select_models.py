"""Copy the best-validation run of each algorithm to models/<algo>/final/ (used by the UI).

    python -m evaluation.select_models
"""
from __future__ import annotations

import json
import shutil

from environment.config import ROOT


def main():
    for algo in ("dqn", "ppo"):
        best, best_val = None, -float("inf")
        for meta in (ROOT / "models" / algo).glob(f"{algo}_s*/best.json"):
            v = json.loads(meta.read_text())["val_reward"]
            print(f"{algo}: {meta.parent.name} val_reward={v:.2f}")
            if v > best_val:
                best, best_val = meta.parent, v
        if best is None:
            print(f"{algo}: no runs found")
            continue
        dst = ROOT / "models" / algo / "final"
        dst.mkdir(exist_ok=True)
        for f in ("best.pt", "best.json"):
            shutil.copy(best / f, dst / f)
        print(f"{algo}: selected {best.name} -> {dst}")


if __name__ == "__main__":
    main()
