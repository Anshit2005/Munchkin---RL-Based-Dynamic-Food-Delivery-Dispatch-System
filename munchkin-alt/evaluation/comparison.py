"""Run every strategy on identical scenarios and seeds, aggregate, and test significance.

    python -m evaluation.comparison --split test --dqn models/dqn/dqn_s0/best.pt \
        --ppo models/ppo/ppo_s0/best.pt --n-seeds 100 --tag final

Several --dqn / --ppo checkpoints (different training seeds) are allowed; they are averaged per
evaluation seed before comparison so the paired tests stay valid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from baselines import MinETAPolicy, MostUrgentOrderPolicy, NearestDriverPolicy
from environment.config import ROOT, load_reward_config, load_scenarios
from simulation.runner import run_episode
from utils.seeding import split_seeds

METRIC_COLS = ["reward", "reward_discounted", "avg_delivery_time", "avg_wait_time", "on_time_pct",
               "late_pct", "total_distance", "utilization", "batch_count", "completion_rate",
               "undelivered", "n_delivered"]
POLICY_ORDER = ["baseline_1", "baseline_2", "baseline_3", "dqn", "ppo"]
POLICY_LABELS = {"baseline_1": "Baseline 1\n(Nearest)", "baseline_2": "Baseline 2\n(Urgent)",
                 "baseline_3": "Baseline 3\n(Min ETA)", "dqn": "DQN", "ppo": "PPO"}


def build_entries(dqn_ckpts=(), ppo_ckpts=()):
    """-> list of (policy, train_seed). Baselines have train_seed = -1."""
    entries = [(NearestDriverPolicy(), -1), (MostUrgentOrderPolicy(), -1), (MinETAPolicy(), -1)]
    if dqn_ckpts:
        from rl.dqn.agent import DQNPolicy
        entries += [(DQNPolicy(checkpoint=str(c)), i) for i, c in enumerate(dqn_ckpts)]
    if ppo_ckpts:
        from rl.ppo.agent import PPOPolicy
        entries += [(PPOPolicy(checkpoint=str(c)), i) for i, c in enumerate(ppo_ckpts)]
    return entries


def run_matrix(entries, scenarios: dict, seeds: list, reward_config=None, progress: bool = True) -> pd.DataFrame:
    rows = []
    jobs = [(e, cfg, s) for cfg in scenarios.values() for s in seeds for e in entries]
    it = tqdm(jobs, desc="episodes") if progress else jobs
    for (policy, train_seed), cfg, s in it:
        r = run_episode(policy, cfg, s, reward_config).row()
        r["train_seed"] = train_seed
        rows.append(r)
    df = pd.DataFrame(rows)
    # every policy must have faced the exact same world for each (scenario, seed)
    assert (df.groupby(["scenario", "seed"])["fingerprint"].nunique() == 1).all(), "scenarios differ across policies!"
    assert (df["final_tick"] == 180).all()
    return df


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Average over training seeds so that each (policy, scenario, eval seed) is one row."""
    cols = [c for c in METRIC_COLS if c in df.columns]
    return df.groupby(["policy", "scenario", "seed"], as_index=False)[cols].mean()


def ci95(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ps = per_seed(df)
    rows = []
    for (pol, scn), g in ps.groupby(["policy", "scenario"]):
        row = {"policy": pol, "scenario": scn, "n_seeds": len(g)}
        for m in METRIC_COLS:
            if m in g:
                row[m] = g[m].mean()
                row[m + "_ci95"] = ci95(g[m])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["policy"] = pd.Categorical(out["policy"], POLICY_ORDER, ordered=True)
    return out.sort_values(["scenario", "policy"]).reset_index(drop=True)


def paired_tests(df: pd.DataFrame, rl=("dqn", "ppo"), ref=("baseline_1", "baseline_2", "baseline_3"),
                 metrics=("reward", "avg_delivery_time", "avg_wait_time", "on_time_pct", "completion_rate")) -> pd.DataFrame:
    ps = per_seed(df)
    rows = []
    for scn, g in ps.groupby("scenario"):
        wide = {m: g.pivot(index="seed", columns="policy", values=m) for m in metrics}
        for a in rl:
            for b in ref:
                for m in metrics:
                    w = wide[m]
                    if a not in w or b not in w:
                        continue
                    x, y = w[a].values, w[b].values
                    ok = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[ok], y[ok]
                    diff = x - y
                    try:
                        p = stats.wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0
                    except ValueError:
                        p = float("nan")
                    better_low = m in ("avg_delivery_time", "avg_wait_time")
                    wins = float(np.mean(diff < 0) if better_low else np.mean(diff > 0))
                    rows.append({"scenario": scn, "metric": m, "rl": a, "reference": b,
                                 "mean_diff": float(diff.mean()), "win_rate": wins, "p_value": float(p),
                                 "n": int(ok.sum())})
    return pd.DataFrame(rows)


def write_tables(summary: pd.DataFrame, tag: str, out_dir: Path = ROOT / "results" / "tables"):
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["policy", "n_delivered", "completion_rate", "avg_delivery_time", "avg_wait_time", "on_time_pct",
            "total_distance", "utilization", "reward"]
    for scn, g in summary.groupby("scenario"):
        t = g[keep].copy()
        for c in keep[1:]:
            ci = g[c + "_ci95"].values
            t[c] = [f"{v:.1f} ± {e:.1f}" if not np.isnan(e) else f"{v:.1f}" for v, e in zip(g[c], ci)]
        t.to_csv(out_dir / f"summary_{tag}_{scn}.csv", index=False)
        with open(out_dir / f"summary_{tag}_{scn}.tex", "w", encoding="utf-8") as f:
            f.write(t.to_latex(index=False, escape=True))
    summary.to_csv(out_dir / f"summary_{tag}_all.csv", index=False)


def leaderboard(summary: pd.DataFrame, metric: str = "reward") -> pd.DataFrame:
    """Mean metric across scenarios and mean rank (1 = best) per policy."""
    low_better = metric in ("avg_delivery_time", "avg_wait_time", "late_pct", "total_distance", "undelivered")
    piv = summary.pivot(index="scenario", columns="policy", values=metric)
    ranks = piv.rank(axis=1, ascending=low_better)
    return pd.DataFrame({"mean_" + metric: piv.mean(), "mean_rank": ranks.mean()}).sort_values("mean_rank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--dqn", nargs="*", default=[])
    ap.add_argument("--ppo", nargs="*", default=[])
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.split
    scen = load_scenarios()
    entries = build_entries(args.dqn, args.ppo)
    df = run_matrix(entries, scen, split_seeds(args.split, args.n_seeds), load_reward_config())
    raw = ROOT / "results" / "raw" / f"eval_{tag}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)
    summary = summarize(df)
    write_tables(summary, tag)
    tests = paired_tests(df)
    tests.to_csv(ROOT / "results" / "tables" / f"paired_tests_{tag}.csv", index=False)
    print(summary[["scenario", "policy", "avg_delivery_time", "on_time_pct", "completion_rate", "reward"]]
          .to_string(index=False))
    print("\nLeaderboard (reward):\n", leaderboard(summary))
    print("saved", raw)


if __name__ == "__main__":
    main()
