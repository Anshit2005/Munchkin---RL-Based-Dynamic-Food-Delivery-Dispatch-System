"""All plots from spec section 27.

    python -m evaluation.visualization --input results/raw/eval_final.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment.config import ROOT
from evaluation.comparison import POLICY_LABELS, POLICY_ORDER, ci95, per_seed

PLOTS = ROOT / "results" / "plots"
COLORS = {"baseline_1": "#9aa5b1", "baseline_2": "#7b8794", "baseline_3": "#52606d",
          "dqn": "#2f80ed", "ppo": "#eb5757"}
SCN_LABEL = {"low_demand": "Low demand", "medium_demand": "Medium demand", "high_demand": "High demand",
             "driver_shortage": "Driver shortage", "driver_surplus": "Driver surplus"}


def _save(fig, name: str):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.png", dpi=150)
    fig.savefig(PLOTS / f"{name}.pdf")
    plt.close(fig)


def _policies(ps):
    return [p for p in POLICY_ORDER if p in set(ps["policy"])]


def bar_metric(ps, metric, ylabel, name, scenario="medium_demand"):
    g = ps[ps["scenario"] == scenario]
    pols = _policies(g)
    means = [g[g.policy == p][metric].mean() for p in pols]
    errs = [ci95(g[g.policy == p][metric]) for p in pols]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(pols)), means, yerr=errs, capsize=4, color=[COLORS[p] for p in pols])
    ax.set_xticks(range(len(pols)), [POLICY_LABELS[p] for p in pols])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs algorithm ({SCN_LABEL.get(scenario, scenario)}, 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, name)


def grouped_scenarios(ps, scenarios, metrics, name, suptitle):
    pols = _policies(ps)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    width = 0.8 / len(pols)
    for ax, (metric, label) in zip(axes, metrics):
        for i, p in enumerate(pols):
            vals = [ps[(ps.policy == p) & (ps.scenario == s)][metric].mean() for s in scenarios]
            errs = [ci95(ps[(ps.policy == p) & (ps.scenario == s)][metric]) for s in scenarios]
            ax.bar(np.arange(len(scenarios)) + i * width, vals, width, yerr=errs, capsize=2,
                   color=COLORS[p], label=POLICY_LABELS[p].replace("\n", " "))
        ax.set_xticks(np.arange(len(scenarios)) + width * (len(pols) - 1) / 2,
                      [SCN_LABEL[s] for s in scenarios])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(suptitle)
    _save(fig, name)


def training_curve(run_dirs, algo, name, window=50):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    curves = []
    for rd in run_dirs:
        p = Path(rd) / "train_log.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        y = d["return"] if "return" in d else d["mean_return"]
        ys = y.rolling(window if algo == "dqn" else max(3, window // 10), min_periods=1).mean()
        ax.plot(d["step"], ys, alpha=0.35, lw=1)
        curves.append((d["step"].values, ys.values))
    if curves:
        grid = np.linspace(max(c[0][0] for c in curves), min(c[0][-1] for c in curves), 100)
        mean = np.mean([np.interp(grid, s, y) for s, y in curves], axis=0)
        ax.plot(grid, mean, color=COLORS[algo], lw=2.2, label=f"{algo.upper()} (mean of {len(curves)} runs)")
        ax.legend()
    ax.set_xlabel("environment steps")
    ax.set_ylabel("training episode reward (random scenarios)")
    ax.set_title(f"{algo.upper()} training reward curve")
    ax.grid(alpha=0.3)
    _save(fig, name)


def reward_components(ps_raw, name, scenario="medium_demand"):
    g = ps_raw[ps_raw.scenario == scenario]
    comps = ["rw_delivery", "rw_wait", "rw_late", "rw_travel", "rw_idle", "rw_terminal", "rw_batch"]
    pols = _policies(g)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pos = np.zeros(len(pols)); neg = np.zeros(len(pols))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(comps):
        if c not in g:
            continue
        v = np.array([g[g.policy == p][c].mean() for p in pols])
        bottom = np.where(v >= 0, pos, neg)
        ax.bar(range(len(pols)), v, bottom=bottom, color=cmap(i), label=c[3:])
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.set_xticks(range(len(pols)), [POLICY_LABELS[p] for p in pols])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean reward contribution")
    ax.set_title(f"Reward components ({SCN_LABEL.get(scenario, scenario)})")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, name)


def make_all(raw_csv: str, dqn_runs=(), ppo_runs=()):
    raw = pd.read_csv(raw_csv)
    ps = per_seed(raw)
    scn = "medium_demand"
    bar_metric(ps, "avg_delivery_time", "Avg delivery time (s)", "01_avg_delivery_time", scn)
    bar_metric(ps, "avg_wait_time", "Avg customer waiting time (s)", "02_avg_waiting_time", scn)
    bar_metric(ps, "on_time_pct", "On-time delivery (%)", "03_on_time_pct", scn)
    bar_metric(ps, "total_distance", "Total travel distance (units)", "04_total_distance", scn)
    bar_metric(ps, "utilization", "Driver utilization (%)", "05_driver_utilization", scn)
    bar_metric(ps, "reward", "Episode reward", "06_episode_reward", scn)
    grouped_scenarios(ps, ["low_demand", "medium_demand", "high_demand"],
                      [("on_time_pct", "On-time (%)"), ("avg_delivery_time", "Avg delivery time (s)"),
                       ("reward", "Episode reward")], "07_demand_levels", "Performance under low / medium / high demand")
    grouped_scenarios(ps, ["driver_shortage", "medium_demand", "driver_surplus"],
                      [("on_time_pct", "On-time (%)"), ("avg_delivery_time", "Avg delivery time (s)"),
                       ("reward", "Episode reward")], "08_driver_supply", "Performance under driver shortage / baseline / surplus")
    training_curve(dqn_runs, "dqn", "09_dqn_training_curve")
    training_curve(ppo_runs, "ppo", "10_ppo_training_curve")
    reward_components(raw, "11_reward_components", scn)
    grouped_scenarios(ps, list(SCN_LABEL), [("completion_rate", "Completion rate (%)")],
                      "12_completion_rate", "Completion rate by scenario")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--dqn-runs", nargs="*", default=[])
    ap.add_argument("--ppo-runs", nargs="*", default=[])
    a = ap.parse_args()
    make_all(a.input, a.dqn_runs, a.ppo_runs)
    print("plots written to", PLOTS)


if __name__ == "__main__":
    main()
