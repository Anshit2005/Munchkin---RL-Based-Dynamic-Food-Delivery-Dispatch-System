# Reinforcement Learning for Dynamic Food Delivery Dispatch

A real-world-inspired food-delivery dispatch problem formulated as an MDP and solved with
**DQN** and **PPO**, compared against three heuristic baselines in the *same* simulator under
identical conditions. Every episode lasts **exactly 3 simulated minutes**.

> Can a learned RL dispatch policy make better sequential assignment decisions than simple
> heuristics when all methods operate in the same 3-minute simulated environment?

Authors: Anshit Agarwal, Namya Dhingra. The design rationale is in [PLAN.md](PLAN.md); results are in
[`results/`](results) and the report in [`report/`](report).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
pytest                            # ~1 minute, 36 tests
```

## Quick start

```bash
python -m ui.game                                                     # game-like GUI
python -m rl.dqn.train --seed 0 --run-name dqn_s0                     # train DQN   (configs/dqn.yaml)
python -m rl.ppo.train --seed 0 --run-name ppo_s0                     # train PPO   (configs/ppo.yaml)
python -m evaluation.select_models                                    # best-val run -> models/<algo>/final
python -m evaluation.comparison --split test --tag final \
    --dqn models/dqn/dqn_s0/best.pt models/dqn/dqn_s1/best.pt models/dqn/dqn_s2/best.pt \
    --ppo models/ppo/ppo_s0/best.pt models/ppo/ppo_s1/best.pt models/ppo/ppo_s2/best.pt
python -m evaluation.visualization --input results/raw/eval_final.csv \
    --dqn-runs models/dqn/dqn_s0 models/dqn/dqn_s1 models/dqn/dqn_s2 \
    --ppo-runs models/ppo/ppo_s0 models/ppo/ppo_s1 models/ppo/ppo_s2
```

The GUI offers Baseline 1/2/3, DQN, PPO and **Run All** (all strategies on the same seed, then a
leaderboard). DQN/PPO buttons are enabled once a trained checkpoint exists under `models/`.

## How the problem is modelled

| Element | Definition |
|---|---|
| World | 10x10 grid, Manhattan distance, L-shaped routes, restaurants/customers/drivers |
| Time | 1 tick = 1 simulated second, **exactly 180 ticks** (`environment/constants.py`) |
| Order lifecycle | Created -> Preparing -> Ready -> Assigned -> Picked up -> Delivered (a driver may be assigned *before* the food is ready and wait at the restaurant) |
| Driver status | `AVAILABLE`, `TO_RESTAURANT`, `AT_RESTAURANT` (waiting for food), `TO_CUSTOMER`; "idle" = available while orders are waiting |
| Observation | fixed 481-d vector: 12 driver slots, 16 order slots, 12x16 estimated-completion pair features, 9 globals |
| Actions | `Discrete(193)`: `assign(driver_i, order_j)` for 12x16 slots, plus `WAIT`; illegal actions are masked |
| Step | assignments do not advance the clock; `WAIT` advances one tick; ticks where only `WAIT` is legal are fast-forwarded |
| Reward | `R = w1*D - w2*W - w3*L - w4*T + w5*B - w6*I`, plus an undelivered-order terminal penalty (`configs/reward.yaml`) |

### Fair comparison
All randomness (restaurants, driver starts, arrivals, customers, prep times, traffic noise) is
pre-sampled from the seed into a `Scenario` trace; the engine consumes no randomness afterwards.
All five strategies therefore see an identical world, and every result row stores a scenario
fingerprint which `evaluation.comparison` asserts to be equal across strategies. Training,
validation and test use disjoint seed pools (`utils/seeding.py`). The 3-minute duration is a
module constant: it is not a field of `EnvConfig` (the loader rejects such a key) and the GUI
only shows it as a fixed label.

### Metrics
Average delivery time (delivered orders), average customer waiting time (all orders; undelivered
orders are censored at the horizon), on-time %, late %, total travel distance, driver utilisation,
successful batching count, discounted and undiscounted episode reward; plus completion rate and
the lifecycle-segment breakdown.

## Strategies
* **Baseline 1 – Nearest Driver:** oldest order -> closest available driver.
* **Baseline 2 – Most Urgent Order:** least slack to deadline first -> nearest feasible driver.
* **Baseline 3 – Minimum Estimated Delivery Time:** greedy argmin of the shared ETA estimator over all feasible pairs.
* **DQN:** Double DQN, masked greedy/epsilon-greedy, replay buffer, target network, Huber loss.
* **PPO:** separate policy and value networks, masked categorical policy, GAE, clipped objective.

Both RL agents use a *pairwise* network that scores each (driver, order) pair with shared weights
(plus a `WAIT` head); a plain MLP is available via `arch: mlp` in the configs.

## Repository layout
```
environment/   MDP + entities (config, city, order, driver, scenario, observation, actions, reward, delivery_env)
simulation/    engine (3-minute horizon), policy protocol, episode runner
baselines/     baseline_1.py, baseline_2.py, baseline_3.py
rl/            common/ (networks, checkpoints, vec env), dqn/, ppo/
evaluation/    metrics, comparison (+paired tests), visualization, model selection
ui/            pygame game: config screen, animated simulation, leaderboard
configs/       env_default, scenarios (spec Table 1), reward, dqn, ppo
tests/         horizon, determinism, lifecycle, MDP/reward/metrics, baselines, RL smoke
models/ results/ report/
```

## Deviations from PLAN.md
* Observations are hand-normalised; there is no running mean/std normaliser.
* Reward lateness is charged per tick on overdue undelivered orders (instead of also charging on delivery), to avoid double-counting.
* The pairwise network only evaluates valid pairs (same result as masking, ~9x faster).
* RL agents were trained without batching; batching is implemented and tested in the environment and baselines.
