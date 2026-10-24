# Dynamic Food-Delivery Dispatch: Complete System Guide

This document explains the implementation in this repository from the ground up. It is
intended to answer two questions:

1. What happens in the game/simulator at every stage of an episode?
2. How do the heuristic, DQN, and PPO policies make decisions from the same information?

The description follows the current source code. The main implementation is in
`environment/`, `simulation/`, `baselines/`, `rl/`, and `evaluation/`.

---

## 1. The problem in one sentence

At each decision point, dispatch available delivery drivers to unassigned food orders so that
more orders are delivered quickly and on time, while avoiding unnecessary travel and idle
drivers.

This is a sequential decision problem rather than a one-shot matching problem:

- new orders arrive during the episode;
- food preparation takes uncertain time;
- drivers move while decisions continue;
- a driver can wait at a restaurant;
- optional batching lets one driver carry two orders from the same restaurant;
- an early assignment changes the driver’s future availability and route.

The project asks whether learned policies can outperform simple dispatch rules when every policy
uses the same simulator, the same information, and the same pre-generated scenarios.

---

## 2. Repository map

| Directory/file | Responsibility |
|---|---|
| `environment/constants.py` | Fixed dimensions, statuses, action encoding, and 180-second horizon |
| `environment/config.py` | YAML loading and validated world/reward dataclasses |
| `environment/scenario.py` | Seeded generation of all exogenous randomness |
| `environment/city.py` | Manhattan distance and deterministic L-shaped movement |
| `environment/order.py` | Order data and lifecycle timestamps |
| `environment/driver.py` | Driver data, route, load, and status |
| `environment/estimates.py` | Observable ETA calculations shared by agents and baselines |
| `environment/observation.py` | 481-number observation vector |
| `environment/actions.py` | Legal-action mask and structured policy view |
| `environment/simulator.py` | Tick-by-tick physical engine |
| `environment/reward.py` | Reward components and terminal penalty |
| `environment/delivery_env.py` | Gymnasium API and decision semantics |
| `simulation/runner.py` | Runs one policy through one episode |
| `baselines/` | Three deterministic heuristic policies |
| `rl/common/` | Networks, vector environments, checkpoints, logging |
| `rl/dqn/` | Double DQN implementation and training loop |
| `rl/ppo/` | PPO implementation and training loop |
| `evaluation/` | Metrics, model selection, comparisons, tables, and plots |
| `ui/` | Pygame configuration, animation, and leaderboard |
| `tests/` | Environment, lifecycle, determinism, reward, baseline, and RL smoke tests |
| `configs/` | Default world, named scenarios, reward, DQN, and PPO parameters |

---

## 3. Basic mathematical model

### 3.1 Grid and distance

The city is a continuous `10 x 10` square. Coordinates are real numbers in `[0, 10]`.
Movement follows an L-shaped route: travel along the x-axis first and then the y-axis.

For points `a = (a_x, a_y)` and `b = (b_x, b_y)`, distance is Manhattan distance:

```text
d(a, b) = |a_x - b_x| + |a_y - b_y|
```

At speed `v`, the nominal travel time is:

```text
time(a, b) = d(a, b) / v
```

The engine moves a driver by at most its current speed budget in each one-second tick.
If the driver reaches a waypoint with distance left in the same tick, the remaining budget
continues to the next waypoint.

The default base speed is `0.13` grid units per second. Traffic changes the actual movement
budget, while policies only observe the nominal traffic category:

```text
light  = 1.2
normal = 1.0
heavy  = 0.7
```

Thus the nominal speed is:

```text
v_nominal = base_speed * traffic_multiplier
```

Actual per-tick speed additionally includes the scenario’s hidden traffic noise.

### 3.2 Markov decision process

The environment is represented as an MDP:

```text
M = (S, A, P, R, gamma)
```

- `S`: simulator state, exposed through a fixed observation vector;
- `A`: 193 discrete actions;
- `P`: deterministic transition for a fixed pre-sampled scenario and chosen action;
- `R`: weighted reward components;
- `gamma`: discount factor, normally `0.99`.

The physical world contains hidden values, such as true preparation time and traffic noise.
Therefore the agent’s practical input is an observation of the state, not the complete internal
state. The scenario is pre-sampled, so the engine itself does not draw new random numbers after
reset.

### 3.3 Fixed horizon

One tick equals one simulated second. Every episode lasts exactly:

```text
EPISODE_DURATION_S = 180 seconds = 3 minutes
N_TICKS = 180
```

The duration is a module constant in `environment/constants.py`. It is deliberately not an
`EnvConfig` field, and YAML rejects keys such as `duration`, `horizon`, or `n_ticks`.
The simulator itself enforces `t >= 180` as the finished condition.

---

## 4. Configuration and named scenarios

`EnvConfig` is immutable and validates restaurant count, driver count, demand, and traffic.
The default configuration is:

| Parameter | Default | Meaning |
|---|---:|---|
| `n_restaurants` | 5 | Restaurants placed in the grid |
| `n_drivers` | 8 | Starting delivery drivers |
| `orders_per_min` | 5.0 | Poisson demand rate |
| `traffic` | `normal` | Nominal traffic category |
| `batching` | false | Whether one driver can carry two same-restaurant orders |
| `base_speed` | 0.13 | Grid units per simulated second |
| `customer_radius` | 4.0 | Maximum customer offset from restaurant |
| `prep_mean`, `prep_std` | 30, 10 | Preparation-time distribution |
| `prep_min`, `prep_max` | 10, 60 | Preparation-time clipping limits |
| `prep_est_noise` | 0.2 | Noise in the observable prep estimate |
| `deadline_slack` | 35.0 | Extra time after estimated travel/preparation |
| `backlog_frac` | 0.5 | Initial backlog rate |
| `traffic_noise` | 0.1 | Hidden AR(1) traffic variation |
| `min_restaurant_spacing` | 2.0 | Initial restaurant separation target |

Named scenarios layer changes over the default:

| Scenario | Restaurants | Drivers | Orders/min |
|---|---:|---:|---:|
| `low_demand` | 5 | 8 | 2 |
| `medium_demand` | 5 | 8 | 5 |
| `high_demand` | 5 | 8 | 8 |
| `driver_shortage` | 5 | 4 | 5 |
| `driver_surplus` | 5 | 12 | 5 |

All five scenarios still use exactly 180 ticks. Changing demand or fleet size does not change
the time available to any algorithm.

---

## 5. Scenario generation and fairness

`generate_scenario(config, seed)` creates a complete immutable `Scenario` before the engine
starts. Its random generator is initialized from `[seed, 12345]`.

### 5.1 Restaurants

Restaurants are repeatedly sampled inside the grid, away from the outer boundary. Rejection
sampling keeps the Manhattan distance between restaurants above the requested minimum. If the
area becomes difficult to pack, the spacing requirement is gradually relaxed.

### 5.2 Driver starts

Each driver receives an independent uniform `(x, y)` starting point in the full grid.

### 5.3 Hidden traffic trace

For each tick, traffic noise follows an AR(1)-style recurrence:

```text
z_t = 0.9 * z_(t-1) + Normal(0, traffic_noise * sqrt(1 - 0.9^2))
traffic_t = clip(level * (1 + z_t), 0.5*level, 1.5*level)
```

The trace has length 180. Agents see only the nominal traffic category in the observation and
ETA estimator; they do not receive `traffic_t`.

### 5.4 Orders

The number of initial backlog orders is:

```text
n_backlog = round(backlog_frac * orders_per_min)
```

Additional orders are drawn from a Poisson distribution:

```text
n_arrivals ~ Poisson(orders_per_min * 180 / 60)
```

Backlog orders have creation time zero. Other creation times are sampled over the horizon and
floored to integer ticks.

For each order:

1. choose a restaurant uniformly;
2. sample a customer offset in a square around that restaurant;
3. clip the customer coordinate to the grid;
4. sample true preparation time from a clipped normal distribution;
5. produce an observable preparation estimate using lognormal noise;
6. calculate the deadline.

The deadline is:

```text
deadline = created_at
         + prep_est
         + d(restaurant, customer) / v_nominal
         + deadline_slack
```

The policy sees `prep_est`, never `prep_actual`.

### 5.5 Fingerprints

The scenario stores a 16-character SHA-256 fingerprint of restaurants, driver starts, complete
order specifications, and traffic. Evaluation asserts that every policy has the same fingerprint
for each `(scenario, seed)` pair. This prevents an accidental fairness failure where policies
are evaluated in different worlds.

Training, validation, and test seeds are disjoint:

```text
training: seeds below 1,000,000
validation: 1,000,000 ... 1,000,199
test: 2,000,000 ... 2,000,099
```

---

## 6. Entities and order lifecycle

### 6.1 Driver states

`DriverStatus` has four values:

1. `AVAILABLE`: no assigned orders and can accept an assignment;
2. `TO_RESTAURANT`: traveling to pick up assigned food;
3. `AT_RESTAURANT`: arrived but waiting for food to become ready;
4. `TO_CUSTOMER`: carrying food toward customer waypoints.

“Idle” is not a fifth stored status. It is computed during reward accounting as an available
driver while at least one order remains pending.

Each driver stores position, route, assigned orders, traveled distance, busy ticks, and capacity.
Capacity is `1` without batching and `2` with batching.

### 6.2 Order states

Order status is derived from timestamps, so there is one source of truth:

```text
PREPARING -> READY -> ASSIGNED -> PICKED_UP -> DELIVERED
```

An order can be assigned while still preparing. In that case, the driver travels to the
restaurant and waits there until all required food is ready.

Important timestamps are:

- `created_at`
- `ready_at`
- `assigned_at`
- `arrived_at_rest`
- `picked_up_at`
- `delivered_at`

The engine keeps useful collections:

- `orders`: all orders spawned so far;
- `pending`: spawned but not assigned;
- `preparing`: food not yet ready;
- `open_orders`: spawned but not delivered.

The conservation invariant is:

```text
created = pending + assigned_open + delivered
```

### 6.3 Batching

When batching is enabled, a driver already traveling to or waiting at a restaurant may receive a
second order only if:

- the second order is unassigned;
- both orders come from the same restaurant;
- the driver has remaining capacity.

The route contains pickup waypoints for both orders followed by the shorter of the two possible
customer-dropoff orders. The batch distance saving is:

```text
saving = separate_distance - shared_route_distance
```

Only positive saving contributes to the batch reward. A batch is counted in final metrics only
when both orders are delivered and its saving is positive.

---

## 7. One episode, step by step

### 7.1 Reset

`DeliveryEnv.reset(seed=seed)`:

1. initializes Gymnasium’s RNG;
2. chooses the configured or sampled environment configuration;
3. generates the complete scenario;
4. creates an `Engine` at `t = 0`;
5. spawns all orders with `created_at <= 0`;
6. initializes reward and decision counters;
7. optionally records an initial frame;
8. optionally fast-forwards ticks where no assignment is possible;
9. builds the first observation and action mask.

The environment does not return an already-finished episode during reset: auto-skip leaves the
last tick for an explicit step.

### 7.2 Choosing an action

At a decision point, a policy receives:

```text
act(observation, info, dispatch_view)
```

The structured `DispatchView` is for heuristics and contains the drivers, visible pending
orders, nominal speed, and legal mask. RL policies primarily use the flat observation and
`info["action_mask"]`.

### 7.3 Assignment action

For driver index `i` and visible order slot `j`:

```text
action = i * 16 + j
```

There are `12 * 16 = 192` possible assignment slots. The final action, index 192, is `WAIT`.

A legal assignment:

1. records `assigned_at` and `driver_id`;
2. removes the order from `pending`;
3. gives an available driver a restaurant pickup waypoint and customer dropoff waypoint;
4. extends a valid batching route if the driver is already heading to/waiting at the same
   restaurant;
5. does not advance time immediately.

Up to 12 assignment actions can be made before the environment forces a time advance. This lets
the policy match several drivers to currently visible orders in one simulated second.

### 7.4 WAIT action

`WAIT` advances exactly one engine tick, unless auto-skip later advances additional ticks with
the same accumulated reward. This is the normal way the policy lets the world evolve.

### 7.5 Invalid actions

An invalid assignment is not silently accepted:

1. the invalid counter increases;
2. `-invalid_penalty` is added;
3. the action is treated as a one-tick wait.

The standard policies always use the legal-action mask, so invalid actions should be zero in
normal evaluation.

### 7.6 Auto-skip

If no assignment is legal, `WAIT` is the only useful action. With `auto_skip=True`, the
environment advances such idle ticks automatically and sums their rewards into the next
transition. It stops as soon as an assignment becomes possible or just before the final tick.

This reduces unnecessary policy calls without changing engine physics. The returned `info["t"]`
therefore may jump by more than one tick after a decision.

### 7.7 Engine tick order

`Engine.advance_tick()` performs the following deterministic sequence:

1. increment `t`;
2. spawn all orders with `created_at <= t`;
3. mark preparation-complete orders as ready;
4. move each non-available driver using the current hidden traffic budget;
5. process pickup and dropoff waypoints;
6. update busy ticks and event counts;
7. finish when `t == 180`.

At a restaurant, a driver can arrive before food is ready and become `AT_RESTAURANT`. When food
becomes ready, pickup occurs on that tick. At a customer waypoint, delivery removes the order
from `open_orders` and may make the driver available.

---

## 8. Observation vector: all 481 values

The Gymnasium observation is a `float32` vector with shape `(481,)`, clipped to `[-2, 2]`.
Unused driver/order slots remain zero. The layout is versioned as `OBS_VERSION = 1`.

```text
12 drivers * 10 features          = 120
16 orders * 10 features            = 160
12 * 16 pair features              = 192
9 global features                  = 9
                                      ---
                                      481
```

### 8.1 Driver block: 10 features each

For driver slot `i`:

1. present flag;
2. x coordinate divided by grid size;
3. y coordinate divided by grid size;
4. available flag;
5. load divided by capacity;
6-9. one-hot status (`AVAILABLE`, `TO_RESTAURANT`, `AT_RESTAURANT`, `TO_CUSTOMER`);
10. estimated seconds until free divided by 180 and capped at 1.

### 8.2 Order block: 10 features each

For visible pending order slot `j`:

1. present flag;
2. restaurant x / grid size;
3. restaurant y / grid size;
4. customer x / grid size;
5. customer y / grid size;
6. age / 180, capped at 1;
7. estimated preparation seconds remaining / 60, capped at 1;
8. ready flag;
9. normalized time to deadline, clipped to `[-1, 1]`;
10. number of visible orders at that restaurant, offset so one order is zero.

The order block contains pending, unassigned orders only. Orders beyond the first 16 are counted
as overflow in the global block and are not individually represented.

### 8.3 Pair block: 192 features

The pair feature at `(driver_i, order_j)` is the estimated completion time in seconds divided
by 90 and clipped to `[0, 2]`.

- `0` means the pair is invalid;
- a valid value is floored at `1e-3`;
- the estimator uses nominal speed and estimated preparation time;
- true preparation and hidden traffic are not used.

This same estimator is used by Baseline 3, observation construction, and the UI.

### 8.4 Global block: 9 features

1. current tick / 180;
2. remaining ticks / 180;
3. pending order count / 16, capped at 1;
4. available drivers / 12;
5. preparing order count / 16, capped at 1;
6. demand rate / 10;
7. traffic multiplier minus 1;
8. pending overflow / 16, capped at 1;
9. assignments already made this tick / 12.

### 8.5 Information boundary

The observation intentionally excludes `prep_actual` and the per-tick traffic noise. This makes
the problem realistic: policies must act using imperfect estimates. Tests explicitly verify that
changing hidden preparation time while keeping the observable estimate fixed does not change the
observation.

---

## 9. Action masking

The action space is always `Discrete(193)`, even when the actual number of drivers or orders is
smaller. A Boolean mask marks legal actions:

- `mask[192]` (`WAIT`) is always true;
- `mask[i * 16 + j]` is true when the engine allows driver `i` to accept order slot `j`;
- all padded or infeasible assignments are false.

Masking is applied consistently:

- heuristics enumerate `DispatchView.valid_pairs()`;
- DQN sets invalid Q-values to negative infinity before argmax;
- PPO sets invalid logits to a large negative number before constructing its categorical
  distribution;
- the policy wrappers fall back to `WAIT` if a selected action is not legal.

This prevents padded slots and impossible assignments from becoming artificial choices.

---

## 10. Reward and its mathematics

Every tick produces a `RewardBreakdown`. Components are kept separate and summed only at the
end:

```text
R_t = delivery_t + wait_t + late_t + travel_t
      + batch_t + idle_t + terminal_t + invalid_t
```

The configured weights are:

```text
w_delivery = 10.0
w_wait     = 2.0
w_late     = 4.0
w_travel   = 0.1
w_batch    = 0.5
w_idle     = 1.0
terminal penalty per undelivered order = 5.0
invalid action penalty = 1.0
```

### 10.1 Tick components

Let:

- `D_t` = deliveries completed on tick `t`;
- `W_t` = created but undelivered orders on tick `t`;
- `L_t` = undelivered orders whose deadline has passed;
- `T_t` = total driver distance traveled on the tick;
- `B_t` = positive distance saved by a batch picked up on the tick;
- `I_t` = available drivers while pending orders exist.

Then:

```text
delivery_t =  w_delivery * D_t
wait_t     = -w_wait     * W_t / 60
late_t     = -w_late     * L_t / 60
travel_t   = -w_travel   * T_t
batch_t    =  w_batch    * B_t
idle_t     = -w_idle     * I_t / 60
```

Waiting and idle penalties are expressed in minutes because one tick is one second.

Lateness is charged per tick while an order is overdue and still undelivered. It is not charged
again at delivery, avoiding double-counting.

### 10.2 Terminal and invalid terms

At `t = 180`, the terminal term is:

```text
terminal = -undelivered_penalty * number_of_undelivered_orders
```

An invalid action contributes `-invalid_penalty` in the transition that treats it as a wait.

### 10.3 Episode return

The undiscounted return is:

```text
G = sum from t=1 to 180 of R_t
```

The environment also records:

```text
G_gamma = sum from t=1 to 180 of gamma^(t-1) * R_t
```

The DQN/PPO learner may additionally multiply rewards by `reward_scale = 0.1` for numerical
optimization. Reported episode rewards remain in the original reward units.

---

## 11. Final metrics

Metrics are computed independently from the reward implementation after the engine finishes.
This keeps evaluation interpretable and prevents the reward formula from being mistaken for the
only measure of performance.

For delivered orders:

```text
delivery_time = delivered_at - created_at
```

For every order, including undelivered orders:

```text
wait_time = min(delivered_at, 180) - created_at
```

The latter is a censored waiting time for orders still open at the horizon.

Reported metrics include:

- average delivery time over delivered orders;
- average customer waiting time over all orders;
- on-time percentage;
- late percentage;
- total driver distance;
- driver utilization;
- positive completed batch count;
- completion rate;
- number undelivered;
- total created and delivered orders;
- lifecycle segment means.

Utilization is:

```text
100 * sum(driver.busy_ticks) / (number_of_drivers * 180)
```

Lifecycle segments are:

1. preparation: `ready_at - created_at`;
2. wait for assignment: `assigned_at - created_at`;
3. travel to restaurant: `arrived_at_rest - assigned_at`;
4. wait at restaurant: `picked_up_at - arrived_at_rest`;
5. travel to customer: `delivered_at - picked_up_at`.

Missing endpoints produce `None` and are excluded from that segment’s mean.

---

## 12. The three baseline policies

All baselines receive the same `DispatchView`, see only observable estimates, and use the same
legal action mask.

### 12.1 Baseline 1: nearest driver

1. enumerate all legal `(driver, order)` pairs;
2. choose the oldest order by `(created_at, id)`;
3. choose the closest feasible driver to that order’s restaurant;
4. break ties by driver id;
5. choose `WAIT` if no pair exists.

This policy prioritizes queue age but does not directly reason about deadlines or customer
dropoff time.

### 12.2 Baseline 2: most urgent order

For each visible order, calculate best estimated completion time:

```text
best_eta(o) = min over feasible drivers d of ETA(d, o)
slack(o) = deadline(o) - current_time - best_eta(o)
```

The policy selects the order with minimum slack, breaking ties by age and id. It then assigns
the nearest feasible driver to that order.

This is urgency-aware but intentionally differs from the minimum-ETA baseline.

### 12.3 Baseline 3: minimum estimated delivery time

Choose the legal pair with the smallest shared ETA estimator:

```text
argmin_(d,o) ETA(d, o)
```

Tie-breaking is by order creation time, driver id, and order id.

This is a greedy local objective: it may deliver a nearby order quickly while ignoring a more
urgent order elsewhere.

---

## 13. Shared ETA estimator

For an available driver, the estimated completion time is:

```text
arrival_to_restaurant = d(driver, restaurant) / nominal_speed
depart_time = max(arrival_to_restaurant, estimated_prep_remaining)
ETA = depart_time + d(restaurant, customer) / nominal_speed
```

For a driver already carrying or traveling toward another order, the estimator models a
same-restaurant batch and tests the shorter customer-dropoff order. It uses the current
driver route context, nominal speed, and estimated preparation times.

The estimator is deliberately not the physics engine. It is a common information-limited
prediction available to every strategy.

---

## 14. Neural policy architecture

The observation has 481 inputs and the action space has 193 outputs. Both RL algorithms support:

- `mlp`: ordinary fully connected network;
- `pairwise`: shared network over driver-order pairs plus a separate `WAIT` head.

### 14.1 Plain MLP

The MLP is:

```text
481 -> hidden -> hidden -> 193
```

with ReLU activations between hidden layers.

### 14.2 Pairwise network

The pairwise model reshapes the observation into driver, order, pair, and global blocks.
It computes pooled driver and order context:

```text
driver_pool = sum(present_driver * driver_features) / max(number_present_drivers, 1)
order_pool  = sum(present_order * order_features) / max(number_present_orders, 1)
context     = [global_features, driver_pool, order_pool]
```

For each valid pair it concatenates:

```text
[driver_features, order_features, pair_ETA, context]
```

and passes the result through a shared MLP producing one score. The same weights are reused for
all driver/order slots, which makes the architecture less tied to a particular slot identity.
A separate context MLP produces the `WAIT` score.

Invalid pairs have zero pair features and are masked by the environment. The implementation
only evaluates nonzero/valid pairs, which is faster than scoring all 192 pairs.

---

## 15. DQN

### 15.1 What DQN learns

DQN learns an action-value function:

```text
Q_theta(s, a) = estimated future discounted return
```

At evaluation time it chooses:

```text
a = argmax over legal actions Q_theta(s, a)
```

### 15.2 Exploration

Training uses masked epsilon-greedy behavior:

```text
with probability epsilon: choose a random legal action
otherwise: choose legal argmax Q
```

Epsilon decreases linearly from `1.0` to `0.05` over 40% of the configured training steps.

### 15.3 Replay and Double DQN update

Transitions are stored as:

```text
(state, action, scaled_reward, next_state, done, next_action_mask)
```

After the replay buffer contains enough data, minibatches are sampled. The online network picks
the best legal next action:

```text
a* = argmax over legal a' Q_theta(next_state, a')
```

The target network evaluates that selected action:

```text
y = r + gamma * (1 - done) * Q_target(next_state, a*)
```

The loss is Huber/smooth-L1 loss:

```text
L = SmoothL1(Q_theta(state, action), y)
```

Gradients are clipped. The target network is hard-synchronized every configured number of steps
unless soft updates are enabled.

### 15.4 Default DQN training

The default configuration uses pairwise hidden size 128, learning rate `0.0005`, replay capacity
60,000, learning start at 3,000 steps, training every 4 environment steps, target sync every
1,000 steps, and 300,000 total steps.

Validation runs periodically on fixed validation seeds. The checkpoint with the highest mean
validation reward is `best.pt`; the final training checkpoint is `last.pt`.

---

## 16. PPO

### 16.1 Policy and value functions

PPO uses two networks:

```text
policy: pi_theta(a | s), producing 193 masked action logits
value:  V_phi(s), producing one scalar
```

The masked logits define a categorical distribution over legal actions only.

### 16.2 Rollouts and GAE

Several environments run synchronously. A rollout stores observations, masks, actions,
log-probabilities, values, rewards, and done flags.

With temporal-difference residual:

```text
delta_t = r_t + gamma * (1 - done_t) * V(s_(t+1)) - V(s_t)
```

Generalized Advantage Estimation is:

```text
Â_t = delta_t + gamma * lambda * (1 - done_t) * Â_(t+1)
```

The return target is:

```text
R_t = Â_t + V(s_t)
```

The default values are `gamma = 0.99` and `lambda = 0.95`.

### 16.3 Clipped PPO objective

Let:

```text
ratio_t = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
```

The clipped policy loss is:

```text
L_policy = -mean(min(ratio_t * Â_t,
                     clip(ratio_t, 1-epsilon, 1+epsilon) * Â_t))
```

The value loss is mean squared error:

```text
L_value = 0.5 * mean((V_phi(s_t) - R_t)^2)
```

Entropy encourages exploration. The combined optimization loss is:

```text
L = L_policy + vf_coef * L_value - entropy_coef * entropy
```

The entropy coefficient decays during training, and updates can stop early when approximate KL
divergence exceeds the configured target.

### 16.4 Default PPO training

The default configuration uses eight environments, 256 steps per rollout, ten epochs per update,
minibatches of 256, policy hidden size 128, value hidden size 256, learning rate `0.0003`, and
1,500,000 total environment steps.

As with DQN, periodic fixed-seed validation selects `best.pt`.

---

## 17. Training, model selection, and evaluation workflow

### 17.1 Install and test

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest
```

### 17.2 Train DQN and PPO

```powershell
python -m rl.dqn.train --seed 0 --run-name dqn_s0
python -m rl.ppo.train --seed 0 --run-name ppo_s0
```

Use `--steps` for a short smoke run and `--arch mlp` to compare against the plain MLP:

```powershell
python -m rl.dqn.train --seed 0 --steps 5000 --arch pairwise
python -m rl.ppo.train --seed 0 --steps 5000 --arch pairwise
```

Each run writes:

- `train_log.csv`;
- `val_log.csv`;
- `best.pt` and `best.json`;
- `last.pt` and `last.json`.

Checkpoint metadata includes algorithm configuration, reward hash, observation version/dimension,
action count, training seed, run name, and (when available) source commit.

### 17.3 Select final models

`evaluation.select_models` scans `models/dqn/dqn_s*/best.json` and
`models/ppo/ppo_s*/best.json`, chooses the highest validation reward for each algorithm, and
copies its best checkpoint to `models/<algorithm>/final/`.

```powershell
python -m evaluation.select_models
```

### 17.4 Compare policies

```powershell
python -m evaluation.comparison --split test --tag final `
  --dqn models/dqn/dqn_s0/best.pt `
  --ppo models/ppo/ppo_s0/best.pt
```

The comparison:

1. loads the three baselines and any supplied checkpoints;
2. runs every policy on every named scenario and test seed;
3. writes raw rows to `results/raw/`;
4. verifies common scenario fingerprints and final tick 180;
5. averages multiple training checkpoints per evaluation seed;
6. calculates means and 95% confidence intervals;
7. runs paired Wilcoxon tests against baselines;
8. writes CSV and LaTeX tables under `results/tables/`.

Pairing matters: policies are compared on the same random world, so differences primarily reflect
decision quality rather than different restaurants, arrivals, or traffic traces.

### 17.5 Visualize

```powershell
python -m evaluation.visualization --input results/raw/eval_final.csv `
  --dqn-runs models/dqn/dqn_s0 models/dqn/dqn_s1 models/dqn/dqn_s2 `
  --ppo-runs models/ppo/ppo_s0 models/ppo/ppo_s1 models/ppo/ppo_s2
```

The UI can also run Baseline 1, Baseline 2, Baseline 3, DQN, PPO, or all strategies on one
seed. “Run All” uses the same scenario for its leaderboard.

---

## 18. Why the comparison is fair

The experimental controls are intentional:

1. **Same horizon:** every episode is 180 ticks.
2. **Same world:** the same scenario fingerprint is used for every policy on a seed.
3. **Same information:** heuristics and RL agents use observable prep estimates and nominal
   traffic, not hidden actual values.
4. **Same legal choices:** all policies use the same action mask and `WAIT` action.
5. **Same simulator:** movement, preparation, pickup, delivery, batching, and accounting are
   performed by one engine.
6. **Disjoint seeds:** training, validation, and test pools do not overlap.
7. **Independent metrics:** final metrics are computed from event timestamps rather than inferred
   from the reward alone.
8. **Checkpoint compatibility:** observation version, dimension, and action count are checked
   when loading a model.

The environment may auto-skip ticks with no legal assignments, but it sums the physical rewards
and does not skip possible decisions.

---

## 19. Tests and invariants

The test suite checks more than that the code runs:

- observation size is exactly 481 and action count exactly 193;
- Gymnasium’s environment checker accepts the environment;
- observations are finite, bounded, and correctly typed;
- `WAIT` is always legal;
- random valid policies never produce invalid actions;
- invalid actions are penalized and advance time;
- hidden preparation time does not leak into observations;
- each reward component matches its hand-calculated formula;
- terminal penalty equals the number of undelivered orders times its weight;
- total episode reward equals the sum of recorded components;
- order timestamps are monotone and food is never picked up before ready;
- drivers never exceed capacity;
- order conservation holds at every step;
- exact single-order movement timing matches Manhattan physics;
- drivers correctly wait at restaurants;
- batching saves distance and is counted correctly;
- deterministic scenario generation and episode behavior remain reproducible;
- all episodes finish at exactly tick 180.

These tests define important behavioral contracts for future changes. If the observation layout,
action count, reward semantics, or horizon changes intentionally, checkpoint compatibility and
the evaluation protocol must be updated together.

---

## 20. A concrete miniature episode

Consider one driver at `(0, 0)`, one restaurant at `(2, 0)`, one customer at `(2, 2)`,
speed `0.13`, and preparation time 10 seconds.

1. At reset, the order is spawned and is `PREPARING`.
2. A policy assigns it. The assignment changes no time; the driver’s route becomes:
   `(2, 0, PICKUP) -> (2, 2, DROPOFF)`.
3. The next ticks move the driver along x. The restaurant takes about
   `2 / 0.13 = 15.38` seconds to reach.
4. The food becomes ready at tick 10, before the driver arrives.
5. At approximately tick 16, the driver reaches the restaurant and picks up the order.
6. It then travels two more grid units to the customer, taking about
   `2 / 0.13 = 15.38` seconds.
7. Delivery occurs around tick 31, the order is removed from open orders, and the driver becomes
   available.
8. The episode continues until tick 180 so later arrivals and reward accounting are still
   processed.

If the driver arrived before tick 10, the status would instead become `AT_RESTAURANT`, and the
driver would wait there until the preparation-completion tick.

---

## 21. Important implementation notes

- The episode duration is fixed; do not add a duration field to `EnvConfig`.
- The visible order capacity is 16. Extra pending orders are represented only by overflow count.
- Assignment actions do not advance time; `WAIT` and forced assignment-cap transitions do.
- A driver may be assigned before food is ready.
- Lateness is charged while overdue and undelivered, not again at delivery.
- Reward components are reported separately.
- RL training uses scaled rewards internally, while reported returns use original units.
- The pairwise network evaluates valid pairs only, but the mathematical action space remains 193.
- Current RL training is unbatched by default; batching is implemented and tested when enabled.
- A checkpoint is not portable across observation/action layout changes unless
  `OBS_VERSION` and compatibility metadata are deliberately updated.

---

## 22. End-to-end mental model

The complete data flow is:

```text
YAML config + seed
        |
        v
generate_scenario()
  restaurants, driver starts, orders, hidden traffic
        |
        v
Engine.reset()
  entities, routes, pending/preparing/open collections
        |
        v
DeliveryEnv._refresh()
  visible order slots + legal mask + 481-number observation
        |
        v
policy.act()
  baseline rule or masked neural action
        |
        v
DeliveryEnv.step()
  assign immediately OR advance one tick
        |
        v
Engine.advance_tick()
  spawn -> prepare -> move -> pickup/dropoff -> accounting
        |
        v
RewardBreakdown + next observation
        |
        v
repeat until t = 180
        |
        v
compute_metrics()
  lifecycle, delivery, waiting, lateness, distance, utilization, batching
```

The central design principle is that all strategies are decision-makers inside the same
deterministic, observable, fixed-horizon dispatch world. The baselines provide interpretable
reference behavior; DQN learns action values from replay; PPO learns a masked stochastic policy
and a value estimate; evaluation then compares their outcomes on paired, unseen scenarios.
