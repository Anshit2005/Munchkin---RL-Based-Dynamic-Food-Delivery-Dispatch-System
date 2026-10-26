# Munchkin - RL Based Dynamic Food Delivery Dispatch System

Munchkin is a head-to-head food-delivery dispatch game and a simulator for comparing dispatch strategies. Two players choose policies that run against matching copies of the same simulated city. Available strategies include three heuristic baselines and trained DQN/PPO policies.

## Requirements

- Python 3.10 or newer
- Tkinter (usually included with Python; Linux distributions may package it separately)

The simulator and interface use Python's standard library. DQN and PPO training require NumPy and PyTorch.

Install dependencies in a virtual environment. Run the activation command for your shell from the project folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run

From the project root:

```bash
python3 -m food_delivery_game
```

When the game opens, enter both player names. Each player then chooses a strategy (heuristic baselines or trained DQN/PPO). Set the shared restaurant count, driver count, arrival rate, and traffic condition in the controls below the maps. Use the **Speed controls** (`1×`, `2×`, `5×`, `10×`) to speed up or slow down the match at any time. Press **Start match** to begin, or use **Step one game minute** to step through decisions manually. A **Map Legend** above the active clock clarifies all symbols for restaurants, cooking/ready food, delivered orders, and drivers.

## Game rules and assumptions

- Each player receives a separate 60-simulated-minute run. Player 1 runs first, then Player 2; automatic play advances one game minute every real second, so a full match takes about two minutes. The active player's clock and map update while the other player's finished or waiting state remains visible. Pausing stops the active run.
- Both runs use the same pre-sampled scenario: map, restaurant and driver positions, order arrivals, customer locations, actual and estimated preparation times, deadlines, and minute-by-minute traffic. Only the dispatch policy differs. Scenario randomness is generated before play, so policy decisions cannot change later random events.
- Orders arrive according to a Poisson process at the configured average orders per simulated minute. The opening state also includes an initial backlog equal to half the configured arrival rate (rounded to an integer). Restaurant, customer, preparation, and traffic samples are seeded and repeatable.
- Restaurants are placed at least 12 map units apart (spacing relaxes only if the map is crowded). Each customer's location is sampled within 28 map units of the order's restaurant and clipped to the map boundary, following the alternate project's local service-area model.
- Actual food preparation time is sampled from a normal distribution (mean 0.5, standard deviation 0.17 game minutes), clipped to 0.15–1.0 minutes. The dispatcher sees a noisy preparation estimate, not the actual ready time. Orders may be assigned while preparing; a driver who arrives early waits at the restaurant.
- Deadlines are estimated from preparation estimate + restaurant-to-customer travel time at nominal traffic + five minutes of slack. Pickup time and realized traffic still affect actual delivery time and whether it is late. All policies see the traffic level's nominal speed; the simulation applies correlated minute-to-minute traffic noise around that level (clipped between 0.5× and 1.5× nominal).
- Traffic levels scale the base travel speed: light = 1.2×, normal = 1.0×, and heavy = 0.7×. Restaurant count, driver count, and average order rate are shared between the two players.
- Each completed delivery earns 10 game points. At the 60-minute cutoff, orders still preparing, waiting for a driver, or in transit remain pending. The player with more game points wins; ties are broken by lower average delivery time, then shorter total travel distance. If all three measures tie, the match is a draw.
- RL training and evaluation also report a separate shaped reward adapted from `munchkin-alt`: +10 per delivery, −2 per pending order-minute, −4 per overdue order-minute, −0.1 per travel-distance unit, −1 per idle driver-minute when unassigned orders are available, and −5 per unfinished order at the episode cutoff. The game winner is still determined by game points and the tie-breakers, not this RL reward. Batch-savings reward is currently zero because the main simulator does not support multi-order batching; invalid actions are masked before selection.
- The decision feed shows each assignment and the policy's reason. The side panels show each player's score, delivered and pending orders, average delivery time, distance travelled, and available drivers.

The one-hour head-to-head match is the interactive game mode. Command-line strategy comparisons use matched seeds and identical scenario settings for every policy; their horizon can be set with `--minutes`.

## Compare the baselines

Run matched episodes for all three strategies from the project root:

```bash
python3 -m food_delivery_game.compare --episodes 20
```

Every strategy receives the same seed for each comparison episode. You can also pass `--seed`, `--drivers`, `--restaurants`, `--orders-per-minute`, `--traffic`, or `--minutes` to change the experiment setup.

## Train DQN and PPO

Both agents use the same simulator as the baselines. At each decision, an agent may assign one
preparing or ready order to an available delivery partner or choose **wait**, which advances the
simulated minute. Assignments earn no immediate reward; deliveries earn game points and the
separate shaped RL reward described above. Training randomizes restaurant count, driver count,
arrival rate, and traffic level. The observation contains normalized driver state (including
restaurant waiting and estimated time until free), order preparation/deadline estimates, pairwise
estimated delivery time, and shared scenario settings. Invalid assignments are masked.

The repository includes the trained DQN and PPO checkpoints (`models/*.pt`) and their metadata,
so evaluators can run comparisons immediately after cloning. DQN begins learning once its replay
buffer reaches 32 transitions and refuses to save a checkpoint if no gradient update occurred.
Both learned policies use deterministic masked action selection during evaluation and in the GUI.

Train either agent from the project root. Validation runs every 25 episodes on fixed
low-demand/light-traffic, medium-demand/normal-traffic, and heavy-demand/heavy-traffic scenarios,
then keeps the checkpoint with the best mean shaped RL reward rather than blindly saving the last
episode. Scenario and reward changes, plus the updated observation layout, make older checkpoints
incompatible; retrain both agents before comparing:

```bash
python3 -m food_delivery_game.rl.train --algorithm dqn --episodes 500
python3 -m food_delivery_game.rl.train --algorithm ppo --episodes 500
```

Checkpoints are saved as `models/dqn.pt` and `models/ppo.pt`; episode logs are saved alongside
them, with metadata describing the observation version and validation score. The shared scenario
report prints delivery points and shaped RL reward separately. To compare the baselines and any
trained RL models on matched seeds:

```bash
python3 -m food_delivery_game.rl.evaluate --episodes 20 --minutes 60
```

Print a terminal report for all three baselines plus DQN and PPO across low, medium, and heavy
demand (five matched episodes per scenario by default):

```bash
python3 -m food_delivery_game.rl.scenario_report
```

Change the episode count or shared setup with options such as
`--episodes 10 --drivers 8 --restaurants 5`.

After training, restart the game. A DQN or PPO strategy appears in the player selector when its
checkpoint exists. DQN selects its highest-value legal action; PPO samples from its learned
categorical policy, which is the action-selection method used during PPO training.

## Project layout

```text
food_delivery_game/
  baselines/        Three heuristic dispatch policies
  __main__.py       Tkinter entry point
  compare.py        Command-line baseline comparison
  config.py         Simulation configuration
  entities.py       Restaurant, driver, and order data models
  environment.py    Discrete-time delivery simulation
  simulation/       Matched episode runner and metrics
  rl/               DQN/PPO agents, training, and matched-seed evaluation
  ui.py             Map, controls, and live status display
```

## Current scope

The environment supports order arrivals, preparation, driver dispatch, travel, pickups, and deliveries. The three baselines are nearest driver, oldest/most urgent ready order, and minimum estimated delivery time. DQN and PPO use masked pairwise neural policies trained with replay learning and clipped policy optimization, respectively. The comparison runners report deliveries, reward points, delivery time, and travel distance over matched episode seeds. The GUI pits two selected strategies against one another.

The current simulator uses a lightweight 2D coordinate map and discrete simulated minutes. Orders arrive at the start of each minute; deliveries can be completed during that minute using the driver's remaining travel distance after pickup. Orders unfinished at the match cutoff remain pending and are counted in the final statistics. The simulator is an educational model, not a reconstruction of any commercial delivery platform.
