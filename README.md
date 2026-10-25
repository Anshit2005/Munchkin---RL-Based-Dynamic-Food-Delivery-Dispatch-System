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

When the game opens, enter both player names. Each player then chooses a strategy. Set the shared restaurant count, driver count, and order arrival rate in the controls below the two maps. Press **Start match** to begin, or use **Step one game minute** to advance the active player's simulation.

## Game rules and assumptions

- Each player receives a separate 60-simulated-minute run. Player 1 runs first, then Player 2; automatic play advances one game minute every real second, so a full match takes about two minutes. The active player's clock and map update while the other player's finished or waiting state remains visible. Pausing stops the active run.
- Each player gets a separate copy of the same scenario: same map, restaurant and driver positions, order arrivals, preparation times, and random seed. Only the dispatch strategy differs.
- Orders arrive at the start of each simulated minute at the shared configured rate. Their restaurant, customer location, and preparation time are generated from the match seed.
- Food preparation takes 0 or 1 game minute in this prototype. The travel rate is a fixed internal setting shared by both players.
- A ready order can be assigned to one available driver. The driver travels to the restaurant, picks up the order, and continues to the customer. Travel speed is fixed inside the simulation so neither player can tune it to gain an advantage.
- Each completed delivery earns 10 points. At the 60-minute cutoff, orders still preparing, waiting for a driver, or in transit remain pending and earn no delivery points.
- The player with more points wins. If points tie, the player with the lower average delivery time wins; if that also ties, the shorter total travel distance wins. If all three measures tie, the match is a draw.
- The decision feed shows each assignment and the policy's reason. The side panels show each player's score, delivered and pending orders, average delivery time, distance travelled, and available drivers.

The one-hour head-to-head match is the interactive game mode. The assignment's formal strategy comparison remains a separate experiment: the command-line comparison uses the default three-simulated-minute horizon, shared seeds, and the same conditions for all baselines.

## Compare the baselines

Run matched episodes for all three strategies from the project root:

```bash
python3 -m food_delivery_game.compare --episodes 20
```

Every strategy receives the same seed for each three-minute comparison episode. You can also pass `--seed`, `--drivers`, `--restaurants`, or `--orders-per-minute` to change the experiment setup.

## Train DQN and PPO

Both agents use the same simulator as the baselines. At each decision, the agent may assign one
ready order to an available delivery partner or choose **wait**, which advances the simulated
minute. Assignments earn no immediate reward; each completed delivery earns 10 points. Training
randomizes the number of restaurants, delivery partners, and order rate across episodes. The
observation contains normalized driver state, ready-order details, pairwise estimated travel time,
and shared scenario settings. Invalid assignments are masked from both algorithms.

Train either agent from the project root:

```bash
python3 -m food_delivery_game.rl.train --algorithm dqn --episodes 500
python3 -m food_delivery_game.rl.train --algorithm ppo --episodes 500
```

Checkpoints are saved as `models/dqn.pt` and `models/ppo.pt`; episode logs are saved alongside
them. To compare the baselines and any trained RL models on matched seeds:

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
