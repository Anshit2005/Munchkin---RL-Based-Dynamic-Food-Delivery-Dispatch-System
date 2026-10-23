# Munchkin - RL Based Dynamic Food Delivery Dispatch System

Munchkin is a head-to-head food-delivery dispatch game and a simulator for comparing dispatch strategies. Two players choose policies that run against matching copies of the same simulated city. The current strategies are three heuristic baselines; PPO will be added in a later commit.

## Requirements

- Python 3.10 or newer
- Tkinter (usually included with Python; Linux distributions may package it separately)

No third-party Python packages are required for the simulator or interface.

## Run

From the project root:

```bash
python3 -m food_delivery_game
```

When the game opens, enter both player names. Each player then chooses a strategy. Set the shared restaurant count, driver count, and order arrival rate in the controls below the two maps. Press **Start match** to begin, or use **Step one game minute** to advance both players together.

## Game rules and assumptions

- A match lasts 60 simulated minutes. Automatic play advances one game minute every real second, so a complete match takes about one minute. Pausing stops both simulations together.
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
  ui.py             Map, controls, and live status display
```

## Current scope

The environment supports order arrivals, preparation, driver dispatch, travel, pickups, and deliveries. The three baselines are nearest driver, oldest/most urgent ready order, and minimum estimated delivery time. The comparison runner reports average deliveries, reward points, delivery time, and travel distance over matched episode seeds. The GUI pits two selected baselines against one another. PPO is planned for a later commit.

The current simulator uses a lightweight 2D coordinate map and discrete simulated minutes. Orders arrive at the start of each minute; deliveries can be completed during that minute using the driver's remaining travel distance after pickup. Orders unfinished at the match cutoff remain pending and are counted in the final statistics. The simulator is an educational model, not a reconstruction of any commercial delivery platform.
