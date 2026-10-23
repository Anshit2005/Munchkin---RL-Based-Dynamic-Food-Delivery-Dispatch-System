# Munchkin - RL Based Dynamic Food Delivery Dispatch System

Munchkin is a configurable food-delivery dispatch simulator for studying sequential dispatch decisions with reinforcement learning. The simulator currently includes three heuristic dispatch baselines and a Tkinter game interface. PPO will be added in a later commit.

## Requirements

- Python 3.10 or newer
- Tkinter (usually included with Python; Linux distributions may package it separately)

No third-party Python packages are required for the simulator or interface.

## Run

From the project root:

```bash
python3 -m food_delivery_game
```

Use **Start / Pause** to advance the simulation and **Reset** to start a new scenario. Configure restaurants, drivers, and order arrival rate before starting. Episodes run for a fixed three simulated minutes; automatic play advances one simulated minute every five seconds, while **Step one minute** advances immediately. The live dashboard tracks elapsed simulation time, deliveries, average delivery time, travel distance, free drivers, and reward points. Each completed delivery earns 10 points; the episode-end summary shows the final score and results. Delivery and episode sounds use the system bell and can be switched off in the controls.

## Compare the baselines

Run matched episodes for all three strategies from the project root:

```bash
python3 -m food_delivery_game.compare --episodes 20
```

Every strategy receives the same seed for each comparison episode. You can also pass `--seed`, `--drivers`, `--restaurants`, or `--orders-per-minute` to change the experiment setup.

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

The environment supports order arrivals, preparation, driver dispatch, travel, pickups, and deliveries. The three baselines are nearest driver, oldest/most urgent ready order, and minimum estimated delivery time. The comparison runner reports average deliveries, reward points, delivery time, and travel distance over matched episode seeds. The GUI lets you run a selected baseline or compare all three. PPO is planned for a later commit.

The current simulator uses a lightweight 2D coordinate map and discrete simulated minutes. It is an educational model, not a reconstruction of any commercial delivery platform.
