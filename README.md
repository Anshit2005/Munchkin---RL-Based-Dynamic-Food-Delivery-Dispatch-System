# Munchkin - RL Based Dynamic Food Delivery Dispatch System

Munchkin is a configurable food-delivery dispatch simulator for studying sequential dispatch decisions with reinforcement learning. This initial version contains the simulation environment and a Tkinter game interface. The explicit heuristic baselines and PPO agent will be added in later commits.

## Requirements

- Python 3.10 or newer
- Tkinter (usually included with Python; Linux distributions may package it separately)

No third-party Python packages are required for the simulator or interface.

## Run

From the project root:

```bash
python3 -m food_delivery_game
```

Use **Start / Pause** to advance the simulation and **Reset** to start a new scenario. Configure restaurants, drivers, order arrival rate, and travel speed before starting. Episodes run for a fixed three simulated minutes.

## Project layout

```text
food_delivery_game/
  __main__.py       Tkinter entry point
  config.py         Simulation configuration
  entities.py       Restaurant, driver, and order data models
  environment.py    Discrete-time delivery simulation
  ui.py             Map, controls, and live status display
```

## Initial version scope

The environment supports order arrivals, preparation, driver dispatch, travel, pickups, deliveries, and basic episode metrics. A simple built-in setup rule keeps this version runnable; it is environment plumbing, not one of the planned evaluated baselines. Later work will add the three explicit baselines and then PPO, with common episode seeds and metrics for comparison.

The current simulator uses a lightweight 2D coordinate map and discrete simulated minutes. It is an educational model, not a reconstruction of any commercial delivery platform.
