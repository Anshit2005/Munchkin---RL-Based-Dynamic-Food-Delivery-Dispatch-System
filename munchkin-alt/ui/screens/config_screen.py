"""Configuration screen: world sliders, scenario presets, algorithm selection.

There is deliberately NO simulation-duration control: every episode is exactly 3 simulated minutes.
"""
from __future__ import annotations

import random

import pygame

from environment.config import EnvConfig, load_env_defaults, load_scenarios
from simulation.runner import run_episode
from ui import policies
from ui.widgets import (ACCENT, BG, GOOD, MUTED, PANEL, TEXT, WARN, Button, Slider, text)

PRESETS = [("low_demand", "Low demand"), ("medium_demand", "Medium demand"), ("high_demand", "High demand"),
           ("driver_shortage", "Driver shortage"), ("driver_surplus", "Driver surplus")]


class ConfigScreen:
    def __init__(self, game):
        self.game = game
        self.base = load_env_defaults()
        self.scenarios = load_scenarios()
        self.avail = policies.availability()
        self.sl_rest = Slider((70, 215, 380, 8), "Restaurants", 1, 8, 5)
        self.sl_drv = Slider((70, 295, 380, 8), "Delivery partners", 1, 12, 8)
        self.sl_rate = Slider((70, 375, 380, 8), "Order arrival rate (orders / min)", 1, 10, 5, 0.5, "{:.1f}")
        self.traffic = "normal"
        self.batching = False
        self.seed = 1_000_000
        self.algo = "all" if all(a for a, _ in self.avail.values()) else "baseline_3"
        self.message = ""
        self._build_buttons()

    # ------------------------------------------------------------------ layout
    def _build_buttons(self):
        self.traffic_btns = [Button((70 + i * 130, 440, 120, 38), n.capitalize(),
                                    lambda n=n: self._set_traffic(n), toggle=True, active=(n == self.traffic))
                             for i, n in enumerate(["light", "normal", "heavy"])]
        self.batch_btn = Button((70, 505, 250, 38), "Order batching: OFF", self._toggle_batch)
        self.seed_minus = Button((70, 575, 40, 38), "-", lambda: self._bump_seed(-1))
        self.seed_plus = Button((330, 575, 40, 38), "+", lambda: self._bump_seed(1))
        self.seed_rand = Button((380, 575, 100, 38), "Random", self._rand_seed)
        self.preset_btns = [Button((560, 180 + i * 46, 220, 38), label, lambda k=k: self._apply_preset(k))
                            for i, (k, label) in enumerate(PRESETS)]
        self.algo_btns = []
        order = ["baseline_1", "baseline_2", "baseline_3", "dqn", "ppo", "all"]
        for i, k in enumerate(order):
            ok = self.avail[k][0] if k in self.avail else all(a for a, _ in self.avail.values()) or True
            label = "Run All" if k == "all" else policies.LABELS[k]
            x = 830 + (i % 2) * 130
            y = 180 + (i // 2) * 60
            self.algo_btns.append((k, Button((x, y, 120, 46), label, lambda k=k: self._set_algo(k),
                                             enabled=ok, toggle=True, active=(k == self.algo),
                                             tooltip=self.avail.get(k, (True, ""))[1])))
        self.start_btn = Button((830, 400, 250, 56), "START SIMULATION", self.start, size=20, active=True)

    def _set_traffic(self, n):
        self.traffic = n
        for b, name in zip(self.traffic_btns, ["light", "normal", "heavy"]):
            b.active = name == n

    def _toggle_batch(self):
        self.batching = not self.batching

    def _bump_seed(self, d):
        self.seed += d

    def _rand_seed(self):
        self.seed = random.randint(1_000_000, 1_999_999)    # evaluation-range seeds, never training seeds

    def _set_algo(self, k):
        self.algo = k
        for kk, b in self.algo_btns:
            b.active = kk == k

    def _apply_preset(self, key):
        c = self.scenarios[key]
        self.sl_rest.value, self.sl_drv.value, self.sl_rate.value = c.n_restaurants, c.n_drivers, c.orders_per_min

    # ------------------------------------------------------------------ logic
    def build_config(self) -> EnvConfig:
        return self.base.replace(name="custom", n_restaurants=int(self.sl_rest.value),
                                 n_drivers=int(self.sl_drv.value), orders_per_min=float(self.sl_rate.value),
                                 traffic=self.traffic, batching=self.batching)

    def start(self):
        cfg = self.build_config()
        keys = [k for k in policies.KEYS if self.avail[k][0]] if self.algo == "all" else [self.algo]
        self.message = "Simulating..."
        self.draw(self.game.screen)
        pygame.display.flip()
        results = []
        for k in keys:
            results.append((k, run_episode(policies.make_policy(k), cfg, self.seed, record=True)))
        from ui.screens.sim_screen import SimScreen
        self.game.goto(SimScreen(self.game, results, cfg, self.seed))

    # ------------------------------------------------------------------ pygame hooks
    def handle(self, event):
        for s in (self.sl_rest, self.sl_drv, self.sl_rate):
            if s.handle(event):
                return
        for b in (self.traffic_btns + [self.batch_btn, self.seed_minus, self.seed_plus, self.seed_rand,
                                       self.start_btn] + self.preset_btns + [b for _, b in self.algo_btns]):
            if b.handle(event):
                return

    def update(self, dt):
        pass

    def draw(self, surf):
        surf.fill(BG)
        text(surf, "Food Delivery Dispatch", (70, 40), 38, TEXT, True)
        text(surf, "Reinforcement learning vs heuristic dispatch - same city, same orders, same 3 minutes",
             (70, 92), 18, MUTED)
        pygame.draw.rect(surf, PANEL, (40, 130, 470, 540), border_radius=14)
        pygame.draw.rect(surf, PANEL, (530, 130, 270, 540), border_radius=14)
        pygame.draw.rect(surf, PANEL, (815, 130, 285, 540), border_radius=14)
        text(surf, "WORLD", (70, 142), 15, MUTED, True)
        text(surf, "SCENARIO PRESETS", (560, 146), 15, MUTED, True)
        text(surf, "STRATEGY", (830, 146), 15, MUTED, True)
        for s in (self.sl_rest, self.sl_drv, self.sl_rate):
            s.draw(surf)
        text(surf, "Traffic / travel conditions", (70, 410), 17, MUTED)
        for b in self.traffic_btns:
            b.draw(surf)
        self.batch_btn.label = f"Order batching: {'ON' if self.batching else 'OFF'}"
        self.batch_btn.active = self.batching
        self.batch_btn.draw(surf)
        text(surf, "Random seed (same seed = same orders for every strategy)", (70, 548), 15, MUTED)
        self.seed_minus.draw(surf)
        self.seed_plus.draw(surf)
        self.seed_rand.draw(surf)
        text(surf, self.seed, (225, 594), 20, TEXT, True, "center")
        for b in self.preset_btns:
            b.draw(surf)
        for _, b in self.algo_btns:
            b.draw(surf)
        for k, b in self.algo_btns:
            if not b.enabled:
                text(surf, "(no trained model found)", (830, 480), 14, WARN)
                break
        self.start_btn.draw(surf)
        # fixed duration banner - intentionally not a slider
        pygame.draw.rect(surf, (36, 52, 48), (560, 420, 220, 70), border_radius=10)
        text(surf, "Episode duration", (670, 436), 15, MUTED, anchor="midtop")
        text(surf, "3:00 (fixed)", (670, 456), 24, GOOD, True, "midtop")
        if self.batching:
            text(surf, "Note: RL agents were trained without batching.", (70, 640), 14, WARN)
        if self.message:
            text(surf, self.message, (830, 470), 18, ACCENT, True)
