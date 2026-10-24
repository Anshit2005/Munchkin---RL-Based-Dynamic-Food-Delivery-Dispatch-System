"""Leaderboard comparing all strategies run on the same seed and world."""
from __future__ import annotations

import math

import pygame

from ui import policies
from ui.widgets import ACCENT, BG, GOOD, MUTED, PANEL, PANEL_LIGHT, TEXT, WARN, Button, text

COLUMNS = [("Rank", None, 60), ("Strategy", None, 220), ("Delivered", "n_delivered", 110),
           ("On-time %", "on_time_pct", 110), ("Avg delivery (s)", "avg_delivery_time", 140),
           ("Avg wait (s)", "avg_wait_time", 120), ("Distance", "total_distance", 100),
           ("Utilization %", "utilization", 120), ("Reward", "reward", 100)]
LOWER_BETTER = {"avg_delivery_time", "avg_wait_time", "total_distance"}


def _val(res, key):
    return res.reward_total if key == "reward" else res.metrics[key]


class LeaderboardScreen:
    def __init__(self, game, results, cfg, seed):
        self.game, self.results, self.cfg, self.seed = game, results, cfg, seed
        self.sort_key = "reward"
        self.btn_back = Button((40, 650, 160, 40), "Back to replay", self._back)
        self.btn_cfg = Button((215, 650, 160, 40), "New simulation", self._config)
        self.row_rects = []

    def _back(self):
        from ui.screens.sim_screen import SimScreen
        self.game.goto(SimScreen(self.game, self.results, self.cfg, self.seed))

    def _config(self):
        from ui.screens.config_screen import ConfigScreen
        self.game.goto(ConfigScreen(self.game))

    def _sorted(self):
        key = self.sort_key
        def k(item):
            v = _val(item[1], key)
            v = -math.inf if (isinstance(v, float) and math.isnan(v)) else v
            return v if key in LOWER_BETTER and False else (-v if key in LOWER_BETTER else v)
        return sorted(self.results, key=k, reverse=True)

    def handle(self, event):
        self.btn_back.handle(event)
        self.btn_cfg.handle(event)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x = 40
            for _, key, w in COLUMNS:
                if key and pygame.Rect(x, 150, w, 36).collidepoint(event.pos):
                    self.sort_key = key
                x += w
            for rect, (k, _) in self.row_rects:
                if rect.collidepoint(event.pos):
                    from ui.screens.sim_screen import SimScreen
                    idx = [kk for kk, _ in self.results].index(k)
                    self.game.goto(SimScreen(self.game, self.results, self.cfg, self.seed, idx))

    def update(self, dt):
        pass

    def draw(self, surf):
        surf.fill(BG)
        text(surf, "Leaderboard", (40, 30), 36, TEXT, True)
        text(surf, f"Same world for every strategy  -  seed {self.seed}  -  exactly 3:00 simulated  -  "
                   f"click a column to sort, a row to replay", (40, 84), 16, MUTED)
        pygame.draw.rect(surf, PANEL, (30, 130, 1060, 500), border_radius=12)
        x = 40
        for label, key, w in COLUMNS:
            col = ACCENT if key == self.sort_key else MUTED
            text(surf, label + (" v" if key == self.sort_key else ""), (x, 158), 15, col, True)
            x += w
        self.row_rects = []
        for r, (k, res) in enumerate(self._sorted()):
            y = 195 + r * 62
            rect = pygame.Rect(36, y, 1048, 54)
            hover = rect.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(surf, (52, 62, 90) if hover else PANEL_LIGHT, rect, border_radius=8)
            self.row_rects.append((rect, (k, res)))
            x = 40
            for label, key, w in COLUMNS:
                if label == "Rank":
                    text(surf, f"#{r + 1}", (x, y + 14), 22, GOOD if r == 0 else TEXT, True)
                elif label == "Strategy":
                    text(surf, policies.LABELS[k], (x, y + 6), 20, TEXT, True)
                    text(surf, policies.SUBTITLES[k], (x, y + 31), 13, MUTED)
                else:
                    v = _val(res, key)
                    s = "-" if (isinstance(v, float) and math.isnan(v)) else (f"{v:.0f}" if key == "n_delivered" else f"{v:.1f}")
                    if key == "n_delivered":
                        s = f"{res.metrics['n_delivered']}/{res.metrics['n_orders']}"
                    text(surf, s, (x, y + 15), 19, TEXT)
                x += w
        self.btn_back.draw(surf)
        self.btn_cfg.draw(surf)
