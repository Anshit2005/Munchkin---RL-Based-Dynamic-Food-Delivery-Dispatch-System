"""Simulation screen: animated replay of a recorded episode."""
from __future__ import annotations

import pygame

from environment.constants import N_TICKS
from ui import policies
from ui.render import DRIVER_COLOR, DRIVER_NAME, MapView
from ui.widgets import ACCENT, BAD, BG, GOOD, MUTED, PANEL, TEXT, WARN, Button, text

SPEEDS = [1, 2, 5, 10, 20]


class SimScreen:
    def __init__(self, game, results, cfg, seed, index=0):
        self.game, self.results, self.cfg, self.seed = game, results, cfg, seed
        self.index = index
        self.speed_i = 2
        self.paused = False
        self.pos = 0.0
        self.map = MapView(pygame.Rect(20, 50, 640, 640), [])
        self._load(index)
        self._build_buttons()

    def _load(self, index):
        self.index = index
        self.key, self.res = self.results[index]
        self.frames = self.res.frames
        sc = self.frames[0]
        self.log = [(f["t"], line) for f in self.frames for line in f["log"]]
        rests = {}
        for f in self.frames:
            for o in f["orders"]:
                rests[o["rx"], o["ry"]] = None
        self.map.restaurants = self._restaurants()
        self.pos = 0.0
        self.paused = False

    def _restaurants(self):
        # restaurants are static; recover them from the recorded order locations plus the config count
        pts = []
        for f in self.frames:
            for o in f["orders"]:
                p = (o["rx"], o["ry"])
                if p not in pts:
                    pts.append(p)
        return pts

    def _build_buttons(self):
        x0 = 690
        self.btn_pause = Button((x0, 640, 90, 40), "Pause", self._toggle_pause)
        self.btn_step = Button((x0 + 100, 640, 70, 40), "Step", self._step)
        self.btn_restart = Button((x0 + 180, 640, 90, 40), "Restart", self._restart)
        self.btn_back = Button((x0 + 280, 640, 110, 40), "Config", self._back)
        self.speed_btns = [Button((x0 + i * 78, 585, 70, 36), f"{s}x", lambda i=i: self._set_speed(i),
                                  toggle=True, active=(i == self.speed_i), size=16) for i, s in enumerate(SPEEDS)]
        self.tab_btns = []
        if len(self.results) > 1:
            for i, (k, _) in enumerate(self.results):
                self.tab_btns.append(Button((x0 + i * 78, 50, 72, 34), policies.LABELS[k].replace("Baseline ", "B"),
                                            lambda i=i: self._load(i), toggle=True, active=(i == self.index), size=15))
            self.btn_board = Button((x0, 92, 180, 34), "Leaderboard", self._leaderboard, size=16)
        else:
            self.btn_board = None

    # ---------------------------------------------------------------- actions
    def _toggle_pause(self):
        self.paused = not self.paused

    def _step(self):
        self.paused = True
        self.pos = min(N_TICKS, int(self.pos) + 1)

    def _restart(self):
        self.pos, self.paused = 0.0, False

    def _set_speed(self, i):
        self.speed_i = i
        for j, b in enumerate(self.speed_btns):
            b.active = j == i

    def _back(self):
        from ui.screens.config_screen import ConfigScreen
        self.game.goto(ConfigScreen(self.game))

    def _leaderboard(self):
        from ui.screens.leaderboard_screen import LeaderboardScreen
        self.game.goto(LeaderboardScreen(self.game, self.results, self.cfg, self.seed))

    # ---------------------------------------------------------------- pygame hooks
    def handle(self, event):
        buttons = [self.btn_pause, self.btn_step, self.btn_restart, self.btn_back] + self.speed_btns + self.tab_btns
        if self.btn_board:
            buttons.append(self.btn_board)
        for b in buttons:
            if b.handle(event):
                for j, tb in enumerate(self.tab_btns):
                    tb.active = j == self.index
                return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._toggle_pause()
            elif event.key == pygame.K_RIGHT:
                self._step()

    def update(self, dt):
        if not self.paused and self.pos < N_TICKS:
            self.pos = min(N_TICKS, self.pos + dt * SPEEDS[self.speed_i])
        if self.pos >= N_TICKS and len(self.results) > 1 and not getattr(self, "_shown_board", False):
            pass

    def draw(self, surf):
        surf.fill(BG)
        i = min(int(self.pos), N_TICKS)
        frame = self.frames[i]
        nxt = self.frames[min(i + 1, N_TICKS)]
        alpha = 0.0 if i >= N_TICKS else (self.pos - i)
        text(surf, f"{self.res.policy.upper().replace('_', ' ')}  -  {self.cfg.n_restaurants} restaurants, "
                   f"{self.cfg.n_drivers} drivers, {self.cfg.orders_per_min:g} orders/min, {self.cfg.traffic} traffic",
             (20, 14), 18, MUTED)
        self.map.draw(surf, frame, nxt, alpha)
        self.map.draw_legend(surf, (24, 698))
        self._panel(surf, frame, i)

    def _panel(self, surf, frame, i):
        x0 = 690
        y = 140 if self.btn_board else 60
        # clock
        sec = int(self.pos)
        text(surf, f"{sec // 60}:{sec % 60:02d}", (x0, y), 54, TEXT, True)
        text(surf, "/ 3:00", (x0 + 128, y + 28), 22, MUTED)
        bar = pygame.Rect(x0, y + 70, 390, 10)
        pygame.draw.rect(surf, (40, 48, 70), bar, border_radius=5)
        pygame.draw.rect(surf, ACCENT, (bar.x, bar.y, int(bar.w * min(1, self.pos / N_TICKS)), bar.h), border_radius=5)
        # live stats
        orders = frame["orders"]
        late = sum(1 for o in orders if o["late"])
        waiting = sum(1 for o in orders if o["status"] < 2)
        drv = frame["drivers"]
        busy = sum(1 for d in drv if d["status"] != 0)
        stats = [("Orders placed", len(orders), TEXT), ("Delivered", frame["n_delivered"], GOOD),
                 ("Waiting for driver", waiting, WARN), ("Late", late, BAD),
                 ("Drivers busy", f"{busy}/{len(drv)}", ACCENT)]
        for k, (lab, val, col) in enumerate(stats):
            cx = x0 + (k % 3) * 135
            cy = y + 100 + (k // 3) * 62
            pygame.draw.rect(surf, PANEL, (cx, cy, 125, 54), border_radius=8)
            text(surf, lab, (cx + 8, cy + 6), 12, MUTED)
            text(surf, val, (cx + 8, cy + 22), 22, col, True)
        # event log
        ly = y + 235
        pygame.draw.rect(surf, PANEL, (x0, ly, 390, 552 - ly), border_radius=8)
        text(surf, "EVENTS", (x0 + 10, ly + 6), 12, MUTED, True)
        recent = [e for e in self.log if e[0] <= i][-7:]
        for k, (t, line) in enumerate(reversed(recent)):
            col = BAD if "LATE" in line else (GOOD if "delivered" in line else TEXT)
            text(surf, f"{t // 60}:{t % 60:02d}  {line}", (x0 + 10, ly + 26 + k * 21), 14, col)
        text(surf, "SPEED (simulated seconds per real second)", (x0, 562), 12, MUTED, True)
        self.btn_pause.label = "Play" if self.paused else "Pause"
        for b in [self.btn_pause, self.btn_step, self.btn_restart, self.btn_back] + self.speed_btns + self.tab_btns:
            b.draw(surf)
        if self.btn_board:
            self.btn_board.draw(surf)
        if self.pos >= N_TICKS:
            m = self.res.metrics
            pygame.draw.rect(surf, (30, 44, 40), (20, 50, 640, 70), border_radius=10)
            text(surf, "Episode complete (3:00)", (36, 58), 20, GOOD, True)
            text(surf, f"delivered {m['n_delivered']}/{m['n_orders']}   on-time {m['on_time_pct']:.0f}%   "
                       f"avg delivery {m['avg_delivery_time']:.0f}s   reward {self.res.reward_total:.1f}",
                 (36, 88), 16, TEXT)
