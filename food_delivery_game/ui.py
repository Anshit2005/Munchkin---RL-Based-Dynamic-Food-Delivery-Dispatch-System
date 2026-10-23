"""Two-player head-to-head game interface for Munchkin."""

from __future__ import annotations

import math
import time
import tkinter as tk
from tkinter import messagebox, ttk

from .baselines import POLICIES, create_policy
from .config import SimulationConfig
from .environment import DeliveryEnvironment
from .entities import DriverStatus, OrderStatus


class DeliveryGame:
    """Run two strategies against matching copies of the same hourly scenario."""

    MATCH_MINUTES = 60
    SECOND_PER_GAME_MINUTE_MS = 1000
    COLORS = {
        "background": "#0b1220",
        "panel": "#111c2f",
        "card": "#17263d",
        "map": "#101b2d",
        "grid": "#1b2c46",
        "text": "#f1f5f9",
        "muted": "#9fb0c7",
        "cyan": "#22d3ee",
        "green": "#34d399",
        "orange": "#fb923c",
        "pink": "#f472b6",
        "purple": "#a78bfa",
        "yellow": "#facc15",
    }

    POLICY_HINTS = {
        "Nearest Driver": "Chooses the closest available driver and ready-order pickup pair.",
        "Oldest / Most Urgent": "Serves the ready order that has waited the longest.",
        "Minimum Estimated Delivery Time": "Chooses the pair with the shortest pickup-plus-delivery ETA.",
    }

    def __init__(self, root: tk.Tk, player1_name: str, player2_name: str) -> None:
        self.root = root
        self.root.title("Munchkin - Head-to-Head Dispatch")
        self.root.configure(background=self.COLORS["background"])
        self.root.minsize(1240, 720)
        self.player_names = [player1_name or "Player 1", player2_name or "Player 2"]
        self.player_colors = [self.COLORS["cyan"], self.COLORS["pink"]]
        self.running = False
        self.match_started = False
        self.match_complete = False
        self._timer_id: str | None = None
        self._animation_id: str | None = None
        self._minute_started_at: float | None = None
        self._progress_fraction = 0.0
        self._match_number = 0
        self._feed_prompt_shown = True
        self.sound_enabled = tk.BooleanVar(value=True)
        self._configure_theme()
        self.strategy_selectors: list[ttk.Combobox] = []
        self.strategy_hints: list[tk.Label] = []
        self.player_stat_labels: list[dict[str, tk.Label]] = []
        self.player_score_labels: list[tk.Label] = []
        self._build_header()
        self._build_main_layout()
        self._reset_match(advance_seed=False)

    def _configure_theme(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", background="#243650", foreground=self.COLORS["text"], padding=(9, 7), borderwidth=0)
        style.map("TButton", background=[("active", "#345071"), ("disabled", "#1a2638")])
        style.configure("TCombobox", fieldbackground="#17263d", background="#243650", foreground=self.COLORS["text"], arrowcolor=self.COLORS["cyan"])
        style.configure("TCheckbutton", background=self.COLORS["panel"], foreground=self.COLORS["text"])
        style.map("TCheckbutton", background=[("active", self.COLORS["panel"])])

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=self.COLORS["background"], padx=18, pady=10)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(header, text="MUNCHKIN", bg=self.COLORS["background"], fg=self.COLORS["cyan"], font=("TkDefaultFont", 22, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="  DISPATCH DUEL", bg=self.COLORS["background"], fg=self.COLORS["muted"], font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, pady=(7, 0))
        self.match_label = tk.Label(header, text="60 GAME MINUTES  •  1 SECOND = 1 GAME MINUTE", bg=self.COLORS["background"], fg=self.COLORS["yellow"], font=("TkDefaultFont", 10, "bold"))
        self.match_label.pack(side=tk.RIGHT, pady=(7, 0))

    def _build_main_layout(self) -> None:
        body = tk.Frame(self.root, bg=self.COLORS["background"])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        self.player_panels = [
            self._build_player_panel(body, 0),
            self._build_player_panel(body, 1),
        ]
        self.player_panels[0].pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.player_panels[1].pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self._build_center_panel(body)

    def _build_player_panel(self, parent: tk.Frame, player_index: int) -> tk.Frame:
        color = self.player_colors[player_index]
        panel = tk.Frame(parent, bg=self.COLORS["panel"], width=245, padx=13, pady=13)
        panel.pack_propagate(False)
        tk.Label(panel, text=f"PLAYER {player_index + 1}", bg=self.COLORS["panel"], fg=color, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        tk.Label(panel, text=self.player_names[player_index], bg=self.COLORS["panel"], fg=self.COLORS["text"], font=("TkDefaultFont", 17, "bold"), wraplength=215).pack(anchor="w", pady=(2, 10))
        tk.Label(panel, text="STRATEGY", bg=self.COLORS["panel"], fg=self.COLORS["muted"], font=("TkDefaultFont", 8, "bold")).pack(anchor="w")
        selector = ttk.Combobox(panel, values=list(POLICIES), state="readonly", width=27)
        selector.current(player_index % len(POLICIES))
        selector.pack(fill=tk.X, pady=(3, 5))
        hint = tk.Label(panel, text=self.POLICY_HINTS[selector.get()], bg=self.COLORS["panel"], fg=color, wraplength=215, justify=tk.LEFT, font=("TkDefaultFont", 9))
        hint.pack(anchor="w", pady=(0, 10))
        selector.bind("<<ComboboxSelected>>", lambda _event, i=player_index: self._update_policy_hint(i))
        self.strategy_selectors.append(selector)
        self.strategy_hints.append(hint)

        score = tk.Label(panel, text="0 PTS", bg="#253754", fg=self.COLORS["yellow"], font=("TkDefaultFont", 24, "bold"), padx=8, pady=8)
        score.pack(fill=tk.X, pady=(0, 12))
        stat_labels: dict[str, tk.Label] = {}
        for key, title in (("delivered", "DELIVERED"), ("pending", "PENDING AT END"), ("avg_time", "AVG DELIVERY"), ("travel", "TRAVELLED"), ("free", "DRIVERS FREE")):
            row = tk.Frame(panel, bg=self.COLORS["panel"])
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text=title, bg=self.COLORS["panel"], fg=self.COLORS["muted"], font=("TkDefaultFont", 8, "bold")).pack(side=tk.LEFT)
            value = tk.Label(row, text="—", bg=self.COLORS["panel"], fg=self.COLORS["text"], font=("TkDefaultFont", 10, "bold"))
            value.pack(side=tk.RIGHT)
            stat_labels[key] = value
        self.player_stat_labels.append(stat_labels)
        self.player_score_labels.append(score)
        return panel

    def _build_center_panel(self, parent: tk.Frame) -> None:
        center = tk.Frame(parent, bg=self.COLORS["background"])
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        heading = tk.Frame(center, bg=self.COLORS["background"])
        heading.pack(fill=tk.X, pady=(0, 6))
        tk.Label(heading, text="SAME CITY  •  SAME ORDERS  •  TWO POLICIES", bg=self.COLORS["background"], fg=self.COLORS["text"], font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT)
        self.activity_label = tk.Label(heading, text="● READY", bg=self.COLORS["background"], fg=self.COLORS["green"], font=("TkDefaultFont", 10, "bold"))
        self.activity_label.pack(side=tk.RIGHT)

        maps = tk.Frame(center, bg=self.COLORS["background"])
        maps.pack(fill=tk.BOTH, expand=True)
        self.map_canvases: list[tk.Canvas] = []
        for i in range(2):
            map_card = tk.Frame(maps, bg=self.COLORS["card"], padx=5, pady=5)
            map_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5) if i == 0 else (5, 0))
            tk.Label(map_card, text=f"{self.player_names[i]}'s simulation", bg=self.COLORS["card"], fg=self.player_colors[i], font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))
            canvas = tk.Canvas(map_card, width=380, height=320, bg=self.COLORS["map"], highlightthickness=1, highlightbackground="#31445f")
            canvas.pack(fill=tk.BOTH, expand=True)
            self.map_canvases.append(canvas)

        progress_row = tk.Frame(center, bg=self.COLORS["background"])
        progress_row.pack(fill=tk.X, pady=(7, 2))
        tk.Label(progress_row, text="ACTIVE PLAYER CLOCK", bg=self.COLORS["background"], fg=self.COLORS["muted"], font=("TkDefaultFont", 8, "bold")).pack(side=tk.LEFT)
        self.clock_label = tk.Label(progress_row, text="00:00 / 60:00", bg=self.COLORS["background"], fg=self.COLORS["cyan"], font=("TkDefaultFont", 9, "bold"))
        self.clock_label.pack(side=tk.RIGHT)
        self.progress = tk.Canvas(center, height=14, bg="#17263d", highlightthickness=0)
        self.progress.pack(fill=tk.X, pady=(0, 6))
        self.progress.bind("<Configure>", lambda _event: self._update_progress())

        feed_panel = tk.Frame(center, bg=self.COLORS["card"], padx=8, pady=5)
        feed_panel.pack(fill=tk.X, pady=(0, 7))
        tk.Label(feed_panel, text="LIVE DECISION FEED  •  why each algorithm chose its assignment", bg=self.COLORS["card"], fg=self.COLORS["yellow"], font=("TkDefaultFont", 8, "bold")).pack(anchor="w")
        self.decision_feed = tk.Text(feed_panel, height=4, wrap=tk.WORD, state=tk.DISABLED, bg=self.COLORS["card"], fg=self.COLORS["text"], relief=tk.FLAT, font=("TkFixedFont", 8), padx=3, pady=2)
        self.decision_feed.pack(fill=tk.X, pady=(2, 0))

        self._build_common_controls(center)

    def _build_common_controls(self, parent: tk.Frame) -> None:
        controls = tk.Frame(parent, bg=self.COLORS["panel"], padx=10, pady=7)
        controls.pack(fill=tk.X)
        tk.Label(controls, text="COMMON MATCH SETTINGS", bg=self.COLORS["panel"], fg=self.COLORS["text"], font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        sliders = tk.Frame(controls, bg=self.COLORS["panel"])
        sliders.pack(fill=tk.X)
        self.restaurant_count = self._add_scale(sliders, "Restaurants", 1, 10, 5, self.COLORS["orange"])
        self.restaurant_count.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.driver_count = self._add_scale(sliders, "Delivery partners", 1, 16, 8, self.COLORS["green"])
        self.driver_count.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.order_rate = self._add_scale(sliders, "Orders per minute", 0.5, 5, 1.5, self.COLORS["pink"], resolution=0.5)
        self.order_rate.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        buttons = tk.Frame(controls, bg=self.COLORS["panel"])
        buttons.pack(fill=tk.X, pady=(4, 0))
        self.start_button = ttk.Button(buttons, text="▶  Start match", command=self.toggle_run)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        self.step_button = ttk.Button(buttons, text="⏭  Step one game minute", command=self.step_once)
        self.step_button.pack(side=tk.LEFT, padx=5)
        self.reset_button = ttk.Button(buttons, text="↻  New match", command=self.reset_match)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(buttons, text="Sound", variable=self.sound_enabled).pack(side=tk.RIGHT, padx=5)

    def _add_scale(self, parent: tk.Frame, label: str, low: float, high: float, initial: float, color: str, resolution: float = 1) -> tk.Frame:
        frame = tk.Frame(parent, bg=self.COLORS["panel"])
        tk.Label(frame, text=label, bg=self.COLORS["panel"], fg=self.COLORS["muted"], font=("TkDefaultFont", 8)).pack(anchor="w")
        scale = tk.Scale(
            frame, from_=low, to=high, resolution=resolution, orient=tk.HORIZONTAL, length=170,
            bg=self.COLORS["panel"], fg=self.COLORS["text"], troughcolor="#26364e",
            activebackground=color, highlightthickness=0, bd=0, showvalue=True,
            font=("TkDefaultFont", 8),
        )
        scale.set(initial)
        scale.pack(fill=tk.X)
        frame.scale = scale
        return frame

    def _update_policy_hint(self, player_index: int) -> None:
        self.strategy_hints[player_index].configure(text=self.POLICY_HINTS[self.strategy_selectors[player_index].get()])
        if not self.match_started and hasattr(self, "policies"):
            self.policies[player_index] = create_policy(self.strategy_selectors[player_index].get())

    def _make_config(self, seed: int) -> SimulationConfig:
        return SimulationConfig(
            restaurant_count=int(self.restaurant_count.scale.get()),
            driver_count=int(self.driver_count.scale.get()),
            orders_per_minute=float(self.order_rate.scale.get()),
            episode_minutes=self.MATCH_MINUTES,
            seed=seed,
        )

    def _reset_match(self, advance_seed: bool = True) -> None:
        self._stop_timers()
        if advance_seed:
            self._match_number += 1
        self.running = False
        self.match_started = False
        self.match_complete = False
        self.active_player_index = 0
        self._summary_shown = False
        self._progress_fraction = 0.0
        self._feed_prompt_shown = True
        self.start_button.configure(text="▶  Start match")
        config = self._make_config(1000 + self._match_number)
        self.envs = [DeliveryEnvironment(config), DeliveryEnvironment(config)]
        self.policies = [create_policy(selector.get()) for selector in self.strategy_selectors]
        self._feed_indices = [0, 0]
        self._set_locked(False)
        self._set_activity("● READY FOR DUEL", self.COLORS["green"])
        self._draw_all()
        self._append_new_decisions()

    def reset_match(self) -> None:
        self._reset_match(advance_seed=True)

    def _set_locked(self, locked: bool) -> None:
        for selector in self.strategy_selectors:
            selector.configure(state="disabled" if locked else "readonly")
        for control in (self.restaurant_count, self.driver_count, self.order_rate):
            control.scale.configure(state=tk.DISABLED if locked else tk.NORMAL)

    def toggle_run(self) -> None:
        if self.match_complete:
            self._reset_match(advance_seed=True)
        elif not self.match_started:
            self._reset_match(advance_seed=False)
        self.running = not self.running
        self.match_started = True
        self._set_locked(True)
        if self.running:
            self.start_button.configure(text="Ⅱ  Pause match")
            self._set_activity(f"● PLAYER {self.active_player_index + 1} LIVE", self.player_colors[self.active_player_index])
            self._schedule_minute()
        else:
            self.start_button.configure(text="▶  Resume match")
            self._stop_timers()
            self._set_activity("Ⅱ PAUSED", self.COLORS["yellow"])

    def step_once(self) -> None:
        if self.match_complete:
            return
        if not self.match_started:
            self._reset_match(advance_seed=False)
        if self.running:
            self.running = False
            self._stop_timers()
            self.start_button.configure(text="▶  Resume match")
        self.match_started = True
        self._set_locked(True)
        self._progress_fraction = 0.0
        self._step_active_player()

    def _schedule_minute(self) -> None:
        if not self.running:
            return
        self._minute_started_at = time.monotonic() - self._progress_fraction * self.SECOND_PER_GAME_MINUTE_MS / 1000
        remaining = max(1, round(self.SECOND_PER_GAME_MINUTE_MS * (1 - self._progress_fraction)))
        self._timer_id = self.root.after(remaining, self._run_minute)
        self._animate_progress_later()

    def _animate_progress_later(self) -> None:
        if self.running:
            self._animation_id = self.root.after(50, self._animate_progress)

    def _animate_progress(self) -> None:
        self._animation_id = None
        if not self.running or self._minute_started_at is None:
            return
        elapsed = (time.monotonic() - self._minute_started_at) * 1000
        self._progress_fraction = min(elapsed / self.SECOND_PER_GAME_MINUTE_MS, 1.0)
        self._update_progress()
        self._animate_progress_later()

    def _run_minute(self) -> None:
        self._timer_id = None
        self._cancel_animation()
        self._minute_started_at = None
        if not self.running:
            return
        self._progress_fraction = 0.0
        self._step_active_player()
        if self.match_complete:
            self.running = False
            self.start_button.configure(text="▶  Start next match")
            self._set_locked(False)
            self._show_results()
        else:
            self._schedule_minute()

    def _step_active_player(self) -> None:
        player_index = self.active_player_index
        env = self.envs[player_index]
        old_delivered = len(env.delivered_orders)
        env.step(self.policies[player_index])
        self._draw_all()
        self._append_new_decisions()
        if len(env.delivered_orders) > old_delivered:
            self._play_sound()

        if env.done and player_index == 0:
            self.active_player_index = 1
            self._progress_fraction = 0.0
            self._set_activity("● PLAYER 2 UP NEXT", self.player_colors[1])
            self._update_progress()
        elif env.done:
            self.match_complete = True
            self.running = False
            self._set_activity("★ MATCH COMPLETE", self.COLORS["yellow"])
            self._play_sound()
            self._set_locked(False)
            self.start_button.configure(text="▶  Start next match")
            self._show_results()
        elif self.running:
            self._set_activity(f"● PLAYER {player_index + 1} LIVE", self.player_colors[player_index])
        else:
            self._set_activity(f"● PLAYER {self.active_player_index + 1} READY FOR NEXT MINUTE", self.player_colors[self.active_player_index])

    def _append_new_decisions(self) -> None:
        self.decision_feed.configure(state=tk.NORMAL)
        if not self.match_started:
            self.decision_feed.delete("1.0", tk.END)
            self.decision_feed.insert(tk.END, "Choose strategies, then start the match to see each decision and its reason.\n")
            self._feed_prompt_shown = True
            self.decision_feed.configure(state=tk.DISABLED)
            return
        if self.match_started and self._feed_prompt_shown:
            self.decision_feed.delete("1.0", tk.END)
            self._feed_prompt_shown = False
        added_events = 0
        for i, env in enumerate(self.envs):
            for event in env.dispatch_log[self._feed_indices[i]:]:
                self.decision_feed.insert(
                    tk.END,
                    f"{self.player_names[i]} · {event.time:02d}:00 · {event.strategy}\n"
                    f"    D{event.driver_id + 1} → O{event.order_id}: {event.explanation}\n",
                    f"player{i + 1}",
                )
                added_events += 1
            self._feed_indices[i] = len(env.dispatch_log)
        self.decision_feed.tag_configure("player1", foreground=self.player_colors[0])
        self.decision_feed.tag_configure("player2", foreground=self.player_colors[1])
        line_count = int(self.decision_feed.index("end-1c").split(".")[0])
        if line_count > 28:
            self.decision_feed.delete("1.0", f"{line_count - 28}.0")
        if added_events == 0:
            active_env = self.envs[self.active_player_index]
            self.decision_feed.insert(tk.END, f"{max(0, active_env.time - 1):02d}:00  No assignment: no ready orders or all drivers are busy.\n")
        self.decision_feed.configure(state=tk.DISABLED)
        self.decision_feed.see(tk.END)

    def _draw_all(self) -> None:
        for i, (canvas, env) in enumerate(zip(self.map_canvases, self.envs)):
            self._draw_map(canvas, env, i)
            self._update_player_stats(i, env)
        self._update_progress()

    def _draw_map(self, canvas: tk.Canvas, env: DeliveryEnvironment, player_index: int) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 380)
        height = max(canvas.winfo_height(), 320)
        for x in range(0, width, 40):
            canvas.create_line(x, 0, x, height, fill=self.COLORS["grid"])
        for y in range(0, height, 40):
            canvas.create_line(0, y, width, y, fill=self.COLORS["grid"])

        def coords(point):
            return (12 + point.x / env.config.width * (width - 24), 12 + point.y / env.config.height * (height - 24))

        for restaurant in env.restaurants:
            x, y = coords(restaurant.location)
            canvas.create_rectangle(x - 7, y - 7, x + 7, y + 7, fill=self.COLORS["orange"], outline="#fed7aa", width=2)
            canvas.create_text(x, y, text="R", fill="#431407", font=("TkDefaultFont", 7, "bold"))
            canvas.create_text(x, y - 13, text=f"R{restaurant.restaurant_id + 1}", fill="#ffedd5", font=("TkDefaultFont", 7, "bold"))

        queue_positions: dict[int, int] = {}
        for order in env.orders:
            if order.status not in (OrderStatus.PREPARING, OrderStatus.READY):
                continue
            x, y = coords(order.restaurant_location)
            position = queue_positions.get(order.restaurant_id, 0)
            queue_positions[order.restaurant_id] = position + 1
            angle = position * 0.85
            offset = 12 + (position // 8) * 5
            x += math.cos(angle) * offset
            y += math.sin(angle) * offset
            color = self.COLORS["purple"] if order.status == OrderStatus.PREPARING else self.COLORS["yellow"]
            radius = 4 if order.status == OrderStatus.PREPARING else 6
            canvas.create_polygon(x, y - radius, x + radius, y, x, y + radius, x - radius, y, fill=color, outline="#f8fafc")
            canvas.create_text(x + 7, y - 7, text=f"O{order.order_id}", anchor=tk.W, fill=self.COLORS["text"], font=("TkDefaultFont", 7, "bold"))

        for order in env.delivered_orders:
            if order.delivered_at is None or env.time - order.delivered_at > 3:
                continue
            x, y = coords(order.customer_location)
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=self.COLORS["green"], outline="#dcfce7")
            canvas.create_text(x, y, text="✓", fill="#052e16", font=("TkDefaultFont", 7, "bold"))

        for driver in env.drivers:
            x, y = coords(driver.location)
            if driver.destination is not None:
                dx, dy = coords(driver.destination)
                canvas.create_line(x, y, dx, dy, fill=self.player_colors[player_index], width=2, dash=(4, 3), arrow=tk.LAST)
                if driver.order_id is not None:
                    canvas.create_text(dx + 5, dy + 8, text=f"O{driver.order_id}", anchor=tk.W, fill=self.player_colors[player_index], font=("TkDefaultFont", 7, "bold"))
            available = driver.status == DriverStatus.AVAILABLE
            color = self.COLORS["green"] if available else self.player_colors[player_index]
            canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill=color, outline="#f8fafc", width=1)
            canvas.create_text(x, y, text="D", fill="#082f49", font=("TkDefaultFont", 7, "bold"))
            canvas.create_text(x + 7, y - 8, text=f"D{driver.driver_id + 1}", anchor=tk.W, fill="#f8fafc", font=("TkDefaultFont", 7, "bold"))

    def _update_player_stats(self, player_index: int, env: DeliveryEnvironment) -> None:
        delivered = env.delivered_orders
        average = sum(o.delivered_at - o.created_at for o in delivered if o.delivered_at is not None) / len(delivered) if delivered else None
        values = {
            "delivered": f"{len(delivered)} / {len(env.orders)}",
            "pending": str(len(env.pending_orders)),
            "avg_time": f"{average:.1f} min" if average is not None else "—",
            "travel": f"{env.total_travel_distance:.0f} units",
            "free": str(sum(d.status == DriverStatus.AVAILABLE for d in env.drivers)),
        }
        for key, value in values.items():
            self.player_stat_labels[player_index][key].configure(text=value)
        self.player_score_labels[player_index].configure(text=f"{env.total_reward:.0f} PTS")

    def _update_progress(self) -> None:
        if not hasattr(self, "progress") or not hasattr(self, "envs"):
            return
        self.progress.delete("all")
        width = max(self.progress.winfo_width(), 1)
        height = self.progress.winfo_height()
        active_env = self.envs[self.active_player_index]
        elapsed = active_env.time + (self._progress_fraction if self.running else 0.0)
        fraction = min(elapsed / self.MATCH_MINUTES, 1.0)
        self.progress.create_rectangle(0, 0, width, height, fill="#17263d", outline="")
        if fraction:
            self.progress.create_rectangle(0, 0, width * fraction, height, fill=self.COLORS["cyan"], outline="")
            self.progress.create_oval(max(1, width * fraction - 5), 2, width * fraction + 5, height - 2, fill="#e0fbff", outline="")
        elapsed_seconds = min(round(elapsed * 60), self.MATCH_MINUTES * 60)
        hours, rem = divmod(elapsed_seconds, 60)
        self.clock_label.configure(text=f"P{self.active_player_index + 1} · {hours:02d}:{rem:02d} / 60:00")

    def _results_for(self, env: DeliveryEnvironment) -> dict[str, float | int]:
        delivered = env.delivered_orders
        average = sum(o.delivered_at - o.created_at for o in delivered if o.delivered_at is not None) / len(delivered) if delivered else math.inf
        return {
            "score": env.total_reward,
            "delivered": len(delivered),
            "average": average,
            "distance": env.total_travel_distance,
        }

    def _show_results(self) -> None:
        if self._summary_shown:
            return
        self._summary_shown = True
        results = [self._results_for(env) for env in self.envs]
        winner = self._choose_winner(results[0], results[1])
        if winner is None:
            headline = "It's a draw!"
        else:
            headline = f"{self.player_names[winner]} wins!"
        lines = [headline, "", "Winner order: score, then faster average delivery, then shorter distance.", ""]
        for i, (name, env, result) in enumerate(zip(self.player_names, self.envs, results)):
            avg = f"{result['average']:.2f} min" if math.isfinite(float(result["average"])) else "no deliveries"
            lines.extend([
                f"{name} — {self.policies[i].name}",
                f"Score: {result['score']:.0f} points  |  Delivered: {result['delivered']} / {len(env.orders)}",
                f"Pending at cutoff: {len(env.pending_orders)}  |  Average delivery: {avg}",
                f"Travel distance: {env.total_travel_distance:.1f} units",
                "",
            ])
        lines.append("Orders still in preparation or travel at the 60-minute cutoff remain pending.")
        messagebox.showinfo("Dispatch duel results", "\n".join(lines), parent=self.root)

    @staticmethod
    def _choose_winner(first: dict[str, float | int], second: dict[str, float | int]) -> int | None:
        if first["score"] != second["score"]:
            return 0 if first["score"] > second["score"] else 1
        if first["average"] != second["average"]:
            return 0 if first["average"] < second["average"] else 1
        if first["distance"] != second["distance"]:
            return 0 if first["distance"] < second["distance"] else 1
        return None

    def _play_sound(self) -> None:
        if self.sound_enabled.get():
            self.root.bell()

    def _set_activity(self, text: str, color: str) -> None:
        self.activity_label.configure(text=text, fg=color)

    def _stop_timers(self) -> None:
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None
        self._cancel_animation()
        self._minute_started_at = None

    def _cancel_animation(self) -> None:
        if self._animation_id is not None:
            self.root.after_cancel(self._animation_id)
            self._animation_id = None
