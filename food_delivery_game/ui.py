"""Vibrant Tkinter game interface for the delivery simulator."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk

from .baselines import POLICIES, create_policy
from .config import SimulationConfig
from .environment import DeliveryEnvironment
from .entities import DriverStatus, OrderStatus
from .simulation.runner import average_metrics, compare_policies


class DeliveryGame:
    MAP_WIDTH = 700
    MAP_HEIGHT = 400
    MINUTE_INTERVAL_MS = 5000
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

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.configure(background=self.COLORS["background"])
        self.running = False
        self._after_id: str | None = None
        self._animation_id: str | None = None
        self._minute_started_at: float | None = None
        self._progress_fraction = 0.0
        self._summary_shown = False
        self.sound_enabled = tk.BooleanVar(value=True)
        self._configure_theme()
        self._build_header()
        self._build_controls()
        self._build_game_area()
        self.reset_simulation()

    def _configure_theme(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", background="#243650", foreground=self.COLORS["text"], padding=(10, 7), borderwidth=0)
        style.map("TButton", background=[("active", "#345071"), ("disabled", "#1a2638")])
        style.configure("TCombobox", fieldbackground="#17263d", background="#243650", foreground=self.COLORS["text"], arrowcolor=self.COLORS["cyan"])
        style.configure("TCheckbutton", background=self.COLORS["panel"], foreground=self.COLORS["text"])
        style.map("TCheckbutton", background=[("active", self.COLORS["panel"])])

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=self.COLORS["background"], padx=18, pady=12)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(header, text="MUNCHKIN", bg=self.COLORS["background"], fg=self.COLORS["cyan"], font=("TkDefaultFont", 21, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="  FOOD DISPATCH ARENA", bg=self.COLORS["background"], fg=self.COLORS["muted"], font=("TkDefaultFont", 11, "bold")).pack(side=tk.LEFT, pady=(6, 0))
        self.score_label = tk.Label(header, text="★ 0 PTS", bg="#243650", fg=self.COLORS["yellow"], font=("TkDefaultFont", 16, "bold"), padx=14, pady=6)
        self.score_label.pack(side=tk.RIGHT)

    def _build_controls(self) -> None:
        panel = tk.Frame(self.root, bg=self.COLORS["panel"], padx=14, pady=14, width=245)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 10), pady=(0, 12))
        panel.pack_propagate(False)
        tk.Label(panel, text="SCENARIO SETUP", bg=self.COLORS["panel"], fg=self.COLORS["text"], font=("TkDefaultFont", 13, "bold")).pack(anchor="w", pady=(0, 12))
        self.restaurant_count = self._add_scale(panel, "Restaurants", 1, 10, 5, self.COLORS["orange"])
        self.driver_count = self._add_scale(panel, "Delivery partners", 1, 16, 8, self.COLORS["green"])
        self.order_rate = self._add_scale(panel, "Orders per minute", 0, 12, 5, self.COLORS["pink"])
        tk.Label(panel, text="⏱  Fixed episode: 3 simulated minutes", bg=self.COLORS["panel"], fg=self.COLORS["muted"], font=("TkDefaultFont", 9)).pack(anchor="w", pady=(0, 10))

        tk.Label(panel, text="DISPATCH STRATEGY", bg=self.COLORS["panel"], fg=self.COLORS["text"], font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(3, 4))
        self.policy_selector = ttk.Combobox(panel, values=list(POLICIES), state="readonly", width=29)
        self.policy_selector.current(0)
        self.policy_selector.pack(anchor="w", fill=tk.X, pady=(0, 9))

        ttk.Button(panel, text="▶  Start / Resume", command=self.toggle_run).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="⏭  Step one minute", command=self.step_once).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="↻  Reset scenario", command=self.reset_simulation).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="★  Compare baselines", command=self.show_comparison).pack(fill=tk.X, pady=(3, 8))
        ttk.Checkbutton(panel, text="Enable game sounds", variable=self.sound_enabled).pack(anchor="w", pady=(2, 4))
        tk.Label(panel, text="+10 reward points per delivery", bg=self.COLORS["panel"], fg=self.COLORS["yellow"], font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(5, 0))
        tk.Label(panel, text="PPO agent will be added in the next stage.", bg=self.COLORS["panel"], fg=self.COLORS["muted"], wraplength=210, justify=tk.LEFT, font=("TkDefaultFont", 9)).pack(anchor="w", pady=(14, 0))

    def _add_scale(self, parent: tk.Frame, label: str, low: int, high: int, initial: int, color: str) -> tk.Scale:
        tk.Label(parent, text=label, bg=self.COLORS["panel"], fg=self.COLORS["muted"], font=("TkDefaultFont", 9)).pack(anchor="w")
        scale = tk.Scale(
            parent, from_=low, to=high, orient=tk.HORIZONTAL, length=205,
            bg=self.COLORS["panel"], fg=self.COLORS["text"], troughcolor="#26364e",
            activebackground=color, highlightthickness=0, bd=0, showvalue=True,
            font=("TkDefaultFont", 9),
        )
        scale.set(initial)
        scale.pack(anchor="w", pady=(0, 7))
        return scale

    def _build_game_area(self) -> None:
        area = tk.Frame(self.root, bg=self.COLORS["background"], padx=(0), pady=(0))
        area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12), pady=(0, 12))
        heading = tk.Frame(area, bg=self.COLORS["background"])
        heading.pack(fill=tk.X, pady=(0, 7))
        tk.Label(heading, text="LIVE DELIVERY MAP", bg=self.COLORS["background"], fg=self.COLORS["text"], font=("TkDefaultFont", 13, "bold")).pack(side=tk.LEFT)
        self.activity_label = tk.Label(heading, text="● READY", bg=self.COLORS["background"], fg=self.COLORS["green"], font=("TkDefaultFont", 10, "bold"))
        self.activity_label.pack(side=tk.RIGHT)

        self.stat_labels: dict[str, tk.Label] = {}
        stats = tk.Frame(area, bg=self.COLORS["background"])
        stats.pack(fill=tk.X, pady=(0, 10))
        cards = [
            ("ORDERS", "0", self.COLORS["purple"]),
            ("DELIVERED", "0", self.COLORS["green"]),
            ("AVG TIME", "—", self.COLORS["cyan"]),
            ("TRAVEL", "0", self.COLORS["orange"]),
            ("DRIVERS FREE", "0", self.COLORS["pink"]),
        ]
        for title, initial, accent in cards:
            card = tk.Frame(stats, bg=self.COLORS["card"], padx=9, pady=7)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
            tk.Label(card, text=title, bg=self.COLORS["card"], fg=accent, font=("TkDefaultFont", 8, "bold")).pack(anchor="w")
            value = tk.Label(card, text=initial, bg=self.COLORS["card"], fg=self.COLORS["text"], font=("TkDefaultFont", 13, "bold"))
            value.pack(anchor="w", pady=(2, 0))
            self.stat_labels[title] = value

        clock_row = tk.Frame(area, bg=self.COLORS["background"])
        clock_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(clock_row, text="SIMULATION CLOCK", bg=self.COLORS["background"], fg=self.COLORS["muted"], font=("TkDefaultFont", 8, "bold")).pack(side=tk.LEFT)
        self.clock_label = tk.Label(clock_row, text="00:00 / 03:00", bg=self.COLORS["background"], fg=self.COLORS["cyan"], font=("TkDefaultFont", 9, "bold"))
        self.clock_label.pack(side=tk.RIGHT)
        self.progress = tk.Canvas(area, height=16, bg="#17263d", highlightthickness=0)
        self.progress.pack(fill=tk.X, pady=(0, 9))
        self.progress.bind("<Configure>", lambda _event: self._update_progress())

        self.canvas = tk.Canvas(area, width=self.MAP_WIDTH, height=self.MAP_HEIGHT, background=self.COLORS["map"], highlightthickness=1, highlightbackground="#31445f")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        legend = tk.Frame(area, bg=self.COLORS["background"])
        legend.pack(fill=tk.X, pady=(7, 0))
        for mark, caption, color in (
            ("■", "Restaurant", self.COLORS["orange"]),
            ("●", "Available driver", self.COLORS["green"]),
            ("●", "Busy driver", self.COLORS["cyan"]),
            ("◆", "Ready order", self.COLORS["pink"]),
            ("○", "Preparing order", self.COLORS["purple"]),
        ):
            tk.Label(legend, text=mark, bg=self.COLORS["background"], fg=color, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=(0, 3))
            tk.Label(legend, text=caption, bg=self.COLORS["background"], fg=self.COLORS["muted"], font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(0, 11))

    def _make_config(self) -> SimulationConfig:
        return SimulationConfig(
            restaurant_count=int(self.restaurant_count.get()),
            driver_count=int(self.driver_count.get()),
            orders_per_minute=float(self.order_rate.get()),
        )

    def reset_simulation(self) -> None:
        self._stop_timer()
        self._progress_fraction = 0.0
        self.running = False
        self._summary_shown = False
        self.env = DeliveryEnvironment(self._make_config())
        self._set_activity("● READY", self.COLORS["green"])
        self._draw()

    def _selected_policy(self):
        return create_policy(self.policy_selector.get())

    def toggle_run(self) -> None:
        if self.env.done:
            self.reset_simulation()
        self.running = not self.running
        if self.running:
            self._set_activity(f"● LIVE  {self.policy_selector.get()}", self.COLORS["green"])
            self._schedule_step()
        else:
            self._stop_timer()
            self._set_activity("Ⅱ PAUSED", self.COLORS["yellow"])

    def step_once(self) -> None:
        self._stop_timer()
        self.running = False
        if self.env.done:
            self._set_activity("★ EPISODE COMPLETE", self.COLORS["yellow"])
            return
        self._progress_fraction = 0.0
        self._set_activity("⏭ STEP", self.COLORS["cyan"])
        self._advance_one_minute()

    def show_comparison(self) -> None:
        self._stop_timer()
        self.running = False
        self._set_activity("Ⅱ PAUSED FOR COMPARISON", self.COLORS["yellow"])
        results = compare_policies(self._make_config(), episodes=20)
        lines = ["Matched comparison over 20 seeds", "3 simulated minutes per episode", ""]
        for name, episodes in results.items():
            delivered, reward, avg_time, distance = average_metrics(episodes)
            time_text = f"{avg_time:.2f} min" if delivered else "no deliveries"
            lines.append(f"{name}\n  Delivered: {delivered:.2f}   Reward: {reward:.0f} pts\n  Avg delivery: {time_text}   Travel: {distance:.1f}")
        messagebox.showinfo("Baseline comparison", "\n\n".join(lines), parent=self.root)

    def _schedule_step(self) -> None:
        if self.running:
            self._minute_started_at = time.monotonic() - self._progress_fraction * self.MINUTE_INTERVAL_MS / 1000
            remaining_ms = max(1, round(self.MINUTE_INTERVAL_MS * (1 - self._progress_fraction)))
            self._after_id = self.root.after(remaining_ms, self._advance)
            self._schedule_progress_animation()

    def _schedule_progress_animation(self) -> None:
        if self.running:
            self._animation_id = self.root.after(80, self._animate_progress)

    def _animate_progress(self) -> None:
        self._animation_id = None
        if not self.running or self._minute_started_at is None:
            return
        elapsed = (time.monotonic() - self._minute_started_at) * 1000
        self._progress_fraction = min(elapsed / self.MINUTE_INTERVAL_MS, 1.0)
        self._update_progress()
        self._schedule_progress_animation()

    def _advance(self) -> None:
        self._after_id = None
        self._cancel_progress_animation()
        self._minute_started_at = None
        if self.running and not self.env.done:
            self._progress_fraction = 0.0
            self._advance_one_minute()
        if self.env.done:
            self.running = False
            return
        self._schedule_step()

    def _advance_one_minute(self) -> None:
        if self.env.done:
            return
        before = len(self.env.delivered_orders)
        self.env.step(self._selected_policy())
        self._draw()
        newly_delivered = len(self.env.delivered_orders) - before
        if newly_delivered:
            self._play_sound()
        if self.env.done:
            self.running = False
            self._set_activity("★ EPISODE COMPLETE", self.COLORS["yellow"])
            self._play_sound()
            self._show_final_summary()
        elif self.running:
            self._set_activity(f"● LIVE  {self.policy_selector.get()}", self.COLORS["green"])
        else:
            self._set_activity("● READY FOR NEXT MINUTE", self.COLORS["cyan"])

    def _play_sound(self) -> None:
        if self.sound_enabled.get():
            self.root.bell()

    def _set_activity(self, text: str, color: str) -> None:
        self.activity_label.configure(text=text, fg=color)

    def _stop_timer(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._cancel_progress_animation()
        self._minute_started_at = None

    def _cancel_progress_animation(self) -> None:
        if self._animation_id is not None:
            self.root.after_cancel(self._animation_id)
            self._animation_id = None

    def _coords(self, point) -> tuple[float, float]:
        x = 16 + point.x / self.env.config.width * (self.MAP_WIDTH - 32)
        y = 16 + point.y / self.env.config.height * (self.MAP_HEIGHT - 32)
        return x, y

    def _update_progress(self) -> None:
        if not hasattr(self, "progress") or not hasattr(self, "env"):
            return
        self.progress.delete("all")
        width = max(self.progress.winfo_width(), 1)
        height = max(self.progress.winfo_height(), 1)
        elapsed_minutes = self.env.time + self._progress_fraction
        fraction = min(elapsed_minutes / self.env.config.episode_minutes, 1.0)
        self.progress.create_rectangle(0, 0, width, height, fill="#17263d", outline="")
        if fraction > 0:
            self.progress.create_rectangle(0, 0, width * fraction, height, fill=self.COLORS["cyan"], outline="")
            self.progress.create_oval(max(1, width * fraction - 6), 2, width * fraction + 5, height - 2, fill="#e0fbff", outline="")
        elapsed_seconds = min(round(elapsed_minutes * 60), self.env.config.episode_minutes * 60)
        minutes, seconds = divmod(elapsed_seconds, 60)
        self.clock_label.configure(text=f"{minutes:02d}:{seconds:02d} / 03:00")

    def _update_stats(self) -> None:
        summary = self.env.state_summary()
        delivered = self.env.delivered_orders
        avg_time = sum(o.delivered_at - o.created_at for o in delivered if o.delivered_at is not None) / len(delivered) if delivered else None
        values = {
            "ORDERS": str(summary["orders"]),
            "DELIVERED": f"{summary['delivered']}  /  {summary['orders']}",
            "AVG TIME": f"{avg_time:.1f}m" if avg_time is not None else "—",
            "TRAVEL": f"{self.env.total_travel_distance:.0f} u",
            "DRIVERS FREE": str(summary["available_drivers"]),
        }
        for name, value in values.items():
            self.stat_labels[name].configure(text=value)
        self.score_label.configure(text=f"★ {self.env.total_reward:.0f} PTS")
        self._update_progress()

    def _draw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), self.MAP_WIDTH)
        height = max(self.canvas.winfo_height(), self.MAP_HEIGHT)
        for x in range(0, width, 50):
            self.canvas.create_line(x, 0, x, height, fill=self.COLORS["grid"], width=1)
        for y in range(0, height, 50):
            self.canvas.create_line(0, y, width, y, fill=self.COLORS["grid"], width=1)

        for restaurant in self.env.restaurants:
            x, y = self._coords(restaurant.location)
            self.canvas.create_rectangle(x - 9, y - 9, x + 9, y + 9, fill=self.COLORS["orange"], outline="#fed7aa", width=2)
            self.canvas.create_text(x, y, text="R", fill="#431407", font=("TkDefaultFont", 8, "bold"))
            self.canvas.create_text(x, y - 17, text=f"{restaurant.restaurant_id + 1}", fill="#ffedd5", font=("TkDefaultFont", 8, "bold"))

        for order in self.env.orders:
            if order.status in (OrderStatus.DELIVERED, OrderStatus.TO_RESTAURANT, OrderStatus.TO_CUSTOMER):
                continue
            x, y = self._coords(order.customer_location)
            color = self.COLORS["pink"] if order.status == OrderStatus.READY else self.COLORS["purple"]
            radius = 7 if order.status == OrderStatus.READY else 5
            self.canvas.create_polygon(x, y - radius, x + radius, y, x, y + radius, x - radius, y, fill=color, outline="#fce7f3", width=1)

        for driver in self.env.drivers:
            x, y = self._coords(driver.location)
            if driver.destination is not None:
                dx, dy = self._coords(driver.destination)
                self.canvas.create_line(x, y, dx, dy, fill=self.COLORS["cyan"], width=2, dash=(5, 4), arrow=tk.LAST)
            available = driver.status == DriverStatus.AVAILABLE
            color = self.COLORS["green"] if available else self.COLORS["cyan"]
            self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill=color, outline="#e0f2fe", width=2)
            self.canvas.create_text(x, y, text=str(driver.driver_id + 1), fill="#082f49" if not available else "#052e16", font=("TkDefaultFont", 7, "bold"))

        self._update_stats()

    def _show_final_summary(self) -> None:
        if self._summary_shown:
            return
        self._summary_shown = True
        summary = self.env.state_summary()
        delivered = self.env.delivered_orders
        average_time = sum(o.delivered_at - o.created_at for o in delivered if o.delivered_at is not None) / len(delivered) if delivered else 0.0
        delivery_rate = 100 * len(delivered) / len(self.env.orders) if self.env.orders else 0.0
        avg_text = f"{average_time:.2f} simulated minutes" if delivered else "No deliveries completed"
        details = (
            f"Final reward: {self.env.total_reward:.0f} points\n\n"
            f"Orders generated: {len(self.env.orders)}\n"
            f"Orders delivered: {len(delivered)} ({delivery_rate:.1f}%)\n"
            f"Orders still pending: {len(self.env.pending_orders)}\n"
            f"Average delivery time: {avg_text}\n"
            f"Total travel distance: {self.env.total_travel_distance:.1f}\n"
            f"Drivers available at finish: {summary['available_drivers']}\n"
            f"Strategy: {self.policy_selector.get()}"
        )
        messagebox.showinfo("Episode results", details, parent=self.root)
