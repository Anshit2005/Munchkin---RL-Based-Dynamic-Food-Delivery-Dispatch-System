"""Tkinter game-style visualization for the delivery simulator."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .config import SimulationConfig
from .environment import DeliveryEnvironment
from .entities import DriverStatus, OrderStatus


class DeliveryGame:
    MAP_WIDTH = 650
    MAP_HEIGHT = 455

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.running = False
        self._after_id: str | None = None
        self._build_controls()
        self._build_map()
        self._build_status()
        self.reset_simulation()

    def _build_controls(self) -> None:
        panel = ttk.Frame(self.root, padding=12)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(panel, text="Scenario setup", font=("TkDefaultFont", 13, "bold")).pack(anchor="w", pady=(0, 10))
        self.restaurant_count = self._add_scale(panel, "Restaurants", 1, 10, 5)
        self.driver_count = self._add_scale(panel, "Delivery partners", 1, 16, 8)
        self.order_rate = self._add_scale(panel, "Orders per minute", 0, 12, 5, resolution=1)
        self.travel_speed = self._add_scale(panel, "Travel speed", 8, 60, 24)
        ttk.Label(panel, text="Episode length: fixed at 3 simulated minutes").pack(anchor="w", pady=(8, 12))
        ttk.Button(panel, text="Reset scenario", command=self.reset_simulation).pack(fill=tk.X, pady=3)
        self.run_button = ttk.Button(panel, text="Start", command=self.toggle_run)
        self.run_button.pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="Step 1 minute", command=self.step_once).pack(fill=tk.X, pady=3)
        ttk.Separator(panel).pack(fill=tk.X, pady=12)
        ttk.Label(panel, text="Dispatch policy: setup rule", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(panel, text="Baselines and PPO are added in later stages.", wraplength=205).pack(anchor="w", pady=(4, 0))

    def _add_scale(self, parent: ttk.Frame, label: str, low: int, high: int, initial: int, resolution: int = 1) -> tk.Scale:
        ttk.Label(parent, text=label).pack(anchor="w")
        scale = tk.Scale(parent, from_=low, to=high, orient=tk.HORIZONTAL, resolution=resolution, length=205)
        scale.set(initial)
        scale.pack(anchor="w", pady=(0, 8))
        return scale

    def _build_map(self) -> None:
        area = ttk.Frame(self.root, padding=(0, 12, 12, 8))
        area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(area, text="Delivery map", font=("TkDefaultFont", 13, "bold")).pack(anchor="w", pady=(0, 8))
        self.canvas = tk.Canvas(area, width=self.MAP_WIDTH, height=self.MAP_HEIGHT, background="#f5f7fa", highlightthickness=1, highlightbackground="#cbd5e1")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        legend = ttk.Label(area, text="■ Restaurant    ● Available driver    ● Busy driver    ○ Preparing order    ◆ Ready order")
        legend.pack(anchor="w", pady=(8, 0))

    def _build_status(self) -> None:
        self.status = ttk.Label(self.root, text="", padding=(12, 4), justify=tk.LEFT)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _make_config(self) -> SimulationConfig:
        return SimulationConfig(
            restaurant_count=int(self.restaurant_count.get()),
            driver_count=int(self.driver_count.get()),
            orders_per_minute=float(self.order_rate.get()),
            travel_speed=float(self.travel_speed.get()),
        )

    def reset_simulation(self) -> None:
        self._stop_timer()
        self.running = False
        self.run_button.configure(text="Start")
        self.env = DeliveryEnvironment(self._make_config())
        self._draw()

    def toggle_run(self) -> None:
        if self.env.done:
            self.reset_simulation()
        self.running = not self.running
        self.run_button.configure(text="Pause" if self.running else "Start")
        if self.running:
            self._schedule_step()

    def step_once(self) -> None:
        self._stop_timer()
        self.running = False
        self.run_button.configure(text="Start")
        if not self.env.done:
            self.env.step()
        self._draw()

    def _schedule_step(self) -> None:
        if self.running:
            self._after_id = self.root.after(650, self._advance)

    def _advance(self) -> None:
        self._after_id = None
        if self.running and not self.env.done:
            self.env.step()
            self._draw()
        if self.env.done:
            self.running = False
            self.run_button.configure(text="Start")
        else:
            self._schedule_step()

    def _stop_timer(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _coords(self, point) -> tuple[float, float]:
        x = 18 + point.x / self.env.config.width * (self.MAP_WIDTH - 36)
        y = 18 + point.y / self.env.config.height * (self.MAP_HEIGHT - 36)
        return x, y

    def _draw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(8, 8, self.MAP_WIDTH - 8, self.MAP_HEIGHT - 8, outline="#d8dee9")
        for restaurant in self.env.restaurants:
            x, y = self._coords(restaurant.location)
            self.canvas.create_rectangle(x - 7, y - 7, x + 7, y + 7, fill="#e76f51", outline="")
            self.canvas.create_text(x, y - 15, text=f"R{restaurant.restaurant_id + 1}", fill="#7f1d1d", font=("TkDefaultFont", 8))
        for order in self.env.orders:
            if order.status in (OrderStatus.DELIVERED, OrderStatus.TO_RESTAURANT, OrderStatus.TO_CUSTOMER):
                continue
            x, y = self._coords(order.customer_location)
            color = "#f4a261" if order.status == OrderStatus.READY else "#94a3b8"
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="")
        for driver in self.env.drivers:
            x, y = self._coords(driver.location)
            color = "#2a9d8f" if driver.status == DriverStatus.AVAILABLE else "#457b9d"
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, outline="white", width=1)
        summary = self.env.state_summary()
        self.status.configure(text=(
            f"Time: {summary['time']} / 3 min    Orders: {summary['orders']}    "
            f"Preparing: {summary['preparing']}    Ready: {summary['ready']}    "
            f"In transit: {summary['in_transit']}    Delivered: {summary['delivered']}    "
            f"Available drivers: {summary['available_drivers']}"
        ))
