"""Discrete-time simulation engine (1 tick = 1 simulated second, exactly 180 ticks).

Tick order (deterministic):
  1. t += 1
  2. spawn orders with created_at <= t
  3. orders whose preparation completed become READY
  4. drivers move (L-shaped path); leftover movement carries over waypoint to waypoint
  5. waypoint arrivals: pickup (if all food ready, else wait at restaurant) / delivery
  6. accounting (busy ticks, waiting/overdue/idle counts)
  7. at t == 180 the episode is finished; advance_tick() then raises.
The horizon is enforced here, in the engine itself.
"""
from __future__ import annotations

from environment.city import at_point, manhattan, move_towards, EPS
from environment.constants import DriverStatus, N_TICKS
from environment.driver import DROPOFF, PICKUP, Driver, Waypoint
from environment.estimates import dropoff_order
from environment.order import Order
from environment.restaurant import Restaurant
from environment.scenario import Scenario


class TickEvents:
    __slots__ = ("n_delivered", "n_on_time", "n_pickups", "distance", "n_waiting", "n_overdue",
                 "n_idle", "batch_saving")

    def __init__(self):
        self.n_delivered = 0
        self.n_on_time = 0
        self.n_pickups = 0
        self.distance = 0.0
        self.n_waiting = 0       # created, undelivered orders
        self.n_overdue = 0       # of which past their deadline
        self.n_idle = 0          # available drivers that could be serving an unassigned order
        self.batch_saving = 0.0


class Engine:
    def __init__(self, scenario: Scenario, record: bool = False):
        self.scenario = scenario
        self.cfg = scenario.config
        self.t = 0
        self.restaurants = [Restaurant(i, x, y) for i, (x, y) in enumerate(scenario.restaurants)]
        self.drivers = [Driver(i, x, y, self.cfg.capacity) for i, (x, y) in enumerate(scenario.driver_starts)]
        self.orders: list[Order] = []       # all spawned orders
        self.pending: list[Order] = []      # spawned, unassigned (creation order)
        self.preparing: list[Order] = []
        self.open_orders: list[Order] = []  # spawned, undelivered
        self.batches: list[dict] = []
        self._next_spec = 0
        self.record = record
        self.event_log: list[str] = []
        self._spawn()

    # ------------------------------------------------------------------ properties
    @property
    def finished(self) -> bool:
        return self.t >= N_TICKS

    # ------------------------------------------------------------------ spawning
    def _spawn(self):
        specs = self.scenario.orders
        while self._next_spec < len(specs) and specs[self._next_spec].created_at <= self.t:
            s = specs[self._next_spec]
            rest = self.restaurants[s.restaurant_id]
            o = Order(self._next_spec, s.restaurant_id, rest.x, rest.y, s.cx, s.cy,
                      s.created_at, s.prep_actual, s.prep_est, s.deadline)
            self._next_spec += 1
            self.orders.append(o)
            self.pending.append(o)
            self.preparing.append(o)
            self.open_orders.append(o)
            if self.record:
                self.event_log.append(f"Order {o.id} placed at restaurant {o.restaurant_id}")

    # ------------------------------------------------------------------ decisions
    def can_assign(self, d: Driver, o: Order) -> bool:
        if o.assigned_at is not None:
            return False
        if d.status == DriverStatus.AVAILABLE:
            return True
        if (self.cfg.batching and d.status in (DriverStatus.TO_RESTAURANT, DriverStatus.AT_RESTAURANT)
                and d.orders and d.load < d.capacity
                and all(x.restaurant_id == o.restaurant_id for x in d.orders)):
            return True
        return False

    def assign(self, d: Driver, o: Order):
        if not self.can_assign(d, o):
            raise ValueError(f"invalid assignment driver {d.id} -> order {o.id}")
        o.assigned_at = self.t
        o.driver_id = d.id
        self.pending.remove(o)
        if d.status == DriverStatus.AVAILABLE:
            d.orders = [o]
            d.route = [Waypoint(o.rx, o.ry, PICKUP, o), Waypoint(o.cx, o.cy, DROPOFF, o)]
            d.status = DriverStatus.TO_RESTAURANT
        else:
            o1 = d.orders[0]
            d.orders.append(o)
            seq = dropoff_order(o.rx, o.ry, [o1, o])
            d.route = ([Waypoint(o.rx, o.ry, PICKUP, o1), Waypoint(o.rx, o.ry, PICKUP, o)]
                       + [Waypoint(x.cx, x.cy, DROPOFF, x) for x in seq])
        if self.record:
            self.event_log.append(f"Driver {d.id} assigned to order {o.id}")

    # ------------------------------------------------------------------ time
    def advance_tick(self) -> TickEvents:
        if self.finished:
            raise RuntimeError("episode finished: the 3-minute horizon has been reached")
        self.t += 1
        t = self.t
        ev = TickEvents()
        self._spawn()

        if self.preparing:
            still = []
            for o in self.preparing:
                if o.created_at + o.prep_actual <= t:
                    o.ready_at = t
                else:
                    still.append(o)
            self.preparing = still

        budget = self.cfg.base_speed * float(self.scenario.traffic[t - 1])
        for d in self.drivers:
            if d.status != DriverStatus.AVAILABLE:
                d.busy_ticks += 1
                self._drive(d, budget, ev)

        ev.n_waiting = len(self.open_orders)
        ev.n_overdue = sum(1 for o in self.open_orders if t > o.deadline)
        n_avail = sum(1 for d in self.drivers if d.status == DriverStatus.AVAILABLE)
        ev.n_idle = min(n_avail, len(self.pending))
        return ev

    def _drive(self, d: Driver, budget: float, ev: TickEvents):
        t = self.t
        while True:
            if not d.route:
                d.status = DriverStatus.AVAILABLE
                return
            wp = d.route[0]
            if not at_point(d.x, d.y, wp.x, wp.y):
                if budget <= EPS:
                    return
                d.x, d.y, used = move_towards(d.x, d.y, wp.x, wp.y, budget)
                d.distance += used
                ev.distance += used
                budget -= used
                if not at_point(d.x, d.y, wp.x, wp.y):
                    return
            if wp.kind == PICKUP:
                group = []
                for w in d.route:
                    if w.kind != PICKUP:
                        break
                    group.append(w)
                for w in group:
                    if w.order.arrived_at_rest is None:
                        w.order.arrived_at_rest = t
                if all(w.order.ready_at is not None for w in group):
                    for w in group:
                        w.order.picked_up_at = t
                    del d.route[:len(group)]
                    d.status = DriverStatus.TO_CUSTOMER
                    ev.n_pickups += len(group)
                    if len(group) == 2:
                        self._record_batch(d, group[0].order, group[1].order, ev)
                    if self.record:
                        self.event_log.append(f"Driver {d.id} picked up " +
                                              ", ".join(str(w.order.id) for w in group))
                else:
                    d.status = DriverStatus.AT_RESTAURANT
                    return
            else:
                o = wp.order
                o.delivered_at = t
                d.route.pop(0)
                d.orders.remove(o)
                self.open_orders.remove(o)
                ev.n_delivered += 1
                late = t > o.deadline
                if not late:
                    ev.n_on_time += 1
                if self.record:
                    self.event_log.append(f"Order {o.id} delivered by driver {d.id}" + (" (LATE)" if late else ""))
                if not d.route:
                    d.status = DriverStatus.AVAILABLE
                    return

    def _record_batch(self, d: Driver, o1: Order, o2: Order, ev: TickEvents):
        rx, ry = o1.rx, o1.ry
        a, b = dropoff_order(rx, ry, [o1, o2])
        shared = manhattan(rx, ry, a.cx, a.cy) + manhattan(a.cx, a.cy, b.cx, b.cy)
        separate = manhattan(rx, ry, o1.cx, o1.cy) + manhattan(rx, ry, o2.cx, o2.cy)
        saved = separate - shared
        self.batches.append({"orders": (o1, o2), "saved": saved})
        ev.batch_saving += max(0.0, saved)

    # ------------------------------------------------------------------ recording
    def pop_log(self) -> list:
        out, self.event_log = self.event_log, []
        return out

    def snapshot(self) -> dict:
        t = self.t
        return {
            "t": t,
            "drivers": [{"id": d.id, "x": d.x, "y": d.y, "status": int(d.status), "load": d.load,
                         "route": [(w.x, w.y, w.kind) for w in d.route]} for d in self.drivers],
            "orders": [{"id": o.id, "rx": o.rx, "ry": o.ry, "cx": o.cx, "cy": o.cy,
                        "status": int(o.status), "driver": o.driver_id,
                        "late": (o.delivered_at is None and t > o.deadline) or o.is_late()}
                       for o in self.orders],
            "n_delivered": sum(1 for o in self.orders if o.delivered_at is not None),
            "log": self.pop_log(),
        }
