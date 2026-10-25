"""Serve the most urgent dispatchable order, using the closest available driver."""

from .base import Assignment, estimated_completion_minutes


class OldestOrderPolicy:
    name = "Oldest / Most Urgent"

    def select_assignment(self, env):
        available = [d for d in env.drivers if d.status.value == "Available"]
        orders = env.dispatchable_orders
        if not available or not orders:
            return None
        best_eta = {
            order.order_id: min(estimated_completion_minutes(env, driver, order) for driver in available)
            for order in orders
        }
        order = min(orders, key=lambda o: (
            o.deadline_at - env.time - best_eta[o.order_id], o.created_at, o.order_id
        ))
        driver = min(available, key=lambda d: (
            estimated_completion_minutes(env, d, order), d.driver_id
        ))
        waiting = env.time - order.created_at
        slack = order.deadline_at - env.time - best_eta[order.order_id]
        return Assignment(
            driver.driver_id,
            order.order_id,
            f"Most urgent order (estimated deadline slack {slack:.1f} min); "
            f"delivery ETA {estimated_completion_minutes(env, driver, order):.2f} min "
            f"after waiting {waiting} min",
        )
