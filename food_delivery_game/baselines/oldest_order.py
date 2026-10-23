"""Serve the oldest ready order, using the closest available driver."""

from .base import Assignment


class OldestOrderPolicy:
    name = "Oldest / Most Urgent"

    def select_assignment(self, env):
        available = [d for d in env.drivers if d.status.value == "Available"]
        ready = [o for o in env.orders if o.status.value == "Ready"]
        if not available or not ready:
            return None
        order = min(ready, key=lambda o: (o.created_at, o.order_id))
        driver = min(
            available,
            key=lambda d: (d.location.distance_to(order.restaurant_location), d.driver_id),
        )
        return Assignment(driver.driver_id, order.order_id)
