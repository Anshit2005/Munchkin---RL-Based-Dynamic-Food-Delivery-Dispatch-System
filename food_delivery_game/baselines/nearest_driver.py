"""Choose the currently closest driver/order pair by pickup distance."""

from .base import Assignment


class NearestDriverPolicy:
    name = "Nearest Driver"

    def select_assignment(self, env):
        available = [d for d in env.drivers if d.status.value == "Available"]
        ready = [o for o in env.orders if o.status.value == "Ready"]
        if not available or not ready:
            return None
        driver, order = min(
            ((d, o) for d in available for o in ready),
            key=lambda pair: (pair[0].location.distance_to(pair[1].restaurant_location), pair[1].order_id, pair[0].driver_id),
        )
        distance = driver.location.distance_to(order.restaurant_location)
        return Assignment(driver.driver_id, order.order_id, f"Closest pickup pair: {distance:.1f} distance units")
