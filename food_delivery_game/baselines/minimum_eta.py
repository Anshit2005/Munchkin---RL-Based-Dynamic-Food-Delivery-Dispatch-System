"""Choose the feasible pair with the lowest estimated arrival time."""

from .base import Assignment


class MinimumETAPolicy:
    name = "Minimum Estimated Delivery Time"

    def select_assignment(self, env):
        available = [d for d in env.drivers if d.status.value == "Available"]
        ready = [o for o in env.orders if o.status.value == "Ready"]
        if not available or not ready:
            return None

        def estimated_minutes(pair):
            driver, order = pair
            pickup = driver.location.distance_to(order.restaurant_location)
            delivery = order.restaurant_location.distance_to(order.customer_location)
            return (pickup + delivery) / env.config.travel_speed

        driver, order = min(
            ((d, o) for d in available for o in ready),
            key=lambda pair: (estimated_minutes(pair), pair[1].order_id, pair[0].driver_id),
        )
        return Assignment(driver.driver_id, order.order_id)
