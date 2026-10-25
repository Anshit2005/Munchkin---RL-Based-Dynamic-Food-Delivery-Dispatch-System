"""Choose the feasible pair with the lowest estimated arrival time."""

from .base import Assignment, estimated_completion_minutes


class MinimumETAPolicy:
    name = "Minimum Estimated Delivery Time"

    def select_assignment(self, env):
        available = [d for d in env.drivers if d.status.value == "Available"]
        orders = env.dispatchable_orders
        if not available or not orders:
            return None
        driver, order = min(
            ((d, o) for d in available for o in orders),
            key=lambda pair: (estimated_completion_minutes(env, pair[0], pair[1]),
                              pair[1].order_id, pair[0].driver_id),
        )
        eta = estimated_completion_minutes(env, driver, order)
        return Assignment(driver.driver_id, order.order_id, f"Lowest estimated delivery time: {eta:.2f} min")
