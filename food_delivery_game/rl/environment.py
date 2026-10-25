"""Action-level interface over the project's minute-based delivery simulator."""

from __future__ import annotations

from .state import MAX_ORDERS, WAIT_ACTION, encode_state
from ..environment import DeliveryEnvironment


class RLDeliveryEnvironment:
    """One assignment or one minute of simulated time is a single RL transition.

    Assignment actions have zero immediate reward. WAIT advances the simulator by one minute
    and returns its delivery reward. This lets the agent dispatch several drivers during the
    same minute, matching the heuristic policy interface.
    """

    def __init__(self, config=None):
        self.sim = DeliveryEnvironment(config)
        self._observation = None
        self._mask = None
        self._orders = None

    @property
    def config(self):
        return self.sim.config

    @property
    def done(self):
        return self.sim.done

    def reset(self, seed: int | None = None):
        self.sim.reset(seed)
        self.sim.prepare_minute()
        return self._refresh()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode is complete; call reset() before stepping again")
        if self._mask is None:
            self._refresh()
        action = int(action)
        reward = 0.0
        if action == WAIT_ACTION:
            result = self.sim.advance_minute()
            reward = result.reward
            done = result.done
            if not done:
                self.sim.prepare_minute()
        elif 0 <= action < WAIT_ACTION and self._mask[action]:
            driver_index, order_slot = divmod(action, MAX_ORDERS)
            order = self._orders[order_slot]
            driver_id = self.sim.drivers[driver_index].driver_id
            if not self.sim.assign_order(driver_id, order.order_id):
                raise RuntimeError("action mask and simulator assignment validation disagreed")
            done = False
        else:
            raise ValueError(f"invalid or masked action: {action}")
        observation, mask, orders = self._refresh() if not done else (self._observation, self._mask, self._orders)
        return observation, reward, done, {"action_mask": mask, "ready_orders": len(orders)}

    def _refresh(self):
        self._observation, self._mask, self._orders = encode_state(self.sim)
        return self._observation, self._mask, {"action_mask": self._mask, "ready_orders": len(self._orders)}

