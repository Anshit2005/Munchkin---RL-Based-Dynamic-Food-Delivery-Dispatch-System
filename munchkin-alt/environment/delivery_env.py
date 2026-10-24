"""DeliveryEnv: the MDP (Gymnasium API) over the 3-minute simulator.

Step semantics: assignment actions do not advance the clock; WAIT (or an invalid action)
advances it by one tick. With auto_skip, ticks where WAIT is the only legal action are
fast-forwarded and their rewards summed into the transition. Reaching t == 180 terminates.
"""
from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.actions import DispatchView, compute_mask
from environment.config import EnvConfig, RewardConfig, load_env_defaults, load_reward_config
from environment.constants import (ASSIGN_CAP, MAX_ORDERS, N_ACTIONS, N_TICKS, OBS_DIM, WAIT_ACTION,
                                   decode_action)
from environment.observation import build_observation
from environment.reward import RewardBreakdown, terminal_reward, tick_reward
from environment.scenario import generate_scenario
from evaluation.metrics import compute_metrics
from simulation.simulator import Engine


class DeliveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None, reward_config: Optional[RewardConfig] = None,
                 sampler: Optional[Callable[[int], EnvConfig]] = None, auto_skip: bool = True,
                 record: bool = False):
        super().__init__()
        self.config = config or load_env_defaults()
        self.rc = reward_config or load_reward_config()
        self.sampler = sampler
        self.auto_skip = auto_skip
        self.record = record
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.engine: Optional[Engine] = None
        self.frames: list = []

    # ------------------------------------------------------------------ API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 1_000_000))
        cfg = (options or {}).get("config")
        if cfg is None:
            cfg = self.sampler(seed) if self.sampler else self.config
        self.current_config = cfg
        self.scenario = generate_scenario(cfg, seed)
        self.engine = Engine(self.scenario, record=self.record)
        self.assign_count = 0
        self.ep_return = 0.0
        self.ep_return_disc = 0.0
        self.ep_breakdown = RewardBreakdown()
        self.n_invalid = 0
        self.n_decisions = 0
        self.frames = [self.engine.snapshot()] if self.record else []
        self._carry = RewardBreakdown()      # reward of ticks skipped inside reset()
        if self.auto_skip:
            self._skip_idle_ticks(self._carry)
        self._refresh()
        return self._obs, self._info(RewardBreakdown(), False)

    def step(self, action: int):
        if self.engine is None or self.engine.finished:
            raise RuntimeError("call reset() before step()")
        action = int(action)
        bd, self._carry = self._carry, RewardBreakdown()
        invalid = False
        self.n_decisions += 1

        if action == WAIT_ACTION:
            self._advance(bd)
        elif 0 <= action < WAIT_ACTION and self._mask[action]:
            di, oj = decode_action(action)
            self.engine.assign(self.engine.drivers[di], self._slots[oj])
            self.assign_count += 1
            if self.assign_count >= ASSIGN_CAP:
                self._advance(bd)
        else:
            invalid = True
            self.n_invalid += 1
            bd.invalid -= self.rc.invalid_penalty
            self._advance(bd)

        if self.auto_skip and not self.engine.finished:
            self._refresh()
            self._skip_idle_ticks(bd)
        self._refresh()

        terminated = bool(self.engine.finished)
        reward = bd.total
        self.ep_return += reward
        self.ep_breakdown.add(bd)
        info = self._info(bd, invalid)
        if terminated:
            info["episode_metrics"] = compute_metrics(self.engine)
            info["episode_return"] = self.ep_return
            info["episode_return_discounted"] = self.ep_return_disc
            info["episode_breakdown"] = self.ep_breakdown.to_dict()
            info["n_invalid"] = self.n_invalid
            info["n_decisions"] = self.n_decisions
        return self._obs, float(reward), terminated, False, info

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    def get_view(self) -> DispatchView:
        return DispatchView(self.engine, self._slots, self._mask)

    # ------------------------------------------------------------------ internals
    def _advance(self, bd: RewardBreakdown):
        ev = self.engine.advance_tick()
        r = tick_reward(ev, self.rc)
        if self.engine.finished:
            r.terminal = terminal_reward(len(self.engine.open_orders), self.rc)
        bd.add(r)
        self.ep_return_disc += (self.rc.gamma ** (self.engine.t - 1)) * r.total
        self.assign_count = 0
        if self.record:
            self.frames.append(self.engine.snapshot())

    def _skip_idle_ticks(self, bd: RewardBreakdown):
        """Fast-forward while WAIT is the only legal action. The final tick is always left to an
        explicit agent step so reset() can never return an already-finished episode."""
        while self.engine.t < N_TICKS - 1:
            self._refresh()
            if self._mask[:WAIT_ACTION].any():
                return
            self._advance(bd)

    def _refresh(self):
        eng = self.engine
        self._slots = eng.pending[:MAX_ORDERS]
        self._mask = compute_mask(eng, self._slots)
        self._obs = build_observation(eng, self._slots, self._mask, self.assign_count)

    def _info(self, bd: RewardBreakdown, invalid: bool) -> dict:
        return {
            "action_mask": self._mask.copy(),
            "t": self.engine.t,
            "reward_breakdown": bd.to_dict(),
            "invalid_action": invalid,
            "scenario_fingerprint": self.scenario.fingerprint,
            "scenario_name": self.current_config.name,
        }
