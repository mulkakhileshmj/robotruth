"""gym-aloha (MuJoCo) backend: ALOHA transfer-cube, the frozen first cell.

Scenario determinism: gym-aloha samples the cube pose from the reset seed, so
reset(seed=s) reconstructs the identical scene. Reward stages (dm-control task):
1 touched, 2 lifted, 3 handed over / near target, 4 success. Those stages are the
evaluator's ground-truth gates. Runs on the GPU box only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from policyci.backend import StepResult, environment_pins
from policyci.scenario import Scenario

TASK = "aloha_transfer_cube"
ENV_ID = "gym_aloha/AlohaTransferCube-v0"


class AlohaTransferCubeBackend:
    backend_id = "gym_aloha.transfer_cube.v0"
    task = TASK
    max_steps = 400
    control_hz = 50.0

    def __init__(self, render: bool = False, max_steps: int = 400):
        import gymnasium as gym
        import gym_aloha  # noqa: F401  (registers envs)
        self.max_steps = max_steps
        self._env = gym.make(ENV_ID, obs_type="pixels_agent_pos", max_episode_steps=max_steps)
        self._render = render
        self._max_reward = 0.0
        self._steps = 0
        self._t_grasp: int | None = None
        self._t_lift: int | None = None

    def pins(self) -> dict:
        return environment_pins({"env_id": ENV_ID, "obs_type": "pixels_agent_pos",
                                 "max_steps": self.max_steps})

    def reset(self, scenario: Scenario) -> Any:
        seed = int(scenario.params["reset_seed"])
        obs, _info = self._env.reset(seed=seed)
        self._max_reward = 0.0
        self._steps = 0
        self._t_grasp = None
        self._t_lift = None
        return obs

    def step(self, action: Any) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action))
        self._steps += 1
        r = float(reward)
        if r >= 1 and self._t_grasp is None:
            self._t_grasp = self._steps
        if r >= 2 and self._t_lift is None:
            self._t_lift = self._steps
        self._max_reward = max(self._max_reward, r)
        return StepResult(obs=obs, reward=r, done=bool(terminated or truncated), info=dict(info))

    def ground_truth(self) -> dict:
        return {
            "success": self._max_reward >= 4,
            "max_reward": self._max_reward,
            "steps": self._steps,
            "touched": self._max_reward >= 1,
            "lifted": self._max_reward >= 2,
            "t_touch": self._t_grasp,
            "t_lift": self._t_lift,
            "timed_out": self._steps >= self.max_steps and self._max_reward < 4,
        }

    def render_frame(self):
        if not self._render:
            return None
        try:
            return self._env.render()
        except Exception:
            return None

    def close(self) -> None:
        self._env.close()
