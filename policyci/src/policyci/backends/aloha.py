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

    def __init__(self, render: bool = False, max_steps: int = 400, video_stride: int = 2):
        import gymnasium as gym
        import gym_aloha  # noqa: F401  (registers envs)
        self.max_steps = max_steps
        self._env = gym.make(ENV_ID, obs_type="pixels_agent_pos", max_episode_steps=max_steps)
        self._render = render
        self._video_stride = max(1, video_stride)
        self._max_reward = 0.0
        self._steps = 0
        self._t_grasp: int | None = None
        self._t_lift: int | None = None
        self._last_frame = None

    @staticmethod
    def _pose_from(params: dict):
        """The 7-vector [x, y, z, qw, qx, qy, qz], or None for a seed-only scenario."""
        if "cube_x" not in params and "cube_y" not in params and "cube_yaw_deg" not in params:
            return None
        import numpy as np
        from policyci.factors import yaw_to_quat
        x = float(params.get("cube_x", 0.1))
        y = float(params.get("cube_y", 0.5))
        z = float(params.get("cube_z", 0.05))
        quat = yaw_to_quat(float(params.get("cube_yaw_deg", 0.0)))
        return np.concatenate([[x, y, z], quat])

    def pins(self) -> dict:
        return environment_pins({"env_id": ENV_ID, "obs_type": "pixels_agent_pos",
                                 "max_steps": self.max_steps})

    def reset(self, scenario: Scenario) -> Any:
        """Reconstruct the scenario's scene.

        A seed-only scenario just reseeds gym-aloha's own sampler. A factor scenario names
        the cube pose explicitly, which is the whole point: a failure can then be described
        as a region rather than a list of hashes. gym-aloha sets the module-level box pose
        from `sample_box_pose` immediately before the dm-control reset, so overriding that
        function for the duration of the call is the least invasive way to place the cube.
        Note that gym-aloha never rotates the cube itself, so a yaw factor is genuinely
        out of distribution rather than an injected fault.
        """
        seed = int(scenario.params["reset_seed"])
        pose = self._pose_from(scenario.params)
        if pose is None:
            obs, _info = self._env.reset(seed=seed)
        else:
            import gym_aloha.env as ge
            original = ge.sample_box_pose
            ge.sample_box_pose = lambda *a, **k: pose
            try:
                obs, _info = self._env.reset(seed=seed)
            finally:
                ge.sample_box_pose = original
        self._max_reward = 0.0
        self._steps = 0
        self._t_grasp = None
        self._t_lift = None
        return obs

    def step(self, action: Any) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action))
        self._steps += 1
        if self._render and self._steps % self._video_stride == 0:
            # the observation already carries the top camera image: no second render pass
            self._last_frame = obs["pixels"]["top"]
        else:
            self._last_frame = None
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
        """The frame captured by the last step, or None. Free: it reuses the observation."""
        return self._last_frame

    def close(self) -> None:
        self._env.close()
