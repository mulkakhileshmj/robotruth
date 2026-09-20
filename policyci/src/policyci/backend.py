"""Simulator backend interface and environment pins.

The platform must not care which simulator is underneath. A backend turns a Scenario into
a live episode and reports its pins: the exact software stack whose determinism the
scenario identity depends on. Physics is not bit-reproducible across engine versions or
drivers, so two runs are comparable only when their pins match; regression.py enforces
that fail-closed.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Any, Protocol

from policyci.scenario import Scenario, content_hash


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "absent"


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "none"


def environment_pins(extra: dict | None = None) -> dict:
    """Everything determinism depends on. Recorded in every run manifest and Passport."""
    pins = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _pkg_version("numpy"),
        "gymnasium": _pkg_version("gymnasium"),
        "gym_aloha": _pkg_version("gym-aloha"),
        "mujoco": _pkg_version("mujoco"),
        "dm_control": _pkg_version("dm-control"),
        "torch": _pkg_version("torch"),
        "lerobot": _pkg_version("lerobot"),
        "gpu": _gpu_name(),
    }
    if extra:
        pins.update(extra)
    return pins


def pins_hash(pins: dict) -> str:
    return content_hash(pins)


@dataclass
class StepResult:
    obs: Any
    reward: float
    done: bool
    info: dict


class SimBackend(Protocol):
    """Contract every simulator backend implements."""

    backend_id: str
    task: str
    max_steps: int
    control_hz: float

    def pins(self) -> dict: ...

    def reset(self, scenario: Scenario) -> Any:
        """Deterministically reconstruct the scenario's initial state. Same scenario in,
        identical scene out, every time, or the backend must raise."""
        ...

    def step(self, action: Any) -> StepResult: ...

    def ground_truth(self) -> dict:
        """Simulator-truth episode summary consumed by the evaluator. Must include:
        success (bool), max_reward (float), steps (int). Backends add task-specific
        stage signals (e.g. grasped, lifted) when the engine exposes them."""
        ...

    def render_frame(self) -> Any | None:
        """RGB frame for replay video, or None if rendering is off."""
        ...

    def close(self) -> None: ...
