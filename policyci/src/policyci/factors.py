"""Named scene factors, so a failure can be described instead of only counted.

A seed-only battery reproduces perfectly but says nothing: "74 scenarios broke" with no
account of what they share. A factor battery gives every scenario named coordinates, which
is what lets the regression engine report a region rather than a list.

Factors are sampled with a Latin hypercube rather than plain uniform draws. Over a few
hundred points it covers the space far more evenly, so a hotspot in one corner is actually
sampled instead of being missed by luck. Sobol would do the same but warns and loses its
balance guarantee unless n is a power of two, which is a silly constraint to put on a
battery size. The sampler is seeded, so the battery stays a pure function of its inputs and
remains content-addressed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from policyci.scenario import Battery, Scenario

FACTORS_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class Factor:
    """One named, continuous scene coordinate."""

    name: str
    low: float
    high: float
    unit: str
    nominal: float | None = None
    note: str = ""

    def denormalise(self, u: float) -> float:
        return float(self.low + u * (self.high - self.low))


@dataclass(frozen=True)
class FactorSpace:
    task: str
    factors: tuple[Factor, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.factors)

    def to_dict(self) -> dict:
        return {"task": self.task, "schema_version": FACTORS_SCHEMA_VERSION,
                "factors": [{"name": f.name, "low": f.low, "high": f.high, "unit": f.unit,
                             "nominal": f.nominal, "note": f.note} for f in self.factors]}


# The ALOHA transfer-cube cell. gym-aloha's own sampler draws the cube position uniformly
# from x in [0.0, 0.2] and y in [0.4, 0.6] at fixed height, and always with an identity
# quaternion: the cube is never rotated, in training or in evaluation. Yaw is therefore a
# genuine out-of-distribution axis rather than an injected fault, which makes it the right
# thing to look for a hotspot along.
ALOHA_TRANSFER_CUBE = FactorSpace(
    task="aloha_transfer_cube",
    factors=(
        Factor("cube_x", 0.0, 0.2, "m", nominal=0.1,
               note="gym-aloha's own training range"),
        Factor("cube_y", 0.4, 0.6, "m", nominal=0.5,
               note="gym-aloha's own training range"),
        Factor("cube_yaw_deg", -30.0, 30.0, "deg", nominal=0.0,
               note="OUT OF DISTRIBUTION: gym-aloha always spawns the cube unrotated"),
    ),
)

SPACES: dict[str, FactorSpace] = {ALOHA_TRANSFER_CUBE.task: ALOHA_TRANSFER_CUBE}


def sample_factor_battery(name: str, space: FactorSpace, backend_id: str, n: int,
                          base_seed: int = 0) -> Battery:
    """A battery whose scenarios carry named coordinates.

    Each scenario keeps a `reset_seed` too, so anything the backend does not control
    explicitly stays reproducible.
    """
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=len(space.factors), seed=base_seed)
    pts = sampler.random(n)
    rng = np.random.Generator(np.random.PCG64(base_seed))
    seeds = rng.integers(0, 2**31 - 1, size=n)

    scenarios = []
    for i in range(n):
        params: dict = {"reset_seed": int(seeds[i])}
        for j, f in enumerate(space.factors):
            params[f.name] = round(f.denormalise(float(pts[i, j])), 6)
        params["_factors_version"] = FACTORS_SCHEMA_VERSION
        scenarios.append(Scenario(index=i, params=params))
    return Battery(name=name, task=space.task, backend_id=backend_id,
                   scenarios=tuple(scenarios))


def yaw_to_quat(yaw_deg: float) -> np.ndarray:
    """Rotation about the vertical axis, as MuJoCo's [w, x, y, z]."""
    half = math.radians(yaw_deg) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def scenario_factor_names(params: dict) -> list[str]:
    """The named factors in a scenario, excluding bookkeeping keys."""
    return sorted(k for k, v in params.items()
                  if not k.startswith("_") and k != "reset_seed"
                  and isinstance(v, (int, float)) and not isinstance(v, bool))
