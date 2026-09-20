"""Policy interface: the one small, boring contract between the platform and any policy.

The runner does not care whether the policy underneath is a VLA, RL, imitation, classical
control, or generated code. It cares that the declared observation/action contract matches
what the backend produces, and it hashes the whole declaration into the run manifest so a
policy whose executable configuration changed cannot silently reuse an old evaluation
("Same Weights, Different Robot", blocked at the door).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from policyci.scenario import content_hash

POLICY_IFACE_VERSION = "0.1"


@dataclass(frozen=True)
class PolicyContract:
    """Machine-readable declaration of the executable policy."""

    name: str                      # e.g. "act_aloha_v18"
    model_id: str                  # hub repo or path
    action_dim: int
    action_semantics: str          # "joint_position" | "ee_delta" | ...
    control_hz: float
    obs_keys: tuple[str, ...]      # observation streams the policy consumes
    weights_sha256: str | None = None
    variant: dict = field(default_factory=dict)  # controlled perturbations, empty for the real policy
    iface_version: str = POLICY_IFACE_VERSION

    @property
    def hash(self) -> str:
        return content_hash({
            "name": self.name, "model_id": self.model_id, "action_dim": self.action_dim,
            "action_semantics": self.action_semantics, "control_hz": self.control_hz,
            "obs_keys": list(self.obs_keys), "weights_sha256": self.weights_sha256,
            "variant": self.variant, "iface_version": self.iface_version,
        })

    def to_dict(self) -> dict:
        return {"name": self.name, "model_id": self.model_id, "action_dim": self.action_dim,
                "action_semantics": self.action_semantics, "control_hz": self.control_hz,
                "obs_keys": list(self.obs_keys), "weights_sha256": self.weights_sha256,
                "variant": self.variant, "iface_version": self.iface_version,
                "hash": self.hash}


class Policy(Protocol):
    contract: PolicyContract

    def reset(self, seed: int | None = None) -> None:
        """Start of episode. `seed` controls any policy-side stochasticity so the
        A-vs-A noise floor is measurable and honest."""
        ...

    def act(self, obs: Any) -> Any: ...
