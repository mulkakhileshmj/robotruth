"""The ExecSpec manifest: everything that makes a checkpoint an executable policy.

Design rules:
- Every field that changes robot behaviour is here. Unknown is a value, not an omission,
  so the checker can fail closed on it.
- Hashes, not contents. A manifest is small enough to commit next to a checkpoint.
- No framework imports. Adapters live in `extract.py`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SPEC_VERSION = "0.1"

ActionMode = Literal["absolute", "delta", "unknown"]
ActionSpace = Literal["joint_position", "joint_velocity", "joint_torque", "eef_pose", "eef_delta", "mixed", "unknown"]
NormType = Literal["mean_std", "min_max", "quantile", "identity", "unknown"]


class _Model(BaseModel):
    model_config = ConfigDict(validate_assignment=True)


class PolicyIdentity(_Model):
    name: str = "unknown"
    family: Literal["lerobot", "openpi", "gr00t", "custom", "unknown"] = "unknown"
    weights_sha256: Optional[str] = None
    weights_files: list[str] = Field(default_factory=list)
    weights_bytes: Optional[int] = None
    config_sha256: Optional[str] = None
    source_path: Optional[str] = None


class Normalization(_Model):
    type: NormType = "unknown"
    stats_sha256: Optional[str] = None
    stats_source: Optional[str] = None
    action_dim: Optional[int] = None


class ActionSemantics(_Model):
    space: ActionSpace = "unknown"
    mode: ActionMode = "unknown"
    dim: Optional[int] = None
    units: Optional[str] = None
    chunk_size: Optional[int] = None
    n_action_steps: Optional[int] = None
    control_hz: Optional[float] = None
    gripper_convention: Optional[str] = None
    normalization: Normalization = Field(default_factory=Normalization)


class CameraSpec(_Model):
    name: str
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    intrinsics_sha256: Optional[str] = None
    extrinsics_sha256: Optional[str] = None


class ObservationSpec(_Model):
    cameras: list[CameraSpec] = Field(default_factory=list)
    state_dim: Optional[int] = None
    state_keys: list[str] = Field(default_factory=list)
    preprocessing_sha256: Optional[str] = None
    image_resize: Optional[str] = None
    normalization: Normalization = Field(default_factory=Normalization)


class Embodiment(_Model):
    robot: str = "unknown"
    unit_id: Optional[str] = None
    controller: Optional[str] = None
    controller_hz: Optional[float] = None


class RuntimeSpec(_Model):
    dtype: Optional[str] = None
    device: Optional[str] = None
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    export_format: Optional[str] = None
    inference_hz: Optional[float] = None


class ExecSpec(_Model):
    spec_version: str = SPEC_VERSION
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    role: Literal["evaluated", "deployed", "unspecified"] = "unspecified"
    policy: PolicyIdentity = Field(default_factory=PolicyIdentity)
    action: ActionSemantics = Field(default_factory=ActionSemantics)
    observation: ObservationSpec = Field(default_factory=ObservationSpec)
    embodiment: Embodiment = Field(default_factory=Embodiment)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    notes: list[str] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list, description="Fields the extractor could not determine.")

    def to_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: Path | str) -> "ExecSpec":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def fingerprint(self) -> str:
        """Stable hash of the behaviour-relevant fields (excludes timestamps, notes, paths)."""
        clean = ExecSpec.model_validate(self.model_dump())  # normalise types (int vs float) before hashing
        payload = {
            "policy": {k: v for k, v in clean.policy.model_dump(mode="json").items() if k != "source_path"},
            "action": clean.action.model_dump(mode="json"),
            "observation": clean.observation.model_dump(mode="json"),
            "embodiment": clean.embodiment.model_dump(mode="json"),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    """Order-independent combined hash of several files (name plus content)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
