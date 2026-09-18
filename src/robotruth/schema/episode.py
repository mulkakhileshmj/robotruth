"""One record per episode. Small enough to write for every rollout, complete enough to audit."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1"


class _Model(BaseModel):
    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)


class FailureClass(str, Enum):
    """Top-level failure classes. Ordered roughly by where in the stack the fault lives."""

    PERCEPTION = "perception"
    GRASP = "grasp"
    PLANNING = "planning"
    ACTION = "action"
    CONTROL = "control"
    HARDWARE = "hardware"
    ENVIRONMENT = "environment"
    LANGUAGE = "language"
    TIMEOUT = "timeout"
    SAFETY_STOP = "safety_stop"
    HUMAN_ERROR = "human_error"
    UNKNOWN = "unknown"


# Subclasses per class. Sources: the assurance blueprint taxonomy, RoboFAC (arXiv 2505.12224,
# most real failures are execution level: grasp pose and trajectory), "How VLAs (Really) Work"
# (arXiv 2604.21192: grasp first, collisions second, semantic confusion third), Dyna's worn
# gripper anecdote, SPACE unit variance, camera drift results.
FAILURE_SUBCLASSES: dict[str, list[str]] = {
    "perception": ["target_not_detected", "wrong_object", "pose_or_depth_error", "occlusion", "lighting_or_reflection", "camera_moved_or_drift"],
    "grasp": ["missed_grasp", "slip", "unstable_grasp", "wrong_grasp_point", "gripper_convention"],
    "planning": ["wrong_decomposition", "invalid_ordering", "missing_prerequisite", "unsafe_plan", "wrong_recovery"],
    "action": ["oscillation", "freeze", "direction_reversal", "inconsistent_chunks", "excessive_magnitude", "wrong_gripper_command"],
    "control": ["tracking_error", "delay_or_jitter", "unexpected_contact", "collision", "jam", "excessive_force", "inadequate_compliance"],
    "hardware": ["dropped_frames", "stale_telemetry", "sensor_drift", "encoder_disagreement", "motor_overheating", "gripper_wear", "network_loss", "battery"],
    "environment": ["object_out_of_reach", "novel_object", "clutter", "human_interference", "fixture_misaligned", "consumable_missing"],
    "language": ["ignored_instruction", "ambiguous_instruction", "negation_failed", "wrong_target_from_language"],
    "timeout": ["no_progress", "slow_but_progressing", "stuck_in_retry"],
    "safety_stop": ["operator_stop", "monitor_stop", "e_stop", "workspace_limit"],
    "human_error": ["bad_reset", "wrong_task_setup", "label_error"],
    "unknown": ["unknown"],
}


class InterventionSource(str, Enum):
    TELEOP = "teleop"           # human took over the arm
    RESET = "reset"             # human reset the scene
    STOP = "stop"               # human or monitor stopped the robot
    VERBAL = "verbal"           # language correction (pi0.7 style coaching)
    AUTOMATIC = "automatic"     # policy or monitor triggered its own recovery
    OTHER = "other"


class JudgeSource(str, Enum):
    HUMAN = "human"
    VLM = "vlm"
    HYBRID = "hybrid"
    AUTOMATIC = "automatic"     # sensor or scripted success check
    UNKNOWN = "unknown"


class Intervention(_Model):
    t_start: float = Field(description="Seconds from episode start.")
    t_end: Optional[float] = None
    source: InterventionSource = InterventionSource.OTHER
    reason: Optional[str] = None
    operator_id: Optional[str] = None
    failure_class: Optional[FailureClass] = None

    @property
    def duration(self) -> Optional[float]:
        return None if self.t_end is None else max(0.0, self.t_end - self.t_start)


class Outcome(_Model):
    success: Optional[bool] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Graded progress or quality in [0, 1].")
    judged_by: JudgeSource = JudgeSource.UNKNOWN
    judge_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    judge_model: Optional[str] = None
    abstained: bool = False
    notes: Optional[str] = None


class FailureInfo(_Model):
    failure_class: FailureClass = FailureClass.UNKNOWN
    subclass: Optional[str] = None
    t_first_evidence: Optional[float] = None
    evidence: list[str] = Field(default_factory=list)
    recovered: Optional[bool] = None
    t_recovered: Optional[float] = None

    @model_validator(mode="after")
    def _check_subclass(self):
        if self.subclass is not None:
            allowed = FAILURE_SUBCLASSES.get(str(self.failure_class), [])
            if self.subclass not in allowed:
                raise ValueError(f"subclass {self.subclass!r} not in taxonomy for {self.failure_class}: {allowed}")
        return self


class Timing(_Model):
    t_start_utc: Optional[str] = None
    duration_s: Optional[float] = None
    time_to_success_s: Optional[float] = None
    timeout_s: Optional[float] = None
    control_hz: Optional[float] = None
    inference_hz_mean: Optional[float] = None
    inference_latency_p95_ms: Optional[float] = None


class Provenance(_Model):
    policy_fingerprint: Optional[str] = Field(default=None, description="ExecSpec.fingerprint() of the deployed policy.")
    policy_name: Optional[str] = None
    policy_weights_sha256: Optional[str] = None
    cell_fingerprint: Optional[str] = Field(default=None, description="Module 4 cell fingerprint hash at session start.")
    unit_fingerprint: Optional[str] = Field(default=None, description="Module 4 robot unit fingerprint hash.")
    raw_log_uri: Optional[str] = None
    raw_log_sha256: Optional[str] = None
    dataset_repo_id: Optional[str] = None
    dataset_episode_index: Optional[int] = None
    logger: str = "robotruth"
    logger_version: str = SCHEMA_VERSION


class EpisodeRecord(_Model):
    schema_version: str = SCHEMA_VERSION
    episode_id: str
    task: str
    instruction: Optional[str] = None
    policy: str
    robot: str = "unknown"
    unit_id: Optional[str] = None
    site_id: Optional[str] = None
    session_id: Optional[str] = None
    condition: Optional[str] = Field(default=None, description="Controlled condition label for paired designs.")
    pair_id: Optional[str] = None
    outcome: Outcome = Field(default_factory=Outcome)
    failure: Optional[FailureInfo] = None
    interventions: list[Intervention] = Field(default_factory=list)
    timing: Timing = Field(default_factory=Timing)
    provenance: Provenance = Field(default_factory=Provenance)
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def autonomous(self) -> bool:
        return not any(i.source in (InterventionSource.TELEOP, InterventionSource.RESET, InterventionSource.STOP) for i in self.interventions)

    @property
    def intervention_seconds(self) -> float:
        return float(sum(i.duration or 0.0 for i in self.interventions))

    def to_result_row(self) -> dict:
        """Row for the module 2 results table."""
        s = self.outcome.success
        return {
            "episode_id": self.episode_id, "policy": self.policy, "task": self.task,
            "success": "" if s is None else int(bool(s)),
            "time_to_success": "" if self.timing.time_to_success_s is None else self.timing.time_to_success_s,
            "timeout": "" if self.timing.timeout_s is None else self.timing.timeout_s,
            "pair_id": self.pair_id or "", "unit_id": self.unit_id or "", "session_id": self.session_id or "",
            "score": "" if self.outcome.score is None else self.outcome.score,
            "autonomous": int(self.autonomous), "n_interventions": len(self.interventions),
            "failure_class": (self.failure.failure_class if self.failure else ""),
            "judged_by": self.outcome.judged_by, "policy_fingerprint": self.provenance.policy_fingerprint or "",
        }
