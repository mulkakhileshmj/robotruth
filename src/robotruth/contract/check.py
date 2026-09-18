"""Compare two ExecSpec manifests and fail closed.

Severity policy (from the evidence, see ROOT_PROBLEM_MEMO section 4):
- FATAL: a mismatch that is known to turn one policy into another. Weights, normalizer
  statistics, action space or mode, chunking, control rate, gripper convention, embodiment,
  observation preprocessing, camera set.
- WARN: a mismatch that changes behaviour in measurable but usually survivable ways.
  dtype, device, export format, camera resolution, inference rate.
- INFO: bookkeeping differences.

An unknown value on a FATAL field is itself FATAL unless `allow_unknown=True`. That is the
fail-closed rule: if you cannot prove the deployed policy matches the evaluated one, the
evaluation does not certify the deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from robotruth.contract.spec import ExecSpec


class Severity(str, Enum):
    FATAL = "fatal"
    WARN = "warn"
    INFO = "info"
    OK = "ok"


@dataclass
class Finding:
    field: str
    severity: Severity
    evaluated: Any
    deployed: Any
    message: str


@dataclass
class CheckResult:
    findings: list[Finding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == Severity.FATAL for f in self.findings)

    def by_severity(self, sev: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == sev]

    def summary(self) -> str:
        n_fatal = len(self.by_severity(Severity.FATAL))
        n_warn = len(self.by_severity(Severity.WARN))
        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict}: {n_fatal} fatal, {n_warn} warn, {len(self.findings)} checks"


def _get(spec: ExecSpec, dotted: str) -> Any:
    obj: Any = spec
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_unknown(v: Any) -> bool:
    return v is None or v == "unknown" or v == [] or v == ""


# (field, severity, human explanation of why it matters)
FATAL_FIELDS: list[tuple[str, str]] = [
    ("policy.weights_sha256", "Different weights are a different policy."),
    ("action.normalization.stats_sha256", "Normalizer statistics are part of the executable policy (arXiv 2606.03724)."),
    ("action.normalization.type", "Normalization type changes the action decode."),
    ("action.space", "Action space mismatch means commands are interpreted in a different frame."),
    ("action.mode", "Absolute versus delta actions are not interchangeable."),
    ("action.dim", "Action dimensionality mismatch."),
    ("action.chunk_size", "Chunk length changes closed-loop behaviour under latency."),
    ("action.n_action_steps", "Executed steps per chunk changes closed-loop behaviour."),
    ("action.control_hz", "Control rate mismatch changes the physical meaning of each action."),
    ("action.gripper_convention", "Gripper open/closed convention inverted or rescaled."),
    ("observation.normalization.stats_sha256", "Observation normalizer statistics differ."),
    ("observation.preprocessing_sha256", "Preprocessing (resize, crop, colour) differs; ROEP shows this masks true capability."),
    ("observation.state_dim", "Proprioceptive state dimensionality differs."),
    ("embodiment.robot", "Different robot model."),
]

WARN_FIELDS: list[tuple[str, str]] = [
    ("runtime.dtype", "Precision changes can change closed-loop behaviour (arXiv 2609.14146)."),
    ("runtime.export_format", "Export path (ONNX, TensorRT) can change closed-loop behaviour even when benchmark score is unchanged."),
    ("runtime.device", "Device changes inference latency, which changes closed-loop behaviour."),
    ("runtime.inference_hz", "Inference rate differs."),
    ("embodiment.unit_id", "Different physical unit: unit-to-unit variance dropped a policy from 98% to 18% (SPACE, arXiv 2606.24049)."),
    ("embodiment.controller", "Low-level controller differs."),
    ("embodiment.controller_hz", "Low-level controller rate differs."),
    ("observation.image_resize", "Image resize policy differs."),
    ("action.units", "Action units differ."),
]

INFO_FIELDS: list[tuple[str, str]] = [
    ("policy.name", "Policy name differs."),
    ("policy.family", "Policy family differs."),
    ("runtime.framework_version", "Framework version differs."),
]


def _camera_findings(evaluated: ExecSpec, deployed: ExecSpec) -> list[Finding]:
    out: list[Finding] = []
    ev = {c.name: c for c in evaluated.observation.cameras}
    de = {c.name: c for c in deployed.observation.cameras}
    if set(ev) != set(de):
        out.append(Finding("observation.cameras", Severity.FATAL, sorted(ev), sorted(de),
                           "Camera set differs. Policies infer camera pose from background cues and collapse when it changes (arXiv 2510.02268)."))
        return out
    for name in ev:
        a, b = ev[name], de[name]
        if (a.width, a.height) != (b.width, b.height):
            out.append(Finding(f"observation.cameras[{name}].resolution", Severity.WARN,
                               (a.width, a.height), (b.width, b.height), "Camera resolution differs."))
        for attr, sev, msg in (
            ("intrinsics_sha256", Severity.FATAL, "Camera intrinsics differ."),
            ("extrinsics_sha256", Severity.FATAL, "Camera extrinsics differ; a moved camera shifted completion by 22 points (PhAIL)."),
        ):
            va, vb = getattr(a, attr), getattr(b, attr)
            if va is None or vb is None:
                out.append(Finding(f"observation.cameras[{name}].{attr}", Severity.WARN, va, vb,
                                   f"{msg} Not recorded on one side; cannot verify."))
            elif va != vb:
                out.append(Finding(f"observation.cameras[{name}].{attr}", sev, va, vb, msg))
    if not out:
        out.append(Finding("observation.cameras", Severity.OK, sorted(ev), sorted(de), "Camera set and calibration hashes match."))
    return out


def check_contract(evaluated: ExecSpec, deployed: ExecSpec, allow_unknown: bool = False,
                   ignore: Optional[set[str]] = None) -> CheckResult:
    """Compare the evaluated manifest against the deployed one."""
    ignore = ignore or set()
    res = CheckResult()

    def compare(fields: list[tuple[str, str]], sev: Severity):
        for f, why in fields:
            if f in ignore:
                continue
            a, b = _get(evaluated, f), _get(deployed, f)
            if _is_unknown(a) or _is_unknown(b):
                if sev == Severity.FATAL and not allow_unknown:
                    res.findings.append(Finding(f, Severity.FATAL, a, b,
                                                f"Unknown on at least one side; fail closed. {why}"))
                else:
                    res.findings.append(Finding(f, Severity.WARN if sev == Severity.FATAL else Severity.INFO, a, b,
                                                f"Unknown on at least one side; cannot verify. {why}"))
                continue
            if a != b:
                res.findings.append(Finding(f, sev, a, b, why))
            else:
                res.findings.append(Finding(f, Severity.OK, a, b, "match"))

    compare(FATAL_FIELDS, Severity.FATAL)
    compare(WARN_FIELDS, Severity.WARN)
    compare(INFO_FIELDS, Severity.INFO)
    res.findings.extend(_camera_findings(evaluated, deployed))
    return res
