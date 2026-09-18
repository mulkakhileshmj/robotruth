from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DriftFinding:
    metric: str
    reference: Any
    current: Any
    delta: float
    tolerance: float
    severity: str  # "ok" | "warn" | "fail"
    message: str


@dataclass
class DriftReport:
    kind: str
    findings: list[DriftFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "fail" for f in self.findings)

    @property
    def worst(self) -> str:
        if any(f.severity == "fail" for f in self.findings):
            return "fail"
        if any(f.severity == "warn" for f in self.findings):
            return "warn"
        return "ok"

    def summary(self) -> str:
        n_fail = sum(f.severity == "fail" for f in self.findings)
        n_warn = sum(f.severity == "warn" for f in self.findings)
        return f"{self.kind} drift: {self.worst.upper()} ({n_fail} fail, {n_warn} warn, {len(self.findings)} metrics)"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "passed": self.passed, "worst": self.worst,
                "findings": [f.__dict__ for f in self.findings]}


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_round).encode()).hexdigest()


def _round(v):
    if isinstance(v, float):
        return round(v, 4)
    if hasattr(v, "tolist"):
        return v.tolist()
    return str(v)


def grade(delta: float, warn: float, fail: float) -> str:
    if abs(delta) >= fail:
        return "fail"
    if abs(delta) >= warn:
        return "warn"
    return "ok"
