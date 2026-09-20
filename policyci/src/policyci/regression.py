"""Regression engine: scenario-by-scenario diff between two runs, with a noise floor.

Fail-closed comparability: two runs are diffable only when battery hash, evaluator version
and simulator pins all match. A diff without a noise floor is reported as such; with an
A-vs-A reference pair it splits observed flips into expected-noise and beyond-noise, and
the overall verdict comes from robotruth's paired interval and anytime-valid sequential
test, never from raw counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robotruth.stats.intervals import wilson, paired_diff_ci
from robotruth.stats.sequential import sequential_paired_test


class ComparabilityError(RuntimeError):
    """Raised when two runs cannot honestly be compared."""


def load_manifest(path: str | Path) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("kind") != "policyci.run":
        raise ValueError(f"{path} is not a policyci run manifest")
    return m


def check_comparable(a: dict, b: dict, allow_pin_mismatch: bool = False) -> list[str]:
    """Return warnings; raise ComparabilityError on any fatal mismatch."""
    problems, warnings = [], []
    if a["battery_hash"] != b["battery_hash"]:
        problems.append("battery hash differs: the runs did not see the same scenarios")
    if a["evaluator_version"] != b["evaluator_version"]:
        problems.append("evaluator version differs: success definitions are not the same")
    if a["backend_id"] != b["backend_id"]:
        problems.append("backend differs")
    if a["pins_hash"] != b["pins_hash"]:
        diff_keys = sorted(k for k in set(a["pins"]) | set(b["pins"])
                           if a["pins"].get(k) != b["pins"].get(k))
        msg = f"simulator pins differ ({', '.join(diff_keys)}): physics may not be reproducible across them"
        if allow_pin_mismatch:
            warnings.append("ALLOWED OVERRIDE: " + msg)
        else:
            problems.append(msg)
    if problems:
        raise ComparabilityError("; ".join(problems))
    return warnings


@dataclass
class Diff:
    a_name: str
    b_name: str
    n: int
    a_rate: object            # robotruth Interval
    b_rate: object
    paired: object            # paired diff Interval
    sequential: object        # SequentialResult
    fixed: list[str] = field(default_factory=list)          # scenario hashes b fixed
    newly_broken: list[str] = field(default_factory=list)   # scenario hashes b broke
    noise_flips: int | None = None       # flips in the A-vs-A reference, same battery
    warnings: list[str] = field(default_factory=list)

    @property
    def significant_regressions(self) -> int | None:
        if self.noise_flips is None:
            return None
        return max(0, len(self.newly_broken) - self.noise_flips)


def _outcome_vectors(a: dict, b: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    hashes = sorted(a["results"], key=lambda h: a["results"][h]["index"])
    xa = np.array([1.0 if a["results"][h]["success"] else 0.0 for h in hashes])
    xb = np.array([1.0 if b["results"][h]["success"] else 0.0 for h in hashes])
    return hashes, xa, xb


def count_flips(a: dict, b: dict) -> int:
    """Scenario-level disagreements between two runs (any direction)."""
    _, xa, xb = _outcome_vectors(a, b)
    return int(np.sum(xa != xb))


def diff_runs(a: dict, b: dict, noise_ref: tuple[dict, dict] | None = None,
              alpha: float = 0.05, allow_pin_mismatch: bool = False) -> Diff:
    warnings = check_comparable(a, b, allow_pin_mismatch=allow_pin_mismatch)
    hashes, xa, xb = _outcome_vectors(a, b)
    n = len(hashes)
    fixed = [h for h, va, vb in zip(hashes, xa, xb) if va == 0 and vb == 1]
    broken = [h for h, va, vb in zip(hashes, xa, xb) if va == 1 and vb == 0]

    noise_flips = None
    if noise_ref is not None:
        ra, rb = noise_ref
        check_comparable(ra, rb, allow_pin_mismatch=allow_pin_mismatch)
        if ra["battery_hash"] != a["battery_hash"]:
            raise ComparabilityError("noise reference ran a different battery")
        # one-directional expectation: flips split roughly evenly between directions
        noise_flips = count_flips(ra, rb) // 2

    return Diff(
        a_name=a["policy"]["name"], b_name=b["policy"]["name"], n=n,
        a_rate=wilson(int(xa.sum()), n, alpha),
        b_rate=wilson(int(xb.sum()), n, alpha),
        paired=paired_diff_ci(xa, xb, alpha),
        sequential=sequential_paired_test(xa.tolist(), xb.tolist(), alpha=alpha, stop_early=False),
        fixed=fixed, newly_broken=broken, noise_flips=noise_flips, warnings=warnings,
    )
