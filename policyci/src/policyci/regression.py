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
class NoiseFloor:
    """What run-to-run variation actually looks like for this cell.

    Measured from two runs of the SAME policy over the same battery. Three things matter
    and the old flips-over-two estimate reported none of them:

    - The quantity compared against `newly_broken` is one-directional (passed in the first
      reference run, failed in the second). Halving a two-directional count assumed the
      flips split evenly, which nothing guarantees.
    - It carries an interval, on a tool whose entire claim is that numbers carry evidence.
    - It distinguishes a cell that is deterministic from one that was never measured.
      Those print the same today and mean opposite things: in a deterministic cell every
      scenario difference is real, which is a STRONGER position, not a missing measurement.
    """

    n: int
    n_eligible: int              # scenarios that passed in the first reference run
    expected_broken: int         # one-directional flips: pass -> fail
    expected_fixed: int          # fail -> pass
    total_flips: int
    rate: object = None          # robotruth Interval on expected_broken / n_eligible
    determinism: str = "stochastic"
    identical_reward: int = 0
    identical_steps: int = 0

    @property
    def is_deterministic(self) -> bool:
        return self.determinism == "deterministic"

    @property
    def statement(self) -> str:
        if self.is_deterministic:
            return (f"This cell is deterministic: two runs of the same policy agreed on all "
                    f"{self.n} scenarios, with identical reward in {self.identical_reward}/{self.n} "
                    f"and identical step counts in {self.identical_steps}/{self.n}. The floor is "
                    f"exactly zero, so every scenario difference below is real.")
        return (f"Measured floor: {self.expected_broken} of {self.n_eligible} passing scenarios "
                f"flipped to failure when the same policy was run again "
                f"({100 * self.rate.estimate:.1f}% [{100 * self.rate.lower:.1f}, "
                f"{100 * self.rate.upper:.1f}]). {self.total_flips} scenarios flipped in either "
                f"direction.")

    def to_dict(self) -> dict:
        d = {"n": self.n, "n_eligible": self.n_eligible,
             "expected_broken": self.expected_broken, "expected_fixed": self.expected_fixed,
             "total_flips": self.total_flips, "determinism": self.determinism,
             "identical_reward": self.identical_reward, "identical_steps": self.identical_steps,
             "statement": self.statement}
        if self.rate is not None:
            d["rate"] = {"estimate": self.rate.estimate, "lower": self.rate.lower,
                         "upper": self.rate.upper, "method": self.rate.method}
        return d


def classify_determinism(ra: dict, rb: dict) -> tuple[str, int, int]:
    """Is this cell bit-deterministic? Compares outcome, reward and step count.

    Equal success rates are not enough: two runs can reach the same rate by different
    routes. Identical reward AND identical step count in every scenario is the real test,
    and it is what first light on ALOHA transfer cube actually showed.
    """
    hashes = list(ra["results"])
    same_reward = sum(1 for h in hashes
                      if ra["results"][h].get("max_reward") == rb["results"].get(h, {}).get("max_reward"))
    same_steps = sum(1 for h in hashes
                     if ra["results"][h].get("steps") == rb["results"].get(h, {}).get("steps"))
    same_outcome = sum(1 for h in hashes
                       if ra["results"][h]["success"] == rb["results"].get(h, {}).get("success"))
    n = len(hashes)
    deterministic = (same_reward == n and same_steps == n and same_outcome == n)
    return ("deterministic" if deterministic else "stochastic"), same_reward, same_steps


def measure_noise_floor(ra: dict, rb: dict, alpha: float = 0.05) -> NoiseFloor:
    """The floor, from two runs of the same policy on the same battery."""
    hashes, xa, xb = _outcome_vectors(ra, rb)
    n = len(hashes)
    broke = int(np.sum((xa == 1) & (xb == 0)))
    fixed = int(np.sum((xa == 0) & (xb == 1)))
    eligible = int(np.sum(xa == 1))
    kind, same_reward, same_steps = classify_determinism(ra, rb)
    rate = wilson(broke, eligible, alpha) if eligible > 0 else None
    return NoiseFloor(n=n, n_eligible=eligible, expected_broken=broke, expected_fixed=fixed,
                      total_flips=broke + fixed, rate=rate, determinism=kind,
                      identical_reward=same_reward, identical_steps=same_steps)


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
    noise_flips: int | None = None       # one-directional flips in the A-vs-A reference
    warnings: list[str] = field(default_factory=list)
    floor: "NoiseFloor | None" = None
    determinism: str | None = None       # "deterministic" | "stochastic" | None if unmeasured
    hotspots: list = field(default_factory=list)

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
              alpha: float = 0.05, allow_pin_mismatch: bool = False,
              scenario_params: dict[str, dict] | None = None) -> Diff:
    """Compare two runs over the same battery.

    `scenario_params` maps scenario hash to named scene factors. When supplied, the diff
    also reports the regions of that space where breakage concentrates, which is the
    difference between handing over 74 videos and naming the condition they share.
    """
    warnings = check_comparable(a, b, allow_pin_mismatch=allow_pin_mismatch)
    hashes, xa, xb = _outcome_vectors(a, b)
    n = len(hashes)
    fixed = [h for h, va, vb in zip(hashes, xa, xb) if va == 0 and vb == 1]
    broken = [h for h, va, vb in zip(hashes, xa, xb) if va == 1 and vb == 0]

    noise_flips = None
    floor = None
    determinism = None
    if noise_ref is not None:
        ra, rb = noise_ref
        check_comparable(ra, rb, allow_pin_mismatch=allow_pin_mismatch)
        if ra["battery_hash"] != a["battery_hash"]:
            raise ComparabilityError("noise reference ran a different battery")
        floor = measure_noise_floor(ra, rb, alpha=alpha)
        noise_flips = floor.expected_broken
        determinism = floor.determinism

    hotspots: list = []
    if scenario_params and broken:
        from policyci.cluster import find_hotspots
        hotspots = find_hotspots(scenario_params, set(broken), hashes, alpha=alpha)

    return Diff(
        a_name=a["policy"]["name"], b_name=b["policy"]["name"], n=n,
        a_rate=wilson(int(xa.sum()), n, alpha),
        b_rate=wilson(int(xb.sum()), n, alpha),
        paired=paired_diff_ci(xa, xb, alpha),
        sequential=sequential_paired_test(xa.tolist(), xb.tolist(), alpha=alpha, stop_early=False),
        fixed=fixed, newly_broken=broken, noise_flips=noise_flips, warnings=warnings,
        floor=floor, determinism=determinism, hotspots=hotspots,
    )
