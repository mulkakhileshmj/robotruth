"""Policy Passport: the machine-readable record behind a deploy decision.

A Passport says what was tested, against what, under which software stack, by which
evaluator, and what came out. It is built only from artefacts that already exist on disk
(run manifests, the battery, the diff), never from anything typed in afterwards, and it
carries the hashes of all of them so two Passports diff cleanly and neither can be edited
without breaking its own digest.

What it deliberately does NOT claim: any statement about real-world performance. Until a
calibration between this simulator and a physical cell exists, a Passport's evidence is
simulation evidence, and the record says so in `scope`. That line is the whole reason the
document is trustworthy.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from policyci.regression import Diff
from policyci.scenario import content_hash

PASSPORT_VERSION = "0.1"


def build_passport(a: dict, b: dict, d: Diff, battery_path: str | Path | None = None,
                   baseline_role: str = "baseline", candidate_role: str = "candidate") -> dict:
    """Assemble the Passport for one candidate, judged against one baseline."""
    battery_hash = a["battery_hash"]
    battery_name = a.get("battery_name")
    if battery_path and Path(battery_path).exists():
        head = json.loads(Path(battery_path).read_text(encoding="utf-8").splitlines()[0])
        if head.get("battery_hash") != battery_hash:
            raise ValueError("battery file does not match the hash the runs recorded")

    failures: dict[str, int] = {}
    for r in b["results"].values():
        if not r["success"]:
            failures[r.get("failure") or "unclassified"] = failures.get(r.get("failure") or "unclassified", 0) + 1

    seq = d.sequential
    passport = {
        "kind": "policyci.passport",
        "passport_version": PASSPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),

        "scope": {
            "evidence": "simulation_only",
            "statement": ("These results characterise behaviour in the named simulator under the "
                          "named battery. They do not estimate real-world success: no sim-to-real "
                          "calibration exists for this cell yet."),
        },

        "candidate": {"role": candidate_role, **b["policy"], "policy_seed": b["policy_seed"]},
        "baseline": {"role": baseline_role, **a["policy"], "policy_seed": a["policy_seed"]},

        "test_definition": {
            "task": a["task"],
            "backend_id": a["backend_id"],
            "battery_name": battery_name,
            "battery_hash": battery_hash,
            "n_scenarios": d.n,
            "evaluator_version": a["evaluator_version"],
            "runner_version": a.get("runner_version"),
        },

        "environment_pins": a["pins"],
        "pins_hash": a["pins_hash"],

        "results": {
            "baseline_success": {"estimate": d.a_rate.estimate, "lower": d.a_rate.lower,
                                 "upper": d.a_rate.upper, "method": d.a_rate.method,
                                 "alpha": d.a_rate.alpha, "n": d.a_rate.n},
            "candidate_success": {"estimate": d.b_rate.estimate, "lower": d.b_rate.lower,
                                  "upper": d.b_rate.upper, "method": d.b_rate.method,
                                  "alpha": d.b_rate.alpha, "n": d.b_rate.n},
            "paired_difference": {"estimate": d.paired.estimate, "lower": d.paired.lower,
                                  "upper": d.paired.upper, "method": d.paired.method},
            "sequential": {"decision": seq.decision, "estimate": seq.estimate,
                           "lower": seq.lower, "upper": seq.upper, "alpha": seq.alpha,
                           "n": seq.n, "anytime_valid": True},
            "newly_broken": len(d.newly_broken),
            "fixed": len(d.fixed),
            "noise_floor_measured": d.noise_flips is not None,
            "noise_floor": d.noise_flips,
            "regressions_beyond_noise": d.significant_regressions,
            "failure_pareto": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        },

        "known_limits": [
            "Collision and placement-error gates are not observable in this environment and are "
            "recorded as unavailable, never as passed.",
            "Physics is not bit-reproducible across simulator versions or drivers; a comparison "
            "is only valid against runs carrying the same pins hash.",
        ],

        "evidence": {
            "baseline_manifest_hash": a["manifest_hash"],
            "candidate_manifest_hash": b["manifest_hash"],
            "baseline_merged_from": a.get("merged_from"),
            "candidate_merged_from": b.get("merged_from"),
            "broken_scenarios": sorted(d.newly_broken),
            "fixed_scenarios": sorted(d.fixed),
            "warnings": list(d.warnings),
        },

        "decision": _decision(d),
    }
    passport["passport_hash"] = content_hash(passport)
    return passport


def _decision(d: Diff) -> dict:
    """The recommendation, and the reason for it in one sentence.

    Deliberately conservative: without a measured noise floor nothing is approved, because
    a flip count with no floor under it is not evidence.
    """
    seq = d.sequential
    if not d.warnings and d.noise_flips is None:
        return {"recommendation": "insufficient_evidence",
                "reason": "No noise floor was measured, so the scenario diff cannot be "
                          "separated from run-to-run variation."}
    if seq.decision == "A_better":
        return {"recommendation": "block",
                "reason": "The candidate is worse than the baseline on matched scenarios, "
                          "by an anytime-valid paired test."}
    if seq.decision == "B_better":
        rec = "approve_with_review" if d.newly_broken else "approve"
        return {"recommendation": rec,
                "reason": ("The candidate is better overall" +
                           (f", but {len(d.newly_broken)} scenarios that passed under the "
                            "baseline now fail and should be reviewed." if d.newly_broken
                            else " and breaks nothing that previously passed."))}
    if seq.decision == "no_difference_within_margin":
        return {"recommendation": "no_change",
                "reason": "The candidate is equivalent to the baseline within the stated margin."}
    return {"recommendation": "insufficient_evidence",
            "reason": f"The comparison is still inside the noise after {seq.n} scenarios; "
                      "more trials are needed to decide."}


def write_passport(passport: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(passport, indent=1), encoding="utf-8")
    return out


def verify_passport(path: str | Path) -> bool:
    """True when the file's contents still hash to the digest it carries."""
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = p.pop("passport_hash", None)
    return claimed is not None and content_hash(p) == claimed
