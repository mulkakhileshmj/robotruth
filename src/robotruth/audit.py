"""Claims audit: which published or internal comparisons survive an interval?

Input: a CSV with columns `policy, task, successes, trials` (one row per policy and task, or
task blank for a pooled number). Output: every pairwise comparison within a task with the
difference, its interval, and whether it is resolved at the requested confidence. This is
the tool behind the first public evidence artifact: run the numbers papers actually report
through it and count how many claims hold up.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as sps

from robotruth.report import Report, Section
from robotruth.stats.intervals import wilson


@dataclass
class Claim:
    policy: str
    task: str
    successes: int
    trials: int


def read_claims(path: Path | str) -> list[Claim]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(Claim(r["policy"].strip(), (r.get("task") or "ALL").strip() or "ALL", int(float(r["successes"])), int(float(r["trials"]))))
    return out


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Newcombe hybrid score interval for p2 - p1 (two independent proportions)."""
    w1, w2 = wilson(k1, n1, alpha), wilson(k2, n2, alpha)
    d = k2 / n2 - k1 / n1
    lo = d - np.sqrt((w1.upper - w1.estimate) ** 2 + (w2.estimate - w2.lower) ** 2)
    hi = d + np.sqrt((w1.estimate - w1.lower) ** 2 + (w2.upper - w2.estimate) ** 2)
    return float(d), float(lo), float(hi)


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(sps.fisher_exact(table)[1])


def audit_claims(claims: list[Claim], alpha: float = 0.05, title: str = "robotruth claims audit") -> Report:
    rep = Report(title)
    by_task: dict[str, list[Claim]] = {}
    for c in claims:
        by_task.setdefault(c.task, []).append(c)

    rate_rows = []
    for c in claims:
        w = wilson(c.successes, c.trials, alpha)
        rate_rows.append({"task": c.task, "policy": c.policy, "trials": c.trials,
                          "rate": f"{w.estimate:.2f} [{w.lower:.2f}, {w.upper:.2f}]", "interval width": round(w.width, 3)})
    rep.add(Section("Reported rates with intervals", "What each number actually says once the interval is attached.", rate_rows))

    comp_rows = []
    n_resolved = 0
    for task, cs in by_task.items():
        for a, b in combinations(cs, 2):
            d, lo, hi = newcombe_diff_ci(a.successes, a.trials, b.successes, b.trials, alpha)
            p = fisher_p(a.successes, a.trials, b.successes, b.trials)
            resolved = (lo > 0) or (hi < 0)
            n_resolved += resolved
            comp_rows.append({"task": task, "A": a.policy, "B": b.policy, "diff (B-A)": round(d, 3),
                              "interval": f"[{lo:+.2f}, {hi:+.2f}]", "Fisher p": round(p, 4),
                              "resolved": "yes" if resolved else "no"})
    n_comp = len(comp_rows)
    body = (f"{n_comp} pairwise comparisons within tasks; {n_resolved} resolved at {100*(1-alpha):.0f}% confidence, "
            f"{n_comp - n_resolved} are inside the noise.")
    rep.add(Section("Pairwise comparisons", body, comp_rows, verdict="INFO"))
    rep.verdict = "WARN" if n_comp and n_resolved < n_comp else "PASS"
    return rep
