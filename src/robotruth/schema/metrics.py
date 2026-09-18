"""Fleet metrics from episode records.

These are the numbers no humanoid company publishes (Epoch AI audit, Feb 2026; Adamo,
Aug 2026): interventions per hour, mean time between interventions, autonomous fraction,
and a failure-class Pareto. Every rate carries an interval.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from robotruth.schema.episode import EpisodeRecord
from robotruth.stats.intervals import Interval, wilson


@dataclass
class FleetMetrics:
    n_episodes: int
    n_judged: int
    success_rate: Optional[Interval]
    autonomous_fraction: Interval
    robot_hours: float
    interventions: int
    interventions_per_hour: Optional[float]
    interventions_per_hour_ci: Optional[tuple[float, float]]
    mean_time_between_interventions_s: Optional[float]
    intervention_time_fraction: Optional[float]
    failure_pareto: list[tuple[str, int, float]] = field(default_factory=list)
    by_policy: dict[str, "FleetMetrics"] = field(default_factory=dict)
    by_unit: dict[str, "FleetMetrics"] = field(default_factory=dict)

    def to_markdown(self, title: str = "Fleet metrics") -> str:
        out = [f"## {title}", ""]
        out.append(f"- Episodes: {self.n_episodes} ({self.n_judged} with an outcome)")
        if self.success_rate:
            out.append(f"- Success rate: {self.success_rate}")
        out.append(f"- Autonomous fraction (no teleop, reset or stop): {self.autonomous_fraction}")
        out.append(f"- Robot hours logged: {self.robot_hours:.2f}")
        if self.interventions_per_hour is not None:
            lo, hi = self.interventions_per_hour_ci or (float('nan'), float('nan'))
            out.append(f"- Interventions per hour: {self.interventions_per_hour:.2f} [{lo:.2f}, {hi:.2f}] ({self.interventions} interventions)")
        if self.mean_time_between_interventions_s is not None:
            out.append(f"- Mean time between interventions: {self.mean_time_between_interventions_s/60:.1f} min")
        if self.intervention_time_fraction is not None:
            out.append(f"- Share of time under intervention: {100*self.intervention_time_fraction:.1f}%")
        if self.failure_pareto:
            out += ["", "| failure class | count | share |", "|---|---|---|"]
            for cls, n, share in self.failure_pareto:
                out.append(f"| {cls} | {n} | {100*share:.0f}% |")
        for label, groups in (("policy", self.by_policy), ("unit", self.by_unit)):
            if len(groups) > 1:
                out += ["", f"| {label} | n | success | autonomous | interventions/h |", "|---|---|---|---|---|"]
                for k, m in sorted(groups.items()):
                    sr = f"{m.success_rate.estimate:.2f} [{m.success_rate.lower:.2f}, {m.success_rate.upper:.2f}]" if m.success_rate else ""
                    iph = f"{m.interventions_per_hour:.2f}" if m.interventions_per_hour is not None else ""
                    out.append(f"| {k} | {m.n_episodes} | {sr} | {m.autonomous_fraction.estimate:.2f} | {iph} |")
        return "\n".join(out)


def _poisson_rate_ci(k: int, exposure_hours: float, alpha: float = 0.05) -> tuple[float, float]:
    from scipy import stats as sps
    lo = 0.0 if k == 0 else sps.chi2.ppf(alpha / 2, 2 * k) / 2
    hi = sps.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo / exposure_hours, hi / exposure_hours


def fleet_metrics(records: Iterable[EpisodeRecord], alpha: float = 0.05, _nested: bool = True) -> FleetMetrics:
    recs = list(records)
    n = len(recs)
    judged = [r for r in recs if r.outcome.success is not None]
    succ = wilson(sum(bool(r.outcome.success) for r in judged), len(judged), alpha) if judged else None
    auton = wilson(sum(r.autonomous for r in recs), n, alpha) if n else Interval(0, 0, 1, 0, "Wilson", alpha)
    durations = np.array([r.timing.duration_s or 0.0 for r in recs], dtype=float)
    hours = float(durations.sum() / 3600.0)
    n_int = sum(len(r.interventions) for r in recs)
    int_time = float(sum(r.intervention_seconds for r in recs))
    iph = n_int / hours if hours > 0 else None
    iph_ci = _poisson_rate_ci(n_int, hours, alpha) if hours > 0 else None
    mtbi = (durations.sum() / n_int) if n_int > 0 else None
    itf = (int_time / durations.sum()) if durations.sum() > 0 else None
    counter = Counter(str(r.failure.failure_class) for r in recs if r.failure is not None)
    total_f = sum(counter.values())
    pareto = [(c, k, k / total_f) for c, k in counter.most_common()] if total_f else []
    fm = FleetMetrics(n, len(judged), succ, auton, hours, n_int, iph, iph_ci, mtbi, itf, pareto)
    if _nested:
        for key, attr in (("by_policy", "policy"), ("by_unit", "unit_id")):
            groups: dict[str, list[EpisodeRecord]] = {}
            for r in recs:
                groups.setdefault(str(getattr(r, attr) or "unknown"), []).append(r)
            setattr(fm, key, {k: fleet_metrics(v, alpha, _nested=False) for k, v in groups.items()})
    return fm
