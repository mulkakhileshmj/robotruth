"""Honest accounting for a runtime monitor.

A failure detector that is never measured on false alarms is a failure detector that will
be switched off in week two. No paper in the FAIL-Detect line reports false alarms per hour
of nominal operation; this module does, with a Poisson interval, next to detection rate
(Wilson interval) and how long before the end of the episode the alarm came.

Definitions used here:

- A "true failure" is a recorded episode whose truth label is failure.
- "Detected" means the Guard left the ok level at least once during that episode.
- "Detection lead" is episode duration minus the time of the first alert, in seconds. It is
  how much warning the operator got before the episode ended (usually the moment the
  failure became obvious or the episode was stopped).
- A "false alarm" is one onset of an alert (ok to slow, handover or stop) inside an episode
  whose truth label is success. An alert that stays raised for 30 steps is one false alarm,
  not 30. Episodes without a truth label count for neither rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
from scipy import stats as sps

from robotruth.stats.intervals import Interval, wilson

LEVELS = ("ok", "slow", "handover", "stop")


@dataclass
class GuardEpisode:
    """What the Guard remembers about one episode after end_episode()."""

    episode_id: str
    truth: Optional[bool]              # True success, False failure, None unknown
    duration_s: float
    n_steps: int
    first_alert_t: Optional[float]     # seconds from episode start, None if never alerted
    max_level: str
    n_alert_onsets: int
    stopped: bool
    max_score: float
    max_score_over_threshold: float    # max of (score - threshold); positive means it crossed

    @property
    def detected(self) -> bool:
        return self.first_alert_t is not None

    @property
    def lead_time_s(self) -> Optional[float]:
        if self.first_alert_t is None:
            return None
        return max(0.0, self.duration_s - self.first_alert_t)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["detected"] = self.detected
        d["lead_time_s"] = self.lead_time_s
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GuardEpisode":
        keys = ("episode_id", "truth", "duration_s", "n_steps", "first_alert_t", "max_level",
                "n_alert_onsets", "stopped", "max_score", "max_score_over_threshold")
        return cls(**{k: d.get(k) for k in keys})


def poisson_rate_interval(k: int, exposure_hours: float, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) interval for a Poisson rate per hour given k events in exposure_hours."""
    if exposure_hours <= 0:
        raise ValueError("exposure must be positive")
    lo = 0.0 if k == 0 else sps.chi2.ppf(alpha / 2, 2 * k) / 2
    hi = sps.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return float(lo / exposure_hours), float(hi / exposure_hours)


@dataclass
class GuardMetrics:
    alpha: float
    n_episodes: int
    n_unlabelled: int
    # failures
    n_failures: int
    n_detected: int
    detection_rate: Optional[Interval]
    lead_time_mean_s: Optional[float]
    lead_time_median_s: Optional[float]
    n_hard_stops_on_failures: int
    # nominal
    n_nominal: int
    nominal_hours: float
    n_false_alarms: int
    false_alarms_per_hour: Optional[float]
    false_alarms_per_hour_ci: Optional[tuple[float, float]]
    n_nominal_with_alarm: int
    nominal_episode_alarm_rate: Optional[Interval]
    n_hard_stops_on_nominal: int
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_episodes(cls, episodes: Iterable[GuardEpisode], alpha: float = 0.05) -> "GuardMetrics":
        eps = list(episodes)
        fails = [e for e in eps if e.truth is False]
        noms = [e for e in eps if e.truth is True]
        unl = [e for e in eps if e.truth is None]
        det = [e for e in fails if e.detected]
        leads = np.array([e.lead_time_s for e in det], dtype=float)
        det_rate = wilson(len(det), len(fails), alpha) if fails else None
        hours = float(sum(e.duration_s for e in noms)) / 3600.0
        n_fa = int(sum(e.n_alert_onsets for e in noms))
        fa_rate = n_fa / hours if hours > 0 else None
        fa_ci = poisson_rate_interval(n_fa, hours, alpha) if hours > 0 else None
        n_nom_alarm = sum(e.detected for e in noms)
        nom_rate = wilson(n_nom_alarm, len(noms), alpha) if noms else None
        notes = []
        if not fails:
            notes.append("No labelled failures: detection rate not estimable.")
        if not noms:
            notes.append("No labelled successes: false alarm rate not estimable.")
        if unl:
            notes.append(f"{len(unl)} episodes without a truth label were ignored.")
        return cls(alpha, len(eps), len(unl), len(fails), len(det), det_rate,
                   float(leads.mean()) if leads.size else None,
                   float(np.median(leads)) if leads.size else None,
                   sum(e.stopped for e in fails),
                   len(noms), hours, n_fa, fa_rate, fa_ci, n_nom_alarm, nom_rate,
                   sum(e.stopped for e in noms), notes)

    def to_markdown(self, title: str = "Guard metrics") -> str:
        out = [f"## {title}", ""]
        out.append(f"- Episodes: {self.n_episodes} ({self.n_failures} failures, {self.n_nominal} successes, "
                   f"{self.n_unlabelled} unlabelled)")
        out += ["", "### Detection on true failures", ""]
        if self.detection_rate is not None:
            out.append(f"- Detection rate: {self.detection_rate}")
            if self.lead_time_mean_s is not None:
                out.append(f"- Warning before episode end: mean {self.lead_time_mean_s:.1f} s, "
                           f"median {self.lead_time_median_s:.1f} s (over {self.n_detected} detected)")
            out.append(f"- Hard-limit stops on failures: {self.n_hard_stops_on_failures}")
        else:
            out.append("- not estimable (no labelled failures)")
        out += ["", "### False alarms on nominal operation", ""]
        if self.false_alarms_per_hour is not None:
            lo, hi = self.false_alarms_per_hour_ci
            out.append(f"- False alarms per hour: {self.false_alarms_per_hour:.2f} [{lo:.2f}, {hi:.2f}] "
                       f"({self.n_false_alarms} alarms in {self.nominal_hours:.2f} h, "
                       f"{100 * (1 - self.alpha):.0f}% Poisson)")
            out.append(f"- Nominal episodes with any alarm: {self.nominal_episode_alarm_rate} "
                       f"(this is the rate the conformal alpha bounds)")
            out.append(f"- Hard-limit stops on nominal episodes: {self.n_hard_stops_on_nominal}")
        else:
            out.append("- not estimable (no labelled successes)")
        if self.notes:
            out += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        def iv(x: Optional[Interval]):
            return None if x is None else {"estimate": x.estimate, "lower": x.lower, "upper": x.upper, "n": x.n}
        return {"alpha": self.alpha, "n_episodes": self.n_episodes, "n_unlabelled": self.n_unlabelled,
                "n_failures": self.n_failures, "n_detected": self.n_detected,
                "detection_rate": iv(self.detection_rate), "lead_time_mean_s": self.lead_time_mean_s,
                "lead_time_median_s": self.lead_time_median_s,
                "n_hard_stops_on_failures": self.n_hard_stops_on_failures,
                "n_nominal": self.n_nominal, "nominal_hours": self.nominal_hours,
                "n_false_alarms": self.n_false_alarms, "false_alarms_per_hour": self.false_alarms_per_hour,
                "false_alarms_per_hour_ci": self.false_alarms_per_hour_ci,
                "n_nominal_with_alarm": self.n_nominal_with_alarm,
                "nominal_episode_alarm_rate": iv(self.nominal_episode_alarm_rate),
                "n_hard_stops_on_nominal": self.n_hard_stops_on_nominal, "notes": self.notes}
