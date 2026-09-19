"""Time-uniform conformal thresholds for sequential failure detection.

The question a runtime monitor has to answer honestly is not "is this score unusual at
this instant" but "will I raise a false alarm at any point during a successful episode".
A per-step threshold at level alpha fires with probability far above alpha over a few
hundred steps. The two calibrators here control the probability of ANY false alarm over a
whole nominal episode, which is what FAIL-Detect (arXiv 2503.08558) calls a time-uniform
or sequential guarantee.

Both are split-conformal: they use held-out nominal score trajectories that were NOT used
to fit the scorer. With n such trajectories and a conformal quantile at rank
ceil((n + 1)(1 - alpha)), a fresh nominal episode that is exchangeable with them alarms
with probability at most alpha. No distributional assumption is made.

Method "max" (FAIL-Detect): take the maximum score over each nominal trajectory, and use
one constant threshold at the conformal quantile of those maxima. Simple and tight.

Method "bonferroni": split the horizon into time bins and alpha equally across bins; in
each bin calibrate on the per-trajectory maximum within that bin at level alpha / n_bins.
Thresholds then vary over the episode (looser early when policies are settling, tighter
later, or the other way round). Needs more calibration data: rank ceil((n + 1)(1 - alpha /
n_bins)) must be at most n, otherwise the bin is "saturated", the largest observed value is
used and the guarantee no longer holds. The report says so.

Measured caveat (2026-09-19, examples/validation/2026-09-19): on a live ACT policy with
44 to 49 nominal calibration episodes, "bonferroni" alarmed on 3 to 4 of 17 held-out
successful episodes in every draw, roughly 20 percent against a nominal alpha of 0.05,
because every bin was saturated at that sample size. "max" alarmed on 0, 0 and 3 of 17
across the same three draws. Use "max" unless you have enough calibration episodes to
clear saturation in every bin, and read the saturation flag in the report before trusting
a bonferroni threshold.

ContrastSetCalibration is the SAFECAST idea (arXiv 2608.04246): calibrate on the union of
nominal episodes and benign-shift episodes (new lighting, a distractor on the table, a
different operator resetting the scene) so that the detector does not fire on harmless
change. It reports how much the threshold moved and how often the benign episodes would
have alarmed under the nominal-only threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


def conformal_quantile(values: np.ndarray, alpha: float) -> tuple[float, bool, float]:
    """Finite-sample conformal upper quantile.

    Returns (threshold, saturated, achievable_alpha). saturated is True when the requested
    alpha needs a rank beyond the number of calibration values; in that case the largest
    value is returned and achievable_alpha = 1 / (n + 1) is the smallest alpha this many
    calibration values can certify.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n == 0:
        raise ValueError("no calibration values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float(v[-1]), True, 1.0 / (n + 1)
    return float(v[rank - 1]), False, alpha


def _as_trajectories(trajectories: Sequence[np.ndarray]) -> list[np.ndarray]:
    out = [np.asarray(t, dtype=np.float64).ravel() for t in trajectories]
    if not out:
        raise ValueError("no trajectories")
    if any(t.size == 0 for t in out):
        raise ValueError("empty trajectory")
    return out


@dataclass
class SequentialConformal:
    """Time-uniform threshold calibrated on nominal score trajectories.

    alpha: bound on the probability of any false alarm over a nominal episode.
    method: "max" or "bonferroni".
    n_bins: time bins for the bonferroni method.
    """

    alpha: float = 0.05
    method: str = "max"
    n_bins: int = 4
    horizon: Optional[int] = None
    thresholds_: Optional[np.ndarray] = None      # per bin (bonferroni) or length 1 (max)
    bin_edges_: Optional[np.ndarray] = None       # step indices, length n_bins + 1
    saturated_: bool = False
    achievable_alpha_: Optional[float] = None
    n_calibration_: int = 0
    calibration_alarm_fraction_: Optional[float] = None
    fitted: bool = False

    def fit(self, nominal_score_trajectories: Sequence[np.ndarray], alpha: Optional[float] = None,
            horizon: Optional[int] = None) -> "SequentialConformal":
        trajs = _as_trajectories(nominal_score_trajectories)
        if alpha is not None:
            self.alpha = float(alpha)
        self.horizon = int(horizon) if horizon is not None else int(max(t.size for t in trajs))
        self.n_calibration_ = len(trajs)
        if self.method == "max":
            maxima = np.array([t.max() for t in trajs])
            thr, sat, ach = conformal_quantile(maxima, self.alpha)
            self.thresholds_ = np.array([thr])
            self.bin_edges_ = np.array([0, self.horizon])
            self.saturated_, self.achievable_alpha_ = sat, ach
        elif self.method == "bonferroni":
            nb = max(1, int(self.n_bins))
            edges = np.linspace(0, self.horizon, nb + 1).round().astype(int)
            edges[-1] = max(edges[-1], self.horizon)
            per_bin_alpha = self.alpha / nb
            thr, sat_any, ach_total = [], False, 0.0
            for b in range(nb):
                lo, hi = int(edges[b]), int(edges[b + 1])
                bin_max = [t[lo:hi].max() for t in trajs if t[lo:hi].size > 0]
                if not bin_max:
                    thr.append(thr[-1] if thr else float("inf"))
                    ach_total += per_bin_alpha
                    continue
                v, sat, ach = conformal_quantile(np.array(bin_max), per_bin_alpha)
                thr.append(v)
                sat_any = sat_any or sat
                ach_total += ach
            self.thresholds_ = np.array(thr)
            self.bin_edges_ = edges
            self.saturated_ = sat_any
            self.achievable_alpha_ = float(ach_total)
        else:
            raise ValueError("method must be 'max' or 'bonferroni'")
        self.fitted = True
        self.calibration_alarm_fraction_ = float(np.mean([self.alarms(t) for t in trajs]))
        return self

    def threshold(self, t: int) -> float:
        """Threshold that applies at step index t (0-based). Beyond the horizon the last bin holds."""
        if not self.fitted:
            raise RuntimeError("SequentialConformal.fit() first")
        if self.thresholds_.size == 1:
            return float(self.thresholds_[0])
        b = int(np.searchsorted(self.bin_edges_, t, side="right") - 1)
        b = min(max(b, 0), self.thresholds_.size - 1)
        return float(self.thresholds_[b])

    def thresholds(self, horizon: Optional[int] = None) -> np.ndarray:
        h = int(horizon if horizon is not None else self.horizon)
        return np.array([self.threshold(t) for t in range(h)])

    def alarms(self, trajectory: np.ndarray) -> bool:
        """Would this score trajectory raise an alarm anywhere?"""
        t = np.asarray(trajectory, dtype=np.float64).ravel()
        return bool(np.any(t > self.thresholds(t.size)))

    def to_dict(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "method": self.method, "n_bins": self.n_bins, "horizon": self.horizon,
                "thresholds": None if self.thresholds_ is None else self.thresholds_.tolist(),
                "bin_edges": None if self.bin_edges_ is None else self.bin_edges_.tolist(),
                "saturated": self.saturated_, "achievable_alpha": self.achievable_alpha_,
                "n_calibration": self.n_calibration_,
                "calibration_alarm_fraction": self.calibration_alarm_fraction_}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SequentialConformal":
        c = cls(alpha=float(d["alpha"]), method=d.get("method", "max"), n_bins=int(d.get("n_bins", 4)),
                horizon=d.get("horizon"))
        if d.get("thresholds") is not None:
            c.thresholds_ = np.asarray(d["thresholds"], dtype=np.float64)
            c.bin_edges_ = np.asarray(d["bin_edges"], dtype=int)
            c.saturated_ = bool(d.get("saturated", False))
            c.achievable_alpha_ = d.get("achievable_alpha")
            c.n_calibration_ = int(d.get("n_calibration", 0))
            c.calibration_alarm_fraction_ = d.get("calibration_alarm_fraction")
            c.fitted = True
        return c

    def summary(self) -> str:
        if not self.fitted:
            return "uncalibrated"
        thr = self.thresholds_
        rng = f"{thr[0]:.3f}" if thr.size == 1 else f"{thr.min():.3f} to {thr.max():.3f}"
        s = f"{self.method} threshold {rng}, alpha={self.alpha:g}, n={self.n_calibration_}"
        if self.saturated_:
            s += f" (SATURATED: only alpha>={self.achievable_alpha_:.3f} certifiable with this many episodes)"
        return s


@dataclass
class ContrastReport:
    """How much including benign shifts moved the threshold."""

    nominal_only: SequentialConformal
    union: SequentialConformal
    n_nominal: int
    n_benign: int
    benign_alarm_fraction_under_nominal: float
    benign_alarm_fraction_under_union: float
    threshold_shift: float          # mean(union thresholds) - mean(nominal thresholds)
    threshold_shift_relative: float  # shift / |mean nominal| (0 if nominal mean is 0)

    def to_dict(self) -> dict[str, Any]:
        return {"n_nominal": self.n_nominal, "n_benign": self.n_benign,
                "benign_alarm_fraction_under_nominal": self.benign_alarm_fraction_under_nominal,
                "benign_alarm_fraction_under_union": self.benign_alarm_fraction_under_union,
                "threshold_shift": self.threshold_shift, "threshold_shift_relative": self.threshold_shift_relative,
                "nominal_only": self.nominal_only.to_dict(), "union": self.union.to_dict()}

    def to_markdown(self) -> str:
        return "\n".join([
            "## Contrast-set calibration",
            "",
            f"- Nominal episodes: {self.n_nominal}; benign-shift episodes: {self.n_benign}",
            f"- Nominal-only: {self.nominal_only.summary()}",
            f"- Union: {self.union.summary()}",
            f"- Threshold moved by {self.threshold_shift:+.3f} ({100 * self.threshold_shift_relative:+.1f}%)",
            f"- Benign episodes that would alarm under the nominal-only threshold: "
            f"{100 * self.benign_alarm_fraction_under_nominal:.0f}%; under the union threshold: "
            f"{100 * self.benign_alarm_fraction_under_union:.0f}%",
        ])


@dataclass
class ContrastSetCalibration:
    """SAFECAST-style calibration on nominal plus benign-shift trajectories.

    fit(nominal, benign) fits one calibrator on nominal only (for the report) and one on the
    union, and keeps the union one as `calibrator`. The union is what the Guard should use:
    the alpha guarantee then covers the benign conditions too.
    """

    alpha: float = 0.05
    method: str = "max"
    n_bins: int = 4
    calibrator: Optional[SequentialConformal] = None
    report: Optional[ContrastReport] = None

    def fit(self, nominal: Sequence[np.ndarray], benign: Sequence[np.ndarray],
            alpha: Optional[float] = None, horizon: Optional[int] = None) -> "ContrastSetCalibration":
        if alpha is not None:
            self.alpha = float(alpha)
        nom = _as_trajectories(nominal)
        ben = _as_trajectories(benign) if len(benign) else []
        h = horizon if horizon is not None else max(t.size for t in nom + ben)
        nominal_only = SequentialConformal(self.alpha, self.method, self.n_bins).fit(nom, horizon=h)
        union = SequentialConformal(self.alpha, self.method, self.n_bins).fit(nom + ben, horizon=h)
        ben_nom = float(np.mean([nominal_only.alarms(t) for t in ben])) if ben else 0.0
        ben_uni = float(np.mean([union.alarms(t) for t in ben])) if ben else 0.0
        m_nom, m_uni = float(nominal_only.thresholds_.mean()), float(union.thresholds_.mean())
        shift = m_uni - m_nom
        rel = shift / abs(m_nom) if abs(m_nom) > 1e-12 else 0.0
        self.calibrator = union
        self.report = ContrastReport(nominal_only, union, len(nom), len(ben), ben_nom, ben_uni, shift, rel)
        return self

    def threshold(self, t: int) -> float:
        if self.calibrator is None:
            raise RuntimeError("ContrastSetCalibration.fit() first")
        return self.calibrator.threshold(t)
