"""Calibrated drift detection over a population of unit fingerprints.

`diff_unit` compares two fingerprints against fixed tolerances, which is the right tool when
you know what a meaningful change is. Often you do not, and what you have instead is a pile
of fingerprints from a working cell. This fits the nominal spread and sets a threshold that
holds a stated false-alarm rate.

The threshold is a split-conformal quantile rather than a multiple of the standard deviation.
Measured on five public datasets, a three-sigma rule on per-joint offsets false-alarmed on 5
to 30 percent of clean episodes, because per-joint estimates are neither Gaussian nor
independent and the maximum over joints has a much heavier tail than each one alone.
Conformal calibration makes no distributional assumption: it takes the rank that guarantees
the rate, so the false-alarm rate is at most alpha by exchangeability alone.

    det = DriftDetector(alpha=0.05).fit(nominal_fingerprints)
    det.alarms(new_fingerprint)      # True if this unit has drifted
    det.score(new_fingerprint)       # how far out, in calibrated units

Fitting needs at least 1/alpha - 1 fingerprints to certify the requested alpha (19 at 0.05).
Below that the detector reports itself saturated rather than claiming a rate it cannot hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from robotruth.guard.conformal import conformal_quantile

FIELDS = ("steady_error", "gain", "lag_s", "backlash", "rmse")
_EPS = 1e-12


def _values(fp: Any, fields: tuple[str, ...]) -> dict[tuple[str, str], float]:
    """{(joint, field): value} for one fingerprint."""
    out: dict[tuple[str, str], float] = {}
    for joint, js in fp.joints.items():
        for f in fields:
            v = getattr(js, f, None)
            if v is not None and np.isfinite(v):
                out[(joint, f)] = float(v)
    return out


@dataclass
class DriftDetector:
    """Conformally calibrated drift alarm over unit fingerprints.

    alpha: target false-alarm rate on fingerprints exchangeable with the fitting set.
    fields: which per-joint statistics to watch. The default watches all of them; pass
        ("steady_error",) to watch calibration offset alone.
    split: fraction of the nominal set used to estimate each statistic's centre and scale,
        the rest calibrates the threshold. Estimating and calibrating on the same
        fingerprints gives an optimistically low threshold and more alarms than alpha.
    """

    alpha: float = 0.05
    fields: tuple[str, ...] = FIELDS
    split: float = 0.5
    seed: int = 0
    center_: dict[str, float] = field(default_factory=dict)
    scale_: dict[str, float] = field(default_factory=dict)
    threshold_: Optional[float] = None
    saturated_: bool = False
    achievable_alpha_: Optional[float] = None
    n_fit_: int = 0
    n_calibration_: int = 0

    @property
    def fitted(self) -> bool:
        return self.threshold_ is not None

    def _raw(self, fp: Any) -> float:
        """Largest standardised deviation across the watched statistics."""
        worst = 0.0
        for (joint, f), v in _values(fp, self.fields).items():
            key = f"{joint}.{f}"
            if key not in self.center_:
                continue
            worst = max(worst, abs(v - self.center_[key]) / self.scale_[key])
        return float(worst)

    def fit(self, nominal: Iterable[Any]) -> "DriftDetector":
        """nominal: fingerprints from a cell believed to be working."""
        fps = list(nominal)
        if len(fps) < 4:
            raise ValueError("need at least four nominal fingerprints")
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(fps))
        n_fit = min(max(int(round(self.split * len(fps))), 2), len(fps) - 2)
        fit_set = [fps[i] for i in order[:n_fit]]
        cal_set = [fps[i] for i in order[n_fit:]]

        pooled: dict[str, list[float]] = {}
        for fp in fit_set:
            for (joint, f), v in _values(fp, self.fields).items():
                pooled.setdefault(f"{joint}.{f}", []).append(v)
        # Median and scaled MAD: a single badly drifted unit in the fitting set would drag a
        # mean and inflate a standard deviation enough to hide the drift it represents.
        self.center_, self.scale_ = {}, {}
        for key, vals in pooled.items():
            a = np.asarray(vals, dtype=np.float64)
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med))) * 1.4826
            self.center_[key] = med
            self.scale_[key] = max(mad, float(a.std()) * 1e-3, _EPS)

        scores = np.array([self._raw(fp) for fp in cal_set], dtype=np.float64)
        self.threshold_, self.saturated_, self.achievable_alpha_ = conformal_quantile(scores, self.alpha)
        self.n_fit_, self.n_calibration_ = len(fit_set), len(cal_set)
        return self

    def score(self, fp: Any) -> float:
        """Deviation relative to the threshold: above 0 means alarm."""
        if not self.fitted:
            raise RuntimeError("DriftDetector.fit() first")
        return self._raw(fp) - float(self.threshold_)

    def alarms(self, fp: Any) -> bool:
        return self.score(fp) > 0.0

    def explain(self, fp: Any, top: int = 5) -> list[tuple[str, float, float]]:
        """(statistic, value, standardised deviation), worst first. Which joint moved."""
        if not self.fitted:
            raise RuntimeError("DriftDetector.fit() first")
        rows = []
        for (joint, f), v in _values(fp, self.fields).items():
            key = f"{joint}.{f}"
            if key in self.center_:
                rows.append((key, v, abs(v - self.center_[key]) / self.scale_[key]))
        rows.sort(key=lambda r: -r[2])
        return rows[:top]

    def to_dict(self) -> dict[str, Any]:
        return {"format": "robotruth.fingerprint.drift/0.1", "alpha": self.alpha,
                "fields": list(self.fields), "split": self.split, "seed": self.seed,
                "center": self.center_, "scale": self.scale_, "threshold": self.threshold_,
                "saturated": self.saturated_, "achievable_alpha": self.achievable_alpha_,
                "n_fit": self.n_fit_, "n_calibration": self.n_calibration_}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DriftDetector":
        det = cls(alpha=float(d.get("alpha", 0.05)), fields=tuple(d.get("fields", FIELDS)),
                  split=float(d.get("split", 0.5)), seed=int(d.get("seed", 0)))
        det.center_ = {k: float(v) for k, v in d.get("center", {}).items()}
        det.scale_ = {k: float(v) for k, v in d.get("scale", {}).items()}
        det.threshold_ = None if d.get("threshold") is None else float(d["threshold"])
        det.saturated_ = bool(d.get("saturated", False))
        det.achievable_alpha_ = d.get("achievable_alpha")
        det.n_fit_ = int(d.get("n_fit", 0))
        det.n_calibration_ = int(d.get("n_calibration", 0))
        return det

    def summary(self) -> str:
        if not self.fitted:
            return "DriftDetector: not fitted"
        s = (f"DriftDetector: alpha={self.alpha:g}, threshold={self.threshold_:.3f} standardised "
             f"units, fitted on {self.n_fit_} and calibrated on {self.n_calibration_} fingerprints")
        if self.saturated_:
            s += (f"; SATURATED, too few calibration fingerprints for alpha={self.alpha:g}, "
                  f"smallest certifiable is {self.achievable_alpha_:.3f}")
        return s
