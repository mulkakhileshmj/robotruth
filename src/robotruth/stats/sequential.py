"""Anytime-valid sequential comparison of two policies.

Why: a fixed-N test is only valid if you decide N in advance and never peek. Robot labs
peek after every rollout, which is exactly what you should do when robot time is scarce,
but it breaks classical p-values. Confidence sequences fix this: the interval is valid at
every time step simultaneously, so you may stop the moment it excludes zero (or your
margin) and the error rate is still controlled. TRI's STEP procedure (RSS 2025,
arXiv 2503.10966) reports up to 32 percent fewer trials from this idea.

Implementation: the predictable plug-in empirical Bernstein confidence sequence of
Waudby-Smith and Ramdas (2023, "Estimating means of bounded random variables by betting",
JRSS-B), for observations in [0, 1]. For a paired comparison we observe
d_i = b_i - a_i in {-1, 0, 1} and map x_i = (d_i + 1) / 2 into [0, 1]; the interval for
E[d] is 2 * CS(x) - 1.

The sequence is closed form and cheap: for lambda_t predictable (depends only on the past),

    center_t = sum(lambda_i x_i) / sum(lambda_i)
    radius_t = (log(2 / alpha) + sum(v_i psi_E(lambda_i))) / sum(lambda_i)
    v_i = 4 (x_i - mu_hat_{i-1})^2,  psi_E(l) = (-log(1 - l) - l) / 4

with lambda_t = min( sqrt(2 log(2/alpha) / (sigma_hat_{t-1}^2 t log(t + 1))), 0.5 ).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import numpy as np


def _psi_e(lam: float) -> float:
    return (-np.log1p(-lam) - lam) / 4.0


@dataclass
class ConfidenceSequence:
    """Running empirical-Bernstein confidence sequence for the mean of x_t in [0, 1]."""

    alpha: float = 0.05
    lam_cap: float = 0.5
    t: int = 0
    _sum_lx: float = 0.0
    _sum_l: float = 0.0
    _sum_vpsi: float = 0.0
    _sum_x: float = 0.0
    _sum_x2: float = 0.0
    _prior_mu: float = 0.5
    _prior_var: float = 0.25
    history: list[tuple[float, float, float]] = field(default_factory=list)

    def _mu_hat(self) -> float:
        # Regularised running mean (prior weight 1) so early lambdas are stable.
        return (self._prior_mu + self._sum_x) / (self.t + 1)

    def _sigma2_hat(self) -> float:
        mu = self._mu_hat()
        return (self._prior_var + self._sum_x2 - 2 * mu * self._sum_x + self.t * mu * mu) / (self.t + 1)

    def update(self, x: float) -> tuple[float, float, float]:
        if not 0.0 <= x <= 1.0:
            raise ValueError("observations must be in [0, 1]")
        t_next = self.t + 1
        sigma2 = max(self._sigma2_hat(), 1e-12)
        lam = min(np.sqrt(2 * np.log(2 / self.alpha) / (sigma2 * t_next * np.log(t_next + 1))), self.lam_cap)
        mu_prev = self._mu_hat()
        v = 4.0 * (x - mu_prev) ** 2  # WSR 2023, Thm 2: v_i = 4 (x_i - mu_hat_{i-1})^2
        self._sum_lx += lam * x
        self._sum_l += lam
        self._sum_vpsi += v * _psi_e(lam)
        self._sum_x += x
        self._sum_x2 += x * x
        self.t = t_next
        centre = self._sum_lx / self._sum_l
        radius = (np.log(2 / self.alpha) + self._sum_vpsi) / self._sum_l
        lo, hi = max(0.0, centre - radius), min(1.0, centre + radius)
        self.history.append((centre, lo, hi))
        return centre, lo, hi

    @property
    def interval(self) -> tuple[float, float, float]:
        if not self.history:
            return 0.5, 0.0, 1.0
        return self.history[-1]


@dataclass
class SequentialResult:
    decision: Literal["B_better", "A_better", "no_difference_within_margin", "undecided"]
    n: int
    estimate: float
    lower: float
    upper: float
    alpha: float
    margin: float
    trace: list[tuple[int, float, float, float]]

    def __str__(self) -> str:
        return (f"{self.decision} after n={self.n}: diff(B-A)={self.estimate:+.3f} "
                f"[{self.lower:+.3f}, {self.upper:+.3f}] (anytime-valid {100*(1-self.alpha):.0f}%)")


def sequential_paired_test(a: Iterable[float], b: Iterable[float], alpha: float = 0.05,
                           margin: float = 0.0, min_n: int = 5, stop_early: bool = True) -> SequentialResult:
    """Anytime-valid paired comparison of success indicators (or graded scores in [0, 1]).

    Decides B_better when the whole confidence sequence for E[b - a] lies above `margin`,
    A_better when it lies below `-margin`, and no_difference_within_margin when it lies
    inside (-margin, margin) with margin > 0. Otherwise undecided. With `stop_early`, the
    trace ends at the first decision; without it, the full sequence is run and the final
    interval is reported.
    """
    cs = ConfidenceSequence(alpha=alpha)
    trace: list[tuple[int, float, float, float]] = []
    decision: str = "undecided"
    for i, (xa, xb) in enumerate(zip(a, b), start=1):
        d = float(xb) - float(xa)
        if not -1.0 <= d <= 1.0:
            raise ValueError("scores must be in [0, 1]")
        c, lo, hi = cs.update((d + 1.0) / 2.0)
        est, dlo, dhi = 2 * c - 1, 2 * lo - 1, 2 * hi - 1
        trace.append((i, est, dlo, dhi))
        if i >= min_n:
            if dlo > margin:
                decision = "B_better"
            elif dhi < -margin:
                decision = "A_better"
            elif margin > 0 and dlo > -margin and dhi < margin:
                decision = "no_difference_within_margin"
            if decision != "undecided" and stop_early:
                break
    if not trace:
        return SequentialResult("undecided", 0, 0.0, -1.0, 1.0, alpha, margin, [])
    n, est, dlo, dhi = trace[-1]
    return SequentialResult(decision, n, est, dlo, dhi, alpha, margin, trace)  # type: ignore[arg-type]
