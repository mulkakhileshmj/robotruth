"""Time-to-success comparison with censoring.

Binary success throws away information. Two policies at 90 percent success where one
finishes in 20 seconds and the other in 60 are not the same policy. PhAIL (arXiv
2605.29710) shows a time-to-success comparison reaches 80 percent power at N of 30 where
binary success needs 600 to 1,500.

Timeouts are right-censored: we know the policy had not succeeded by the timeout, not when
it would have. Kaplan-Meier handles that; the log-rank test compares two censored curves.
A plain KS test is also provided for the uncensored case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class KMCurve:
    times: np.ndarray
    survival: np.ndarray  # probability of NOT having succeeded by time t
    at_risk: np.ndarray
    events: np.ndarray

    @property
    def success_cdf(self) -> np.ndarray:
        return 1.0 - self.survival

    def median_time(self) -> float:
        idx = np.where(self.survival <= 0.5)[0]
        return float(self.times[idx[0]]) if idx.size else float("inf")


def kaplan_meier(times: np.ndarray, succeeded: np.ndarray) -> KMCurve:
    """times: time of success, or timeout duration if not succeeded. succeeded: 1 if success."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(succeeded, dtype=int)
    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]
    uniq = np.unique(t)
    surv, at_risk, events = [], [], []
    s = 1.0
    for u in uniq:
        n_risk = int(np.sum(t >= u))
        d = int(np.sum((t == u) & (e == 1)))
        if n_risk > 0:
            s *= 1.0 - d / n_risk
        surv.append(s)
        at_risk.append(n_risk)
        events.append(d)
    return KMCurve(uniq, np.array(surv), np.array(at_risk), np.array(events))


@dataclass(frozen=True)
class LogRank:
    statistic: float
    p_value: float
    n_a: int
    n_b: int
    events_a: int
    events_b: int
    median_a: float
    median_b: float

    def __str__(self) -> str:
        return (f"log-rank chi2={self.statistic:.3f}, p={self.p_value:.4f}; "
                f"median time A={self.median_a:.1f}, B={self.median_b:.1f}")


def logrank_test(times_a, succ_a, times_b, succ_b) -> LogRank:
    ta, ea = np.asarray(times_a, float), np.asarray(succ_a, int)
    tb, eb = np.asarray(times_b, float), np.asarray(succ_b, int)
    all_t = np.unique(np.concatenate([ta[ea == 1], tb[eb == 1]]))
    o_minus_e = 0.0
    var = 0.0
    for u in all_t:
        n_a = np.sum(ta >= u)
        n_b = np.sum(tb >= u)
        n = n_a + n_b
        d_a = np.sum((ta == u) & (ea == 1))
        d_b = np.sum((tb == u) & (eb == 1))
        d = d_a + d_b
        if n <= 1:
            continue
        e_a = d * n_a / n
        o_minus_e += d_a - e_a
        var += d * (n_a / n) * (n_b / n) * (n - d) / (n - 1)
    stat = float(o_minus_e ** 2 / var) if var > 0 else 0.0
    p = float(sps.chi2.sf(stat, df=1)) if var > 0 else 1.0
    return LogRank(stat, p, ta.size, tb.size, int(ea.sum()), int(eb.sum()),
                   kaplan_meier(ta, ea).median_time(), kaplan_meier(tb, eb).median_time())


def ks_time_to_success(times_a, succ_a, times_b, succ_b, timeout: float | None = None):
    """Two-sample KS on time-to-success where failures are placed at the timeout (PhAIL style).

    Only appropriate when the timeout is identical for both arms; otherwise use logrank_test.
    """
    ta, ea = np.asarray(times_a, float), np.asarray(succ_a, int)
    tb, eb = np.asarray(times_b, float), np.asarray(succ_b, int)
    if timeout is None:
        timeout = float(max(ta.max(), tb.max()))
    xa = np.where(ea == 1, ta, timeout)
    xb = np.where(eb == 1, tb, timeout)
    return sps.ks_2samp(xa, xb)
