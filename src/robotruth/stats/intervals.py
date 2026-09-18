"""Confidence intervals for success rates and their differences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    n: int
    method: str
    alpha: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def __str__(self) -> str:
        pct = 100 * (1 - self.alpha)
        return f"{self.estimate:.3f} [{self.lower:.3f}, {self.upper:.3f}] ({pct:.0f}% {self.method}, n={self.n})"


def wilson(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion. Good coverage even at small n."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else float(max(0.0, centre - half))
    hi = 1.0 if k == n else float(min(1.0, centre + half))
    return Interval(float(p), lo, hi, n, "Wilson", alpha)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Exact (conservative) binomial interval."""
    if n <= 0:
        raise ValueError("n must be positive")
    lo = 0.0 if k == 0 else sps.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sps.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return Interval(k / n, float(lo), float(hi), n, "Clopper-Pearson", alpha)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Interval:
    """Wald-type interval for the difference of paired binary outcomes p_b - p_a.

    Uses the paired (McNemar-style) variance, which is smaller than the unpaired one
    when outcomes on the same trial condition are correlated. That is why interleaved
    paired designs need fewer trials.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal length")
    n = a.size
    d = b - a
    est = d.mean()
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    z = sps.norm.ppf(1 - alpha / 2)
    return Interval(float(est), float(max(-1.0, est - z * se)), float(min(1.0, est + z * se)), n, "paired Wald", alpha)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05, paired: bool = False,
                      n_boot: int = 10_000, seed: int = 0) -> Interval:
    """Percentile bootstrap interval for mean(b) - mean(a). Works for binary or graded scores."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if paired:
        if a.shape != b.shape:
            raise ValueError("paired arrays must have equal length")
        d = b - a
        idx = rng.integers(0, d.size, size=(n_boot, d.size))
        boots = d[idx].mean(axis=1)
        est = d.mean()
        n = d.size
    else:
        ia = rng.integers(0, a.size, size=(n_boot, a.size))
        ib = rng.integers(0, b.size, size=(n_boot, b.size))
        boots = b[ib].mean(axis=1) - a[ia].mean(axis=1)
        est = b.mean() - a.mean()
        n = a.size + b.size
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(est), float(lo), float(hi), int(n), "bootstrap" + (" paired" if paired else ""), alpha)
