"""Bradley-Terry ranking from pairwise outcomes, with task buckets.

RoboArena (CoRL 2025) found a fixed 17-task benchmark correlated only r=0.55 with the
oracle ranking while task-aware pairwise comparison reached r=0.95. Pairwise outcomes
inside the same task bucket cancel the bucket's difficulty; pooling across buckets does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class BTResult:
    policies: list[str]
    strength: np.ndarray            # log-strengths, mean zero
    win_prob: np.ndarray            # [i, j] = P(i beats j)
    n_comparisons: int
    converged: bool
    se: Optional[np.ndarray] = None  # bootstrap standard errors of log-strength

    def ranking(self) -> list[tuple[str, float, float]]:
        order = np.argsort(-self.strength)
        return [(self.policies[i], float(self.strength[i]), float(self.se[i]) if self.se is not None else float("nan")) for i in order]


def _fit(wins: np.ndarray, iters: int, tol: float) -> tuple[np.ndarray, bool]:
    """MM algorithm (Hunter 2004) for BT strengths. wins[i, j] = number of times i beat j."""
    k = wins.shape[0]
    p = np.ones(k)
    n_pairs = wins + wins.T
    converged = False
    for _ in range(iters):
        denom = np.zeros(k)
        for i in range(k):
            mask = n_pairs[i] > 0
            denom[i] = np.sum(n_pairs[i, mask] / (p[i] + p[mask]))
        w = wins.sum(axis=1)
        new = np.where(denom > 0, w / np.maximum(denom, 1e-12), p)
        new = np.maximum(new, 1e-9)
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(p))) < tol:
            p = new
            converged = True
            break
        p = new
    return np.log(p), converged


def bradley_terry(comparisons: Iterable[tuple[str, str, float, Optional[str]]], iters: int = 2000,
                  tol: float = 1e-8, n_boot: int = 200, seed: int = 0) -> BTResult:
    """Fit Bradley-Terry strengths.

    comparisons: (policy_i, policy_j, outcome, task_bucket) where outcome is 1.0 if i won,
    0.0 if j won, 0.5 for a tie. task_bucket may be None. Only comparisons within the same
    bucket are informative about relative strength; the model itself is bucket-agnostic
    because each comparison already conditions on its bucket.
    """
    comps = list(comparisons)
    names = sorted({c[0] for c in comps} | {c[1] for c in comps})
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)

    def build(sub) -> np.ndarray:
        wins = np.zeros((k, k))
        for i, j, out, _ in sub:
            wins[idx[i], idx[j]] += out
            wins[idx[j], idx[i]] += 1.0 - out
        return wins

    strength, conv = _fit(build(comps), iters, tol)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = [comps[i] for i in rng.integers(0, len(comps), len(comps))]
        s, _ = _fit(build(sample), iters // 4, tol * 10)
        boots.append(s)
    se = np.std(np.array(boots), axis=0) if boots else None
    diff = strength[:, None] - strength[None, :]
    win_prob = 1.0 / (1.0 + np.exp(-diff))
    return BTResult(names, strength, win_prob, len(comps), conv, se)
