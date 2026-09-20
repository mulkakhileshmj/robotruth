"""Failure clustering: what do the broken scenarios have in common?

A list of 74 hashes and 74 videos is a complaint. "Fails when the cube is rotated past 18
degrees, 41 of 52 there against 33 of 148 elsewhere" is a bug report. This module turns the
first into the second.

The method is deliberately the least clever one that works, because the output has to be
read and acted on by an engineer:

- Candidate rules are axis-aligned: one factor against one threshold, optionally two such
  conditions joined by AND. A rule an engineer cannot restate in a sentence is not useful,
  whatever its accuracy.
- Thresholds come from the observed quantiles of that factor, so every rule has support.
- A rule is reported only when the difference between inside and outside survives a Fisher
  exact test at the stated level, and both rates are printed with Wilson intervals. A
  hotspot is a claim about a region, and claims here carry evidence.
- Rules are ranked by lift, with ties broken toward larger support, and near-duplicates on
  the same factor are suppressed.

This finds where a policy is weak. It does not explain why, and it cannot discover a region
the battery never sampled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps

from robotruth.stats.intervals import wilson

MIN_SUPPORT = 8          # below this a "region" is an anecdote
MIN_SUPPORT_PAIR = 15    # a conjunction carves harder, so it needs more evidence
MAX_RULES = 5


@dataclass(frozen=True)
class Condition:
    factor: str
    op: str              # ">" or "<="
    threshold: float

    def holds(self, params: dict) -> bool:
        """Evaluate against a scenario's raw parameters.

        A condition may name a derived coordinate such as `abs_cube_yaw_deg`, which is not
        stored in the battery. Resolve it here so a rule stays applicable to any scenario,
        including ones sampled later to probe the region it describes.
        """
        name = self.factor
        if name in params:
            v = params[name]
        elif name.startswith("abs_") and name[4:] in params:
            v = abs(params[name[4:]])
        else:
            return False
        return v > self.threshold if self.op == ">" else v <= self.threshold

    def __str__(self) -> str:
        return f"{self.factor} {self.op} {self.threshold:.3g}"


@dataclass
class Hotspot:
    conditions: tuple[Condition, ...]
    n_inside: int
    n_broken_inside: int
    n_outside: int
    n_broken_outside: int
    p_value: float
    inside_rate: object = field(default=None)       # robotruth Interval
    outside_rate: object = field(default=None)

    @property
    def description(self) -> str:
        return " and ".join(str(c) for c in self.conditions)

    @property
    def lift(self) -> float:
        out = self.n_broken_outside / self.n_outside if self.n_outside else 0.0
        ins = self.n_broken_inside / self.n_inside if self.n_inside else 0.0
        return ins / out if out > 0 else float("inf") if ins > 0 else 1.0

    def holds(self, params: dict) -> bool:
        return all(c.holds(params) for c in self.conditions)

    def __str__(self) -> str:
        return (f"{self.description}: {self.n_broken_inside}/{self.n_inside} broken "
                f"({100 * self.inside_rate.estimate:.1f}%) against "
                f"{self.n_broken_outside}/{self.n_outside} elsewhere "
                f"({100 * self.outside_rate.estimate:.1f}%)")

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "conditions": [{"factor": c.factor, "op": c.op, "threshold": c.threshold}
                           for c in self.conditions],
            "n_inside": self.n_inside, "n_broken_inside": self.n_broken_inside,
            "n_outside": self.n_outside, "n_broken_outside": self.n_broken_outside,
            "inside_rate": {"estimate": self.inside_rate.estimate,
                            "lower": self.inside_rate.lower, "upper": self.inside_rate.upper},
            "outside_rate": {"estimate": self.outside_rate.estimate,
                             "lower": self.outside_rate.lower, "upper": self.outside_rate.upper},
            "lift": self.lift, "p_value": self.p_value,
        }


def _evaluate(mask_inside: np.ndarray, broken: np.ndarray, alpha: float,
              min_support: int = MIN_SUPPORT) -> tuple | None:
    n_in = int(mask_inside.sum())
    n_out = int((~mask_inside).sum())
    if n_in < min_support or n_out < min_support:
        return None
    b_in = int(broken[mask_inside].sum())
    b_out = int(broken[~mask_inside].sum())
    if b_in == 0:
        return None
    # one-sided: is failure ENRICHED inside the region
    table = [[b_in, n_in - b_in], [b_out, n_out - b_out]]
    try:
        _, p = sps.fisher_exact(table, alternative="greater")
    except Exception:
        return None
    if p > alpha:
        return None
    if (b_in / n_in) <= (b_out / n_out):
        return None
    return n_in, b_in, n_out, b_out, float(p)


def find_hotspots(scenario_params: dict[str, dict], broken_hashes: set[str],
                  all_hashes: list[str], alpha: float = 0.05,
                  max_rules: int = MAX_RULES, pairs: bool = True) -> list[Hotspot]:
    """Regions of the factor space where breakage is concentrated.

    `scenario_params` maps scenario hash to its named factors; `broken_hashes` are the
    scenarios that regressed (or failed); `all_hashes` is the population they came from.
    Returns an empty list when the battery has no named factors, which is the honest
    answer for a seed-only battery rather than a fabricated one.
    """
    hashes = [h for h in all_hashes if h in scenario_params]
    if not hashes or not broken_hashes:
        return []

    from policyci.factors import scenario_factor_names
    factors = scenario_factor_names(scenario_params[hashes[0]])
    if not factors:
        return []

    broken = np.array([h in broken_hashes for h in hashes], dtype=bool)
    values = {f: np.array([float(scenario_params[h].get(f, np.nan)) for h in hashes])
              for f in factors}

    # Symmetric effects are common and one-sided rules cannot express them: a policy that
    # fails on a cube rotated either way is described by |yaw| > t, not yaw > t. Add the
    # magnitude as a derived coordinate wherever a factor's samples straddle zero.
    for f in list(factors):
        v = values[f]
        if np.isnan(v).any():
            continue
        if v.min() < 0 < v.max():
            values[f"abs_{f}"] = np.abs(v)
            factors = factors + [f"abs_{f}"]

    singles: list[Hotspot] = []
    for f in factors:
        v = values[f]
        if np.isnan(v).any():
            continue
        cuts = np.unique(np.quantile(v, np.linspace(0.1, 0.9, 9)))
        for t in cuts:
            for op in (">", "<="):
                mask = v > t if op == ">" else v <= t
                got = _evaluate(mask, broken, alpha)
                if got:
                    n_in, b_in, n_out, b_out, p = got
                    singles.append(Hotspot((Condition(f, op, float(t)),),
                                           n_in, b_in, n_out, b_out, p))

    best_per_factor: dict[str, Hotspot] = {}
    for h in singles:
        f = h.conditions[0].factor
        cur = best_per_factor.get(f)
        if cur is None or (h.lift, h.n_inside) > (cur.lift, cur.n_inside):
            best_per_factor[f] = h
    out = list(best_per_factor.values())

    # one round of conjunctions, seeded from the best single rule per factor
    if pairs and len(best_per_factor) >= 2:
        seeds = sorted(best_per_factor.values(), key=lambda h: -h.lift)[:3]
        for h1 in seeds:
            f1 = h1.conditions[0].factor
            for f2 in factors:
                if f2 == f1:
                    continue
                v2 = values[f2]
                if np.isnan(v2).any():
                    continue
                m1 = np.array([h1.conditions[0].holds(scenario_params[h]) for h in hashes])
                for t in np.unique(np.quantile(v2, np.linspace(0.2, 0.8, 7))):
                    for op in (">", "<="):
                        m2 = v2 > t if op == ">" else v2 <= t
                        got = _evaluate(m1 & m2, broken, alpha, min_support=MIN_SUPPORT_PAIR)
                        if got:
                            n_in, b_in, n_out, b_out, p = got
                            cand = Hotspot((h1.conditions[0], Condition(f2, op, float(t))),
                                           n_in, b_in, n_out, b_out, p)
                            if cand.lift > h1.lift * 1.15:   # must beat its own seed clearly
                                out.append(cand)

    for h in out:
        h.inside_rate = wilson(h.n_broken_inside, h.n_inside, alpha)
        h.outside_rate = wilson(h.n_broken_outside, h.n_outside, alpha)

    # One rule per combination of factors. Six phrasings of the same region is not six
    # findings, and an engineer reading six near-identical lines learns nothing.
    best: dict[tuple, Hotspot] = {}
    for h in out:
        key = tuple(sorted(c.factor for c in h.conditions))
        cur = best.get(key)
        if cur is None or (h.lift, h.n_inside) > (cur.lift, cur.n_inside):
            best[key] = h
    ranked = sorted(best.values(), key=lambda h: (-h.lift, -h.n_inside))

    # Drop a conjunction whose region is essentially a single-factor rule already reported.
    kept: list[Hotspot] = []
    for h in ranked:
        fset = {c.factor for c in h.conditions}
        if len(fset) > 1 and any(
                {c.factor for c in k.conditions} < fset and h.lift <= k.lift * 1.15
                for k in kept):
            continue
        kept.append(h)
    return kept[:max_rules]


def describe(hotspots: list[Hotspot], has_factors: bool) -> str:
    """One honest line for a report."""
    if not has_factors:
        return ("No named scene factors in this battery, so the broken scenarios cannot be "
                "described as a region. Use a factor battery to get hotspots.")
    if not hotspots:
        return ("No region of the factor space concentrates the failures beyond chance. "
                "The breakage looks spread across the sampled space.")
    return f"{len(hotspots)} region(s) where failure is concentrated, strongest first."
