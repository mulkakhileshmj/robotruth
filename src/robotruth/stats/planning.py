"""How many trials do you need? Ask before you roll."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Plan:
    n_per_arm: int
    p_a: float
    p_b: float
    alpha: float
    power: float
    paired: bool
    rho: float

    def __str__(self) -> str:
        kind = "paired" if self.paired else "independent"
        return (f"{self.n_per_arm} trials per policy ({kind}) to detect {self.p_a:.2f} -> {self.p_b:.2f} "
                f"at alpha={self.alpha}, power={self.power}")


def required_trials_two_proportions(p_a: float, p_b: float, alpha: float = 0.05, power: float = 0.8,
                                    paired: bool = False, rho: float = 0.0) -> Plan:
    """Sample size per arm for a two-sided test of p_b vs p_a.

    Independent arms use the standard two-proportion normal approximation. Paired designs
    (interleaved A/B on matched conditions) reduce variance by the correlation `rho`
    between the two outcomes on the same condition; rho of 0.3 to 0.5 is typical when
    the condition (object pose, lighting) drives much of the outcome.
    """
    if not (0 < p_a < 1 and 0 < p_b < 1):
        raise ValueError("rates must be strictly inside (0, 1)")
    if p_a == p_b:
        raise ValueError("rates must differ")
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(power)
    var_a, var_b = p_a * (1 - p_a), p_b * (1 - p_b)
    var_diff = var_a + var_b
    if paired:
        var_diff -= 2 * rho * np.sqrt(var_a * var_b)
    n = ((z_a + z_b) ** 2) * var_diff / (p_b - p_a) ** 2
    return Plan(int(np.ceil(n)), p_a, p_b, alpha, power, paired, rho)


def detectable_difference(n_per_arm: int, p_a: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute improvement over p_a you can detect with n trials per arm (independent arms)."""
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(power)
    # Conservative: use the larger variance at p=0.5 bound scaled to p_a.
    var = 2 * p_a * (1 - p_a)
    return float((z_a + z_b) * np.sqrt(var / n_per_arm))
