"""Module 2: honest statistics for robot policy evaluation.

Every function here exists because the field skips it: 0 of 13 audited real-robot VLA
papers report a confidence interval (PhAIL, arXiv 2605.29710); a 50-trial success rate
carries a 20 to 30 point interval (TRI LBM); resolving close pairs by binary success needs
600 to 1,500 paired rollouts.

Contents:
- intervals: Wilson, Clopper-Pearson, bootstrap intervals for rates and differences.
- sequential: anytime-valid confidence sequences (empirical Bernstein, Waudby-Smith and
  Ramdas) so you can stop the moment the answer is clear and still be right.
- planning: how many trials you need before you start.
- timing: time-to-success comparison with censoring (Kaplan-Meier, log-rank, KS).
- pairwise: Bradley-Terry ranking with task buckets.
- schedule: blinded, interleaved A/B/n rollout schedules.
"""

from robotruth.stats.intervals import (
    bootstrap_diff_ci,
    clopper_pearson,
    paired_diff_ci,
    wilson,
)
from robotruth.stats.planning import required_trials_two_proportions, detectable_difference
from robotruth.stats.sequential import ConfidenceSequence, sequential_paired_test
from robotruth.stats.timing import kaplan_meier, logrank_test, ks_time_to_success
from robotruth.stats.pairwise import bradley_terry
from robotruth.stats.schedule import interleaved_schedule

__all__ = [
    "ConfidenceSequence",
    "bootstrap_diff_ci",
    "bradley_terry",
    "clopper_pearson",
    "detectable_difference",
    "interleaved_schedule",
    "kaplan_meier",
    "ks_time_to_success",
    "logrank_test",
    "paired_diff_ci",
    "required_trials_two_proportions",
    "sequential_paired_test",
    "wilson",
]
