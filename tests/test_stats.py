import numpy as np
import pytest

from robotruth.stats import (
    bootstrap_diff_ci,
    bradley_terry,
    clopper_pearson,
    interleaved_schedule,
    kaplan_meier,
    logrank_test,
    paired_diff_ci,
    required_trials_two_proportions,
    sequential_paired_test,
    wilson,
)
from robotruth.stats.sequential import ConfidenceSequence


def test_wilson_basic_properties():
    iv = wilson(45, 50)
    assert 0.75 < iv.lower < iv.estimate < iv.upper < 1.0
    assert wilson(0, 10).lower == 0.0
    assert wilson(10, 10).upper == 1.0
    # The TRI observation: 50 trials at 90% is a ~16 point wide interval.
    assert 0.12 < iv.width < 0.20


def test_clopper_pearson_contains_wilson_roughly():
    w, c = wilson(30, 40), clopper_pearson(30, 40)
    assert c.lower <= w.lower + 0.02 and c.upper >= w.upper - 0.02


def test_wilson_coverage_by_simulation():
    rng = np.random.default_rng(1)
    p, n, hits, sims = 0.85, 30, 0, 4000
    for _ in range(sims):
        k = rng.binomial(n, p)
        iv = wilson(k, n, 0.05)
        hits += iv.lower <= p <= iv.upper
    assert hits / sims > 0.93


def test_paired_ci_tighter_than_unpaired_when_correlated():
    rng = np.random.default_rng(0)
    n = 200
    latent = rng.uniform(size=n)  # condition difficulty shared by both arms
    a = (rng.uniform(size=n) < 0.7 - 0.4 * latent + 0.2).astype(float)
    b = (rng.uniform(size=n) < 0.8 - 0.4 * latent + 0.2).astype(float)
    ci_p = paired_diff_ci(a, b)
    ci_u = bootstrap_diff_ci(a, b, paired=False)
    assert ci_p.width < ci_u.width


def test_confidence_sequence_is_anytime_valid():
    """Coverage of the running CS must hold simultaneously over all t (not just at the end)."""
    rng = np.random.default_rng(7)
    alpha, sims, horizon, mu = 0.05, 400, 300, 0.6
    misses = 0
    for _ in range(sims):
        cs = ConfidenceSequence(alpha=alpha)
        ok = True
        for x in rng.binomial(1, mu, size=horizon):
            _, lo, hi = cs.update(float(x))
            if not (lo <= mu <= hi):
                ok = False
                break
        misses += not ok
    assert misses / sims <= alpha + 0.02, f"anytime miss rate {misses/sims:.3f} exceeds alpha"


def test_sequential_test_finds_real_difference_and_stops_early():
    rng = np.random.default_rng(3)
    n = 600
    a = rng.binomial(1, 0.70, size=n)
    b = rng.binomial(1, 0.90, size=n)
    r = sequential_paired_test(a, b, alpha=0.05)
    assert r.decision == "B_better"
    assert r.n < n  # stopped before using every trial
    assert r.lower > 0


def test_sequential_test_does_not_false_alarm_often():
    rng = np.random.default_rng(11)
    alarms = 0
    sims = 200
    for _ in range(sims):
        a = rng.binomial(1, 0.8, size=150)
        b = rng.binomial(1, 0.8, size=150)
        r = sequential_paired_test(a, b, alpha=0.05)
        alarms += r.decision in ("B_better", "A_better")
    assert alarms / sims <= 0.08


def test_required_trials_matches_textbook_order_of_magnitude():
    plan = required_trials_two_proportions(0.80, 0.90, 0.05, 0.80)
    assert 180 <= plan.n_per_arm <= 220
    paired = required_trials_two_proportions(0.80, 0.90, 0.05, 0.80, paired=True, rho=0.4)
    assert paired.n_per_arm < plan.n_per_arm
    with pytest.raises(ValueError):
        required_trials_two_proportions(0.8, 0.8)


def test_kaplan_meier_and_logrank():
    ta = np.array([10, 12, 15, 30, 30, 30], float); ea = np.array([1, 1, 1, 0, 0, 0])
    tb = np.array([5, 6, 7, 8, 9, 30], float); eb = np.array([1, 1, 1, 1, 1, 0])
    km_a, km_b = kaplan_meier(ta, ea), kaplan_meier(tb, eb)
    assert km_b.median_time() < km_a.median_time()
    lr = logrank_test(ta, ea, tb, eb)
    assert lr.p_value < 0.05
    same = logrank_test(ta, ea, ta, ea)
    assert same.p_value > 0.9


def test_bradley_terry_recovers_order():
    rng = np.random.default_rng(0)
    true = {"p1": 1.0, "p2": 0.0, "p3": -1.0}
    comps = []
    for _ in range(600):
        i, j = rng.choice(list(true), 2, replace=False)
        p = 1 / (1 + np.exp(-(true[i] - true[j])))
        comps.append((i, j, float(rng.uniform() < p), "task"))
    bt = bradley_terry(comps, n_boot=20)
    order = [n for n, _, _ in bt.ranking()]
    assert order == ["p1", "p2", "p3"]
    assert bt.win_prob[bt.policies.index("p1"), bt.policies.index("p3")] > 0.8


def test_interleaved_schedule_balanced_and_blinded():
    s = interleaved_schedule(["base", "cand"], ["pose1", "pose2", "pose3"], repeats=2, seed=5)
    assert len(s.slots) == 12
    counts = {}
    for slot in s.slots:
        counts[slot.policy] = counts.get(slot.policy, 0) + 1
    assert counts == {"base": 6, "cand": 6}
    pairs = {}
    for slot in s.slots:
        pairs.setdefault(slot.pair_id, set()).add(slot.policy)
    assert all(v == {"base", "cand"} for v in pairs.values())
    assert set(s.key.values()) == {"base", "cand"}
    assert all(r.get("policy") is None for r in s.to_rows())
