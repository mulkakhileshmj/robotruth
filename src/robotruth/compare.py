"""Compare two policies from a results table and build the report.

This is the function behind `robotruth stats compare`. It applies every rule in module 2:
intervals on every rate, paired analysis when pair ids exist, an anytime-valid sequential
verdict, time-to-success with censoring, per-task breakdown, and an explicit statement of
how many trials would have been needed if the comparison is inconclusive.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from robotruth.report import Report, Section
from robotruth.results import Results
from robotruth.stats.intervals import bootstrap_diff_ci, paired_diff_ci, wilson
from robotruth.stats.planning import required_trials_two_proportions
from robotruth.stats.sequential import sequential_paired_test
from robotruth.stats.timing import logrank_test


def compare_policies(res: Results, a: str, b: str, alpha: float = 0.05, margin: float = 0.0,
                     title: Optional[str] = None) -> Report:
    rep = Report(title or f"robotruth: is {b} better than {a}?")
    ra, rb = res.subset(a), res.subset(b)
    if not ra.rows or not rb.rows:
        raise ValueError(f"no rows for one of the policies: {a}={len(ra.rows)}, {b}={len(rb.rows)}")

    # 1. Rates with intervals, overall and per task.
    rows = []
    for task in [None] + res.tasks():
        sa, sb = ra.subset(task=task), rb.subset(task=task)
        if not sa.rows or not sb.rows:
            continue
        wa = wilson(int(sa.successes().sum()), len(sa.rows), alpha)
        wb = wilson(int(sb.successes().sum()), len(sb.rows), alpha)
        rows.append({
            "task": task or "ALL", f"{a} n": wa.n, f"{a} rate": f"{wa.estimate:.2f} [{wa.lower:.2f}, {wa.upper:.2f}]",
            f"{b} n": wb.n, f"{b} rate": f"{wb.estimate:.2f} [{wb.lower:.2f}, {wb.upper:.2f}]",
            "intervals overlap": "yes" if (wa.lower <= wb.upper and wb.lower <= wa.upper) else "no",
        })
    rep.add(Section("Success rates with Wilson intervals",
                    f"Every rate carries a {100*(1-alpha):.0f}% interval. Overlapping intervals do not prove equality; "
                    "non-overlap is strong evidence of a difference.", rows))

    # 2. Difference estimate: paired if possible.
    paired = res.has_pairs()
    if paired:
        xa, xb, pairs = res.paired(a, b)
        if xa.size >= 2:
            ci = paired_diff_ci(xa, xb, alpha)
            boot = bootstrap_diff_ci(xa, xb, alpha, paired=True)
            n_pairs = xa.size
            body = (f"{n_pairs} matched pairs (same pair_id and task).\n"
                    f"Difference in success (B - A): {ci}\n"
                    f"Bootstrap check: {boot}\n"
                    f"Discordant pairs: A only {int(np.sum((xa==1)&(xb==0)))}, B only {int(np.sum((xa==0)&(xb==1)))}.")
            verdict = "PASS" if ci.lower > margin else ("FAIL" if ci.upper < -margin else "WARN")
            rep.add(Section("Paired difference", body, verdict=verdict))
            # 3. Anytime-valid sequential verdict, in pair order.
            seq = sequential_paired_test(xa, xb, alpha=alpha, margin=margin, stop_early=False)
            first = next((t for t in seq.trace if (t[2] > margin or t[3] < -margin)), None)
            stop_note = (f"An anytime-valid stopping rule would have decided at n={first[0]}."
                         if first else "The sequence never excluded zero; more paired trials are needed.")
            rep.add(Section("Anytime-valid sequential verdict",
                            f"{seq}\n{stop_note}\nThis interval is valid at every sample size simultaneously "
                            "(empirical Bernstein confidence sequence), so peeking after each rollout is allowed.",
                            table=[{"n": t[0], "diff": t[1], "lower": t[2], "upper": t[3]} for t in seq.trace[:: max(1, len(seq.trace)//20)]],
                            verdict={"B_better": "PASS", "A_better": "FAIL"}.get(seq.decision, "WARN")))
        else:
            paired = False
    if not paired:
        boot = bootstrap_diff_ci(ra.scores(), rb.scores(), alpha, paired=False)
        verdict = "PASS" if boot.lower > margin else ("FAIL" if boot.upper < -margin else "WARN")
        rep.add(Section("Unpaired difference",
                        f"No pair_id column found, so trials are treated as independent.\n"
                        f"Difference in mean score (B - A): {boot}\n"
                        "Interleave and pair your trials next time: it needs fewer rollouts for the same power.",
                        verdict=verdict))

    # 4. Time to success with censoring.
    ta, ea = ra.times()
    tb, eb = rb.times()
    if np.isfinite(ta).all() and np.isfinite(tb).all() and (ea.sum() + eb.sum()) > 0:
        lr = logrank_test(ta, ea, tb, eb)
        rep.add(Section("Time to success (censored)",
                        f"{lr}\nFailures are treated as censored at their timeout. A faster policy at equal success "
                        "rate is a better policy; binary success cannot see that.",
                        verdict="INFO"))

    # 5. If inconclusive, say what it would take.
    pa = ra.successes().mean()
    pb = rb.successes().mean()
    if abs(pa - pb) > 1e-9 and 0 < pa < 1 and 0 < pb < 1:
        plan_ind = required_trials_two_proportions(pa, pb, alpha, 0.8, paired=False)
        plan_pair = required_trials_two_proportions(pa, pb, alpha, 0.8, paired=True, rho=0.4)
        rep.add(Section("What it would take to resolve this",
                        f"Observed rates {pa:.2f} vs {pb:.2f}. To detect that gap with 80% power:\n"
                        f"- independent trials: {plan_ind}\n- paired interleaved trials (rho 0.4): {plan_pair}\n"
                        f"You have {len(ra.rows)} and {len(rb.rows)} trials.", verdict="INFO"))

    verdicts = [s.verdict for s in rep.sections if s.verdict in ("PASS", "FAIL", "WARN")]
    rep.verdict = "FAIL" if "FAIL" in verdicts else ("PASS" if verdicts and all(v == "PASS" for v in verdicts) else "WARN")
    return rep
