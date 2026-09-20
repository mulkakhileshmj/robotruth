"""Why does the guard miss real failures? Per-head signal analysis on real robot logs.

The replay experiment found detection collapsing from 30/30 on injected simulator faults to
5/145 on real BotFails failures and 0/58 on DROID. That leaves one question worth more than
any threshold tweak: is there signal in these scores at all?

This measures, per scorer head, the AUROC of the per-episode maximum score between real
successes and real failures. AUROC is threshold-free, so it separates two very different
diagnoses:

  AUROC near 0.5  the head cannot see real failures. No threshold rescues it, and the fix
                  is a scorer that watches something else.
  AUROC near 1.0  the head sees them and the conformal threshold is simply too high, which
                  is a calibration problem and is fixable.

It also reports chunk-consistency separately, because sliding-window pseudo-chunks overlap
by construction, so that head may be structurally meaningless on replayed logs rather than
merely weak.

Usage: python ops/guard_real_diagnose.py <data_root> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard_real_logs import collect  # noqa: E402

from robotruth.guard import calibrate_guard  # noqa: E402


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a failure > score of a success), ties counted half. 0.5 means no signal."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sources = collect(root, max_episodes=400, chunk=10)

    doc = ["# Why the guard misses real failures", "",
           "Per-head AUROC of the per-episode maximum score, real successes against real "
           "failures. AUROC is threshold-free: 0.5 means the head cannot see these failures at "
           "all and no threshold would help, while a high value means the signal is there and "
           "the threshold is what is wrong.", ""]
    index = {}
    for name, data in sorted(sources.items()):
        eps = data["episodes"]
        succ = [e for e in eps if e["success"]]
        fail = [e for e in eps if not e["success"]]
        if len(succ) < 30 or len(fail) < 10:
            continue
        rng = np.random.default_rng(0)
        order = rng.permutation(len(succ))
        n_cal = int(0.7 * len(succ))
        cal = [succ[i] for i in order[:n_cal]]
        held = [succ[i] for i in order[n_cal:]]
        strip = lambda e: {"actions": e["actions"], "timestamps": e["timestamps"][: e["actions"].shape[0]]}
        guard, _ = calibrate_guard([strip(e) for e in cal], alpha=0.05, method="max",
                                   dt=1.0 / data["fps"], stride=1)

        heads = guard.scorer.heads if hasattr(guard.scorer, "heads") else {"main": (guard.scorer, None)}
        rows = []
        for head_name, (scorer, _cal) in heads.items():
            sub = dict(scorer.components) if hasattr(scorer, "components") else {head_name: scorer}
            for comp_name, comp in sub.items():
                def ep_max(e):
                    scorer.reset()
                    if hasattr(comp, "score_episode"):
                        v = comp.score_episode(strip(e))
                    else:
                        return np.nan
                    v = np.asarray(v, dtype=np.float64)
                    v = v[np.isfinite(v)]
                    return float(v.max()) if v.size else np.nan
                f = np.array([ep_max(e) for e in fail])
                s = np.array([ep_max(e) for e in held])
                f, s = f[np.isfinite(f)], s[np.isfinite(s)]
                a = auroc(f, s)
                rows.append({"head": head_name, "component": comp_name, "auroc": a,
                             "median_failure_score": float(np.median(f)) if f.size else None,
                             "median_success_score": float(np.median(s)) if s.size else None})
        doc += [f"## {name}", "",
                f"{len(succ)} successes, {len(fail)} failures, robot `{data['robot']}`.", "",
                "| head | component | AUROC | median score on failures | median score on held-out successes |",
                "|---|---|---|---|---|"]
        for r in rows:
            doc.append(f"| {r['head']} | {r['component']} | {r['auroc']:.3f} | "
                       f"{r['median_failure_score']:.3f} | {r['median_success_score']:.3f} |")
        doc.append("")
        index[name] = rows
        print(f"{name}: " + ", ".join(f"{r['component']}={r['auroc']:.3f}" for r in rows), flush=True)

    doc += ["", "DIAGNOSE_COMPLETE"]
    (out / "guard_real_diagnose.md").write_text("\n".join(doc), encoding="utf-8")
    (out / "guard_real_diagnose.json").write_text(json.dumps(index, indent=1, default=str), encoding="utf-8")
    print("DIAGNOSE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
