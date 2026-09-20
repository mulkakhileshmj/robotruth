"""Why do several detectors score below chance on real failures?

An AUROC of 0.31 is not "no signal". It means the score is anti-correlated with failure:
episodes that failed look calmer than episodes that succeeded. If that is what is happening,
a monitor deployed on this data would preferentially alarm on successful episodes, which is
worse than useless and worth understanding rather than reporting as a curiosity.

The obvious candidate is that a policy that fails often does less. It stalls, gives up, or
never reaches the contact-rich part of the task, so the action stream is shorter and quieter
exactly when the task went wrong. This measures episode length and per-step motion for
successes and failures separately, and reports the AUROC of raw motion so the direction of
the effect is explicit.

Usage: python ops/why_below_chance.py <data_root> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_comparison import auroc  # noqa: E402
from guard_real_logs import collect  # noqa: E402


def stats_for(ep: dict) -> dict:
    ch = np.asarray(ep["actions"], dtype=np.float64)
    means = ch.mean(axis=1)
    step = np.linalg.norm(np.diff(means, axis=0), axis=1) if means.shape[0] > 1 else np.zeros(1)
    return {"steps": float(ch.shape[0]),
            "motion_per_step": float(step.mean()),
            "total_motion": float(step.sum()),
            "action_spread": float(ch.var(axis=1).mean())}


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sources = collect(root, 400, 10)

    doc = ["# Why several detectors land below chance on real failures", "",
           "An AUROC below 0.5 means the score runs the wrong way: real failures score lower, "
           "so a monitor tuned on this data would alarm on successes first. The table gives the "
           "median of each raw quantity for successful and failed episodes, and the AUROC of "
           "that quantity on its own, so the direction of the effect is visible without any "
           "detector in the way.", ""]
    index = {}
    for name, data in sorted(sources.items()):
        eps = data["episodes"]
        succ = [e for e in eps if e["success"]]
        fail = [e for e in eps if not e["success"]]
        if len(succ) < 30 or len(fail) < 20:
            continue
        s = [stats_for(e) for e in succ]
        f = [stats_for(e) for e in fail]
        doc += [f"## {name}", "",
                f"Robot `{data['robot']}`, {len(succ)} successes, {len(fail)} failures.", "",
                "| quantity | median, successes | median, failures | AUROC of this quantity alone |",
                "|---|---|---|---|"]
        rows = {}
        for k in ("steps", "motion_per_step", "total_motion", "action_spread"):
            sv = np.array([x[k] for x in s])
            fv = np.array([x[k] for x in f])
            a = auroc(fv, sv)
            rows[k] = {"median_success": float(np.median(sv)), "median_failure": float(np.median(fv)),
                       "auroc": float(a)}
            doc.append(f"| {k.replace('_',' ')} | {np.median(sv):.4g} | {np.median(fv):.4g} | {a:.3f} |")
        doc.append("")
        index[name] = rows
        print(f"{name}: " + ", ".join(f"{k}={v['auroc']:.2f}" for k, v in rows.items()), flush=True)

    doc += ["", "WHY_BELOW_CHANCE_COMPLETE"]
    (out / "why_below_chance.md").write_text("\n".join(doc), encoding="utf-8")
    (out / "why_below_chance.json").write_text(json.dumps(index, indent=1, default=str),
                                               encoding="utf-8")
    print("WHY_BELOW_CHANCE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
