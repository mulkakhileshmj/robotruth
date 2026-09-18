"""Convert cons909/V1-LeRobot-SO101-Eval-Videos trial logs into robotruth inputs.

Each camera-config folder holds trial_log.csv with a pre-registered outcome taxonomy
(success, timeout_no_touch, ...). We produce:
  results.csv        robotruth results table; policy = <config>@<checkpoint_step>
  claims.csv         successes and trials per policy for `robotruth stats audit`

Usage: python ops/so101_eval_convert.py <dataset_dir> <out_dir>
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for log in sorted(root.rglob("trial_log.csv")):
        if ".cache" in log.parts:
            continue
        with log.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cfg = r.get("model_config") or log.parent.name
                policy = f"{cfg}@{r.get('checkpoint_step', '')}"
                outcome = (r.get("outcome") or "").strip().lower()
                success = int(outcome == "success")
                dur = r.get("time_to_completion_sec") or ""
                rows.append({
                    "episode_id": f"{cfg}-{r.get('trial_id')}", "policy": policy, "task": "so101_eval",
                    "success": success, "time_to_success": dur if success else "", "timeout": 180,
                    "pair_id": "", "unit_id": "so101_a", "session_id": cfg, "score": success,
                    "outcome_class": outcome, "safety_clamps": r.get("safety_clamp_events", ""),
                    "latency_ms": r.get("avg_loop_latency_ms", ""),
                })
    if not rows:
        print("no trial logs found"); return 1
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    agg: dict[str, Counter] = defaultdict(Counter)
    outcomes: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        agg[r["policy"]]["trials"] += 1
        agg[r["policy"]]["successes"] += r["success"]
        outcomes[r["policy"]][r["outcome_class"]] += 1
    with (out / "claims.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["policy", "task", "successes", "trials"])
        for p, c in sorted(agg.items()):
            w.writerow([p, "so101_eval", c["successes"], c["trials"]])
    lines = ["# SO101 eval outcome classes per policy", ""]
    for p, c in sorted(outcomes.items()):
        lines.append(f"- **{p}**: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    (out / "outcomes.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(rows)} trials across {len(agg)} policies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
