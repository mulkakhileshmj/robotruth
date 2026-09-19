"""Pooled BotFails benchmark: judge and guard against real labelled failures.

The BotFails corpus stores one LeRobot dataset per recording session, and the ground
truth lives in the directory name: expert sessions are successes, anomaly sessions are
failures. validate_real.py sees each session alone (single class, judge skipped), so this
script pools every session, applies the name labels, and runs the same judge and guard
sections over the whole corpus. Output is one index entry merged into the benchmark
report by ops/benchmark_report.py.

Usage: python ops/botfails_bench.py <botfails_root> <val_dir> [--max-episodes 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_real import fingerprint_section, guard_section, judge_section  # noqa: E402

from robotruth.schema import fleet_metrics  # noqa: E402
from robotruth.schema.lerobot_ingest import LeRobotDataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("val")
    ap.add_argument("--max-episodes", type=int, default=300)
    args = ap.parse_args()
    root, val = Path(args.root), Path(args.val)
    out = val / "kantine__BotFails__pooled"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    eps, recs, fps = [], [], 30.0
    n_sets = 0
    for info in sorted(root.rglob("meta/info.json")):
        ds_dir = info.parent.parent
        name = ds_dir.name.lower()
        label = False if "anomaly" in name else (True if "expert" in name else None)
        if label is None:
            continue
        try:
            ds = LeRobotDataset(ds_dir)
        except Exception as e:  # noqa: BLE001
            print(f"skip {ds_dir.name}: {e}")
            continue
        n_sets += 1
        fps = ds.fps or fps
        for ep in ds.episodes(max_episodes=args.max_episodes):
            ep.success = label
            ep.task = ep.task or ds_dir.name
            eps.append(ep)
            recs.append(ep.to_record(f"dataset:{ds.repo_id}", f"BotFails/{ds_dir.name}", ds.robot))
    n_f = sum(not e.success for e in eps)
    print(f"botfails pooled: {n_sets} sessions, {len(eps)} episodes, {n_f} failures, {time.time()-t0:.0f}s")

    fm = fleet_metrics(recs)
    fp_md, fp_json = fingerprint_section(eps, "BotFails pooled")
    j_md, j_json = judge_section(eps, out)
    g_md, g_json = guard_section(eps, out, fps, 10)
    (out / "sections.md").write_text(
        "# BotFails pooled\n\n" + fm.to_markdown("Fleet metrics") + "\n\n## Unit fingerprint\n\n" + fp_md
        + "\n\n## Judge\n\n" + j_md + "\n\n## Guard\n\n" + g_md + "\n", encoding="utf-8")
    entry = {"repo": "kantine/BotFails (pooled sessions)", "robot": "so100 class", "fps": fps,
             "episodes": len(eps),
             "success_rate": fm.success_rate.estimate if fm.success_rate else None,
             "interventions": 0, "fingerprint": fp_json, "judge": j_json, "guard": g_json,
             "seconds": round(time.time() - t0, 1)}
    (val / "extra_index.json").write_text(
        json.dumps({"kantine__BotFails__pooled": entry}, indent=2, default=str), encoding="utf-8")
    print("BOTFAILS_BENCH_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
