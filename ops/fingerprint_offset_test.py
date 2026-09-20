"""Does the fingerprint catch the miscalibration the guard misses? Real logs, injected offsets.

The runtime guard detects stalls and erratic control on real robot logs immediately, but a
constant command offset has to reach roughly two standard deviations before it alarms, and on
some robots it never does. That is not surprising on reflection: a constant offset preserves
motion and chunk agreement, so a per-step anomaly score has little to grip.

Constant offsets are what the unit fingerprint is for. It regresses commanded against
measured joint positions across a whole episode and reports a steady-state error per joint,
which is exactly a miscalibration signal, measured between sessions rather than per step.

This injects the same offsets into the commanded stream of real episodes and asks whether the
fingerprint's own offset estimate moves outside the spread seen across clean episodes. A
detection rule of three nominal standard deviations is used, per joint, alarming if any joint
trips.

Usage: python ops/fingerprint_offset_test.py <data_root> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from robotruth.fingerprint import unit_fingerprint
from robotruth.schema.lerobot_ingest import LeRobotDataset
from robotruth.stats.intervals import wilson

MAGNITUDES = (0.25, 0.5, 1.0, 2.0)


def episode_arrays(root: Path, max_per_source: int = 120):
    """Yield (source, [(t, cmd, meas), ...]) for datasets that expose both streams."""
    sources: dict[str, list] = {}
    for info in sorted(root.rglob("meta/info.json")):
        ds_dir = info.parent.parent
        top = ds_dir
        while top.parent != root and top.parent != top:
            top = top.parent
        try:
            ds = LeRobotDataset(ds_dir)
            eps = list(ds.episodes(max_episodes=max_per_source))
        except Exception:  # noqa: BLE001
            continue
        bucket = sources.setdefault(top.name, [])
        for ep in eps:
            arr = LeRobotDataset.excitation_arrays(ep)
            if arr is None:
                continue
            t, cmd, meas = arr          # cmd and meas are {joint_name: 1-D array}
            if len(t) > 20 and set(cmd) == set(meas):
                bucket.append((t, cmd, meas))
            if len(bucket) >= max_per_source:
                break
    return {k: v for k, v in sources.items() if len(v) >= 40}


def offsets_of(t, cmd, meas, name="unit") -> dict[str, float]:
    fp = unit_fingerprint(t, cmd, meas, name)
    return {j: js.steady_error for j, js in fp.joints.items()}


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sources = episode_arrays(root)

    doc = ["# Constant offsets: the fingerprint catches what the guard cannot", "",
           "The same constant command offsets that the runtime guard needs about two standard "
           "deviations to notice, put through the unit fingerprint instead. The fingerprint "
           "regresses commanded against measured joint positions over a whole episode and "
           "reports a steady-state error per joint. An episode counts as detected when any "
           "joint's offset estimate falls more than three nominal standard deviations from the "
           "clean mean for that joint.", "",
           "This is an offline, between-sessions check, not a per-step one. It is the right "
           "tool for drift and miscalibration; the guard is the right tool for stalls and "
           "erratic control. Neither detects a robot that moves normally and still fails the "
           "task, which is what the vision judge is for.", ""]
    index = {}
    for name, eps in sorted(sources.items()):
        n_cal = int(0.6 * len(eps))
        cal, held = eps[:n_cal], eps[n_cal:]
        clean = [offsets_of(*e) for e in cal]
        joints = sorted({j for c in clean for j in c})
        mu = {j: float(np.mean([c[j] for c in clean if j in c])) for j in joints}
        sd = {j: float(np.std([c[j] for c in clean if j in c]) + 1e-12) for j in joints}

        def alarms(o: dict[str, float]) -> bool:
            return any(abs(o[j] - mu[j]) > 3.0 * sd[j] for j in o if j in mu)

        fa = sum(alarms(offsets_of(*e)) for e in held)
        rows = {}
        for mag in MAGNITUDES:
            hits = 0
            for t, cmd, meas in held:
                shifted = {j: v + mag * (float(np.std(v)) + 1e-9) for j, v in cmd.items()}
                hits += alarms(offsets_of(t, shifted, meas))
            rows[f"{mag:g} sd"] = {"detected": hits, "n": len(held),
                                   "interval": str(wilson(hits, len(held)))}
        doc += [f"## {name}", "",
                f"{len(eps)} episodes with both commanded and measured streams; "
                f"{len(cal)} used for the nominal spread, {len(held)} held out.", "",
                f"False alarms on clean held-out episodes: {fa}/{len(held)} = {wilson(fa, len(held))}.", "",
                "| injected offset | fingerprint detects | rate |", "|---|---|---|"]
        for k, r in rows.items():
            doc.append(f"| {k} | {r['detected']}/{r['n']} | {r['interval']} |")
        doc.append("")
        index[name] = {"n_episodes": len(eps), "false_alarms": fa, "n_held_out": len(held),
                       "offsets": rows}
        print(f"{name}: fa {fa}/{len(held)}, " +
              ", ".join(f"{k} {v['detected']}/{v['n']}" for k, v in rows.items()), flush=True)

    doc += ["", "FINGERPRINT_OFFSET_COMPLETE"]
    (out / "fingerprint_offset.md").write_text("\n".join(doc), encoding="utf-8")
    (out / "fingerprint_offset.json").write_text(json.dumps(index, indent=1, default=str), encoding="utf-8")
    print("FINGERPRINT_OFFSET_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
