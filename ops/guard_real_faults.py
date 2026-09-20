"""Does the guard's real capability transfer to real robots? Injected faults on real logs.

The replay experiment showed the guard is near chance at spotting real task failures: the
robot moves normally and simply fails the task, which the action stream does not reveal.
What the guard actually detects is an execution fault, where the robot stops behaving like
itself: frozen actuation, a miscalibrated offset, erratic control. That claim has only ever
been measured in a simulator.

This measures it on real robot logs. Successful episodes from physical robots are held out,
the same three faults are injected into the recorded action stream halfway through, and the
guard replays both the clean and the faulted versions. Detection here means the scorers
carry over to real sensor noise and real teleoperation jitter; the clean half of the same
episodes gives the false-alarm rate on the identical data.

This is still replay, not closed-loop control, and an injected fault is not a naturally
occurring one. It tests transfer of the scorers, not of the intervention.

Usage: python ops/guard_real_faults.py <data_root> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard_real_logs import collect  # noqa: E402

from robotruth.guard import OfflineReplay, calibrate_guard  # noqa: E402
from robotruth.stats.intervals import wilson  # noqa: E402


def inject(chunks: np.ndarray, fault: str, onset: int, rng, magnitude: float = 1.0) -> np.ndarray:
    """Apply an execution fault to a [T, k, D] chunk stream from `onset` onward.

    `magnitude` is in units of each action dimension's own standard deviation before the
    onset, so the same number means the same severity on a 7-axis Franka and an SO-100. A
    first pass scaled the offset by mean absolute action value instead, which is not
    comparable across robots and made a missed detection impossible to interpret.
    """
    out = np.array(chunks, dtype=np.float64, copy=True)
    if fault == "freeze":
        out[onset:] = out[onset]
        return out
    sd = out[:onset].reshape(-1, out.shape[-1]).std(axis=0) + 1e-9
    if fault == "offset":
        out[onset:] = out[onset:] + magnitude * sd
    elif fault == "noise":
        out[onset:] = out[onset:] + rng.normal(0.0, 1.0, size=out[onset:].shape) * (magnitude * sd)
    else:
        raise ValueError(fault)
    return out


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sources = collect(root, max_episodes=400, chunk=10)

    doc = ["# Guard on real robot logs, with execution faults injected", "",
           "Real successful episodes from physical robots, with a freeze, a constant offset or "
           "added noise applied to the recorded action stream from the midpoint onward. The "
           "unfaulted copies of the same episodes give the false-alarm rate on identical data, "
           "so detection and false alarms are measured on one population.", "",
           "This tests whether the scorers transfer to real robot data. It does not claim the "
           "faults are naturally occurring, and it is replay rather than closed-loop control.", ""]
    index = {}
    for name, data in sorted(sources.items()):
        eps = [e for e in data["episodes"] if e["success"]]
        if len(eps) < 60:
            continue
        fps = data["fps"]
        rng = np.random.default_rng(5)
        order = rng.permutation(len(eps))
        n_cal = int(0.6 * len(eps))
        cal = [eps[i] for i in order[:n_cal]]
        held = [eps[i] for i in order[n_cal:]]
        use_feats = all(e.get("features") is not None for e in eps)
        n_act = eps[0]["actions"].shape[-1]

        def strip(a, e, feats=None):
            d = {"actions": a, "timestamps": e["timestamps"][: a.shape[0]]}
            if use_feats:
                d["features"] = e["features"] if feats is None else feats
            return d

        def fault_features(e, fault, onset, rng, magnitude=1.0):
            """Apply the same command-side fault to the action half of [state, action].

            The recorded state cannot respond to a command it never received, so replay
            understates what the state half would show on a real robot.
            """
            if not use_feats:
                return None
            f = np.array(e["features"], dtype=np.float64, copy=True)
            acts = f[:, -n_act:][:, None, :]
            f[:, -n_act:] = inject(acts, fault, onset, rng, magnitude)[:, 0, :]
            return f

        guard, _ = calibrate_guard([strip(e["actions"], e) for e in cal], alpha=0.05,
                                   method="max", dt=1.0 / fps, stride=1)
        replay = OfflineReplay(guard)

        fa = 0
        for e in held:
            rec = replay.run(strip(e["actions"], e), truth=True, episode_id=f"clean{e['id']}").episode
            fa += rec.first_alert_t is not None

        cases = [("freeze", 1.0)]
        cases += [(f, m) for f in ("offset", "noise") for m in (0.25, 0.5, 1.0, 2.0, 4.0)]
        rows = {}
        for fault, mag in cases:
            hits, lats = 0, []
            for e in held:
                arr = np.asarray(e["actions"], dtype=np.float64)
                onset = arr.shape[0] // 2
                faulted = inject(arr, fault, onset, rng, mag)
                rec = replay.run(strip(faulted, e, fault_features(e, fault, onset, rng, mag)),
                                 truth=False, episode_id=f"{fault}{mag}{e['id']}").episode
                onset_s = onset / fps
                if rec.first_alert_t is not None and rec.first_alert_t >= onset_s - 1.0:
                    hits += 1
                    lats.append(rec.first_alert_t - onset_s)
            key = fault if fault == "freeze" else f"{fault} {mag:g} sd"
            rows[key] = {"detected": hits, "n": len(held),
                         "interval": str(wilson(hits, len(held))),
                         "median_latency_s": float(np.median(lats)) if lats else None}

        doc += [f"## {name}", "",
                f"Robot `{data['robot']}`, {fps:g} Hz. Calibrated on {len(cal)} real successful "
                f"episodes, evaluated on {len(held)} held out. Inputs: "
                f"{'state features and action chunks' if use_feats else 'action chunks only'}.", "",
                f"False alarms on the clean held-out episodes: {fa}/{len(held)} = "
                f"{wilson(fa, len(held))} (bound 0.05).", "",
                "| injected fault | detected | rate | median latency after onset |", "|---|---|---|---|"]
        for fault, r in rows.items():
            lat = "n/a" if r["median_latency_s"] is None else f"{r['median_latency_s']:.2f} s"
            doc.append(f"| {fault} | {r['detected']}/{r['n']} | {r['interval']} | {lat} |")
        doc.append("")
        index[name] = {"n_calibration": len(cal), "n_held_out": len(held), "false_alarms": fa,
                       "false_alarm_interval": str(wilson(fa, len(held))), "faults": rows}
        print(f"{name}: fa {fa}/{len(held)}, " +
              ", ".join(f"{k} {v['detected']}/{v['n']}" for k, v in rows.items()), flush=True)

    doc += ["", "REAL_FAULTS_COMPLETE"]
    (out / "guard_real_faults.md").write_text("\n".join(doc), encoding="utf-8")
    (out / "guard_real_faults.json").write_text(json.dumps(index, indent=1, default=str), encoding="utf-8")
    print("REAL_FAULTS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
