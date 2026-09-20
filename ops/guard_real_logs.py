"""Guard replay against real robot logs with ground-truth failures. GPU box only.

Every guard number published so far comes from a simulator. This replays the same guard
against recorded episodes from physical robots where the dataset itself says which episodes
failed, so the scorers meet real sensor noise, real teleoperation jitter and real failures.

What this is not: a closed-loop test. The guard reads a recorded action stream rather than
gating a live policy, and in some of these corpora the actor is a human teleoperator rather
than a policy. It measures whether the scorers separate real good episodes from real bad
ones; it does not measure what would have happened had the guard been allowed to intervene.

Sources are discovered under <data_root>; each is used only if it carries both classes.
Usage: python ops/guard_real_logs.py <data_root> <out_dir> [--max-episodes N] [--chunk K]
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from robotruth.guard import GuardMetrics, OfflineReplay, calibrate_guard
from robotruth.schema.lerobot_ingest import LeRobotDataset
from robotruth.stats.intervals import wilson


def pseudo_chunks(actions: np.ndarray, k: int) -> np.ndarray:
    """[T, D] -> [T, k, D] sliding windows ending at each step, edge-padded at the start."""
    t = actions.shape[0]
    if t < 2:
        return actions[:, None, :]
    return np.stack([actions[max(0, i - k + 1): i + 1] if i + 1 >= k else
                     np.pad(actions[: i + 1], ((k - i - 1, 0), (0, 0)), mode="edge")
                     for i in range(t)])


def label_from_path(path: Path) -> bool | None:
    name = str(path).lower()
    if "anomaly" in name:
        return False
    if "expert" in name:
        return True
    return None


def collect(data_root: Path, max_episodes: int, chunk: int):
    """Yield (source_name, fps, episodes) where each episode has actions, timestamps, success."""
    sources: dict[str, dict] = {}
    for info in sorted(data_root.rglob("meta/info.json")):
        ds_dir = info.parent.parent
        top = ds_dir
        while top.parent != data_root and top.parent != top:
            top = top.parent
        source = top.name
        path_label = label_from_path(ds_dir)
        try:
            ds = LeRobotDataset(ds_dir)
            eps = list(ds.episodes(max_episodes=max_episodes))
        except Exception:  # noqa: BLE001
            continue
        bucket = sources.setdefault(source, {"fps": ds.fps or 30.0, "robot": ds.robot, "episodes": []})
        for ep in eps:
            success = ep.success if path_label is None else path_label
            if success is None or ep.actions is None or ep.actions.shape[0] < chunk + 5:
                continue
            acts = np.asarray(ep.actions, dtype=np.float64)
            # Per-step features are [measured state, commanded action], matching the live
            # experiments. Without the state half there is no Mahalanobis head, and a constant
            # command offset leaves the action stream's own statistics almost untouched.
            feats = None
            if ep.states is not None:
                st = np.asarray(ep.states, dtype=np.float64)
                if st.shape[0] == acts.shape[0]:
                    feats = np.concatenate([st, acts], axis=1)
            bucket["episodes"].append({
                "features": feats,
                "actions": pseudo_chunks(acts, chunk),
                "timestamps": np.asarray(ep.timestamps, dtype=np.float64),
                "success": bool(success), "id": f"{ds_dir.name}#{ep.index}"})
    return sources


def evaluate_source(name: str, data: dict, seed: int = 0) -> tuple[str, dict]:
    eps = data["episodes"]
    fps = data["fps"]
    succ = [e for e in eps if e["success"]]
    fail = [e for e in eps if not e["success"]]
    if len(succ) < 30 or len(fail) < 10:
        return (f"Skipped: needs at least 30 successful and 10 failed episodes, "
                f"has {len(succ)} and {len(fail)}."), {"skipped": True, "n_success": len(succ), "n_failure": len(fail)}

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(succ))
    n_cal = int(0.7 * len(succ))
    cal = [succ[i] for i in order[:n_cal]]
    held = [succ[i] for i in order[n_cal:]]
    use_feats = all(e.get("features") is not None for e in eps)

    def strip(e):
        d = {"actions": e["actions"], "timestamps": e["timestamps"][: e["actions"].shape[0]]}
        if use_feats:
            d["features"] = e["features"]
        return d

    guard, report = calibrate_guard([strip(e) for e in cal], alpha=0.05, method="max", dt=1.0 / fps, stride=1)
    replay = OfflineReplay(guard)
    for e in held:
        replay.run(strip(e), truth=True, episode_id=e["id"])
    for e in fail:
        replay.run(strip(e), truth=False, episode_id=e["id"])
    m = GuardMetrics.from_episodes(guard.episodes)

    fa = sum(1 for r in guard.episodes if r.truth is True and r.first_alert_t is not None)
    hits = sum(1 for r in guard.episodes if r.truth is False and r.first_alert_t is not None)
    leads = [r.lead_time_s for r in guard.episodes if r.truth is False and r.first_alert_t is not None
             and r.lead_time_s is not None]
    stall = report.get("stall_head", {})
    md = "\n".join([
        f"Robot `{data['robot']}`, {fps:g} Hz. {len(succ)} successful and {len(fail)} failed real episodes. "
        f"Calibrated on {len(cal)} successes, replayed {len(held)} held-out successes and {len(fail)} failures. "
        f"Inputs: {'state features and action chunks' if use_feats else 'action chunks only'}.",
        "",
        f"- Detection on real failures: {hits}/{len(fail)} = {wilson(hits, len(fail))}",
        f"- False alarms on held-out successes: {fa}/{len(held)} = {wilson(fa, len(held))} (bound 0.05)",
        f"- Median warning before the episode ended: {f'{np.median(leads):.1f} s' if leads else 'n/a'}",
        f"- False alarms per hour of robot time: {m.false_alarms_per_hour}",
        f"- Stall head: alpha {stall.get('alpha')}, saturated {stall.get('saturated')}",
    ])
    return md, {"n_success": len(succ), "n_failure": len(fail), "n_calibration": len(cal),
                "n_held_out": len(held), "detected": hits, "false_alarms": fa,
                "detection_interval": str(wilson(hits, len(fail))),
                "false_alarm_interval": str(wilson(fa, len(held))),
                "median_lead_s": float(np.median(leads)) if leads else None,
                "false_alarms_per_hour": str(m.false_alarms_per_hour),
                "stall_head_saturated": bool(stall.get("saturated", False))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("out")
    ap.add_argument("--max-episodes", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=10)
    args = ap.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sources = collect(root, args.max_episodes, args.chunk)
    print(f"collected {len(sources)} sources in {time.time()-t0:.0f}s", flush=True)

    doc = ["# Guard on real robot logs", "",
           "The guard replayed against recorded episodes from physical robots, where the dataset "
           "itself says which episodes failed. Action chunks are sliding windows over the recorded "
           "action stream, standing in for a policy's emitted chunks.", "",
           "Two honest caveats. This is replay, not closed-loop control: the guard watches a "
           "recording and never intervenes, so nothing here says what would have happened if it had. "
           "And in the BotFails corpus the actor is a human teleoperator, so what the scorers "
           "separate is good from bad robot motion, not good from bad policy behaviour.", ""]
    index = {}
    for name, data in sorted(sources.items()):
        print(f"-- {name}: {len(data['episodes'])} labelled episodes", flush=True)
        doc.append(f"## {name}")
        try:
            md, js = evaluate_source(name, data)
        except Exception as e:  # noqa: BLE001
            md, js = f"FAILED: {e}", {"error": str(e)}
            print(traceback.format_exc()[-800:], flush=True)
        doc += ["", md, ""]
        index[name] = js
        print(f"   {js}", flush=True)

    doc += [f"Total wall time {time.time()-t0:.0f} s.", "", "REAL_LOGS_COMPLETE"]
    (out / "guard_real_logs.md").write_text("\n".join(doc), encoding="utf-8")
    (out / "guard_real_logs.json").write_text(json.dumps(index, indent=1, default=str), encoding="utf-8")
    print("REAL_LOGS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
