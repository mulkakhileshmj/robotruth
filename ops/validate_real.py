"""Validate every robotruth module on real LeRobot datasets. Runs on the GPU box.

Usage:
    python ops/validate_real.py <datasets_root> <out_dir> [--max-episodes N] [--chunk K]

<datasets_root> holds one LeRobot dataset directory per subfolder (each with meta/info.json).
For each dataset the script:
  1. ingests episodes into EpisodeRecords, writes episodes.jsonl and fleet metrics;
  2. computes a per-episode unit fingerprint from action (commanded) vs observation.state
     (measured) and reports per-joint lag, backlash, RMSE distributions; across datasets of the
     same robot type this is a real unit-to-unit comparison;
  3. if success labels exist for both classes: fits the judge (action-stream features only,
     no VLM), calibrates with abstain on a held-out split, evaluates on a third split;
  4. calibrates the guard on successful episodes (pseudo-chunks from the action stream) and
     replays held-out successes and failures, reporting detection and false alarms per hour.
Everything is written to <out_dir>/<dataset>/ plus <out_dir>/VALIDATION.md.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from robotruth.fingerprint import unit_fingerprint
from robotruth.schema import EpisodeLog, fleet_metrics
from robotruth.schema.lerobot_ingest import LeRobotDataset, LeRobotEpisode


def _split3(labels: list[str], seed: int = 0):
    """Stratified 40/30/30 split over episode indices."""
    rng = np.random.default_rng(seed)
    a, b, c = [], [], []
    for cls in sorted(set(labels)):
        idx = rng.permutation([i for i, l in enumerate(labels) if l == cls])
        na, nb = int(round(0.4 * len(idx))), int(round(0.3 * len(idx)))
        a += list(idx[:na]); b += list(idx[na:na + nb]); c += list(idx[na + nb:])
    return np.array(a), np.array(b), np.array(c)


def _pseudo_chunks(actions: np.ndarray, k: int) -> np.ndarray:
    """[T, D] -> [T-k+1, k, D] sliding windows, standing in for policy action chunks."""
    T = actions.shape[0]
    if T < k + 2:
        return actions[:, None, :]
    return np.stack([actions[t:t + k] for t in range(T - k + 1)])


def fingerprint_section(eps: list[LeRobotEpisode], name: str) -> tuple[str, dict]:
    stats: dict[str, dict[str, list[float]]] = {}
    n_used = 0
    for ep in eps:
        arr = LeRobotDataset.excitation_arrays(ep)
        if arr is None:
            continue
        t, cmd, meas = arr
        try:
            fp = unit_fingerprint(t, cmd, meas, name)
        except Exception:
            continue
        n_used += 1
        for j, js in fp.joints.items():
            d = stats.setdefault(j, {"lag_ms": [], "backlash": [], "rmse": [], "offset": [], "gain": []})
            d["lag_ms"].append(1000 * js.lag_s); d["backlash"].append(js.backlash); d["rmse"].append(js.rmse)
            d["offset"].append(js.steady_error); d["gain"].append(js.gain)
    if not stats:
        return "No episodes with matching action and state widths; unit fingerprint skipped.", {}
    lines = [f"Per-episode unit fingerprints from {n_used} episodes (action = commanded, observation.state = measured). Median [IQR].", "",
             "| joint | lag ms | backlash | rmse | offset | gain |", "|---|---|---|---|---|---|"]
    summary = {}
    for j, d in stats.items():
        row = {}
        cells = []
        for k in ("lag_ms", "backlash", "rmse", "offset", "gain"):
            v = np.asarray(d[k])
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            row[k] = {"median": float(med), "q1": float(q1), "q3": float(q3)}
            cells.append(f"{med:.4g} [{q1:.3g}, {q3:.3g}]")
        summary[j] = row
        lines.append(f"| {j} | " + " | ".join(cells) + " |")
    return "\n".join(lines), {"n_episodes": n_used, "joints": summary}


def judge_section(eps: list[LeRobotEpisode], out: Path) -> tuple[str, dict]:
    from robotruth.judge import EpisodeSignals, HybridJudge, evaluate, fit_calibrator, fit_fusion
    labelled = [e for e in eps if e.success is not None]
    n_pos = sum(bool(e.success) for e in labelled)
    n_neg = len(labelled) - n_pos
    if n_pos < 6 or n_neg < 6:
        return f"Judge skipped: need at least 6 successes and 6 failures with labels (have {n_pos} / {n_neg}).", {"skipped": True, "n_success": n_pos, "n_failure": n_neg}

    def sig(e: LeRobotEpisode) -> EpisodeSignals:
        grip = e.actions[:, -1] if e.actions.shape[1] > 1 else None
        return EpisodeSignals(actions=e.actions, timestamps=e.timestamps, instruction=e.task, task=e.task,
                              episode_id=str(e.index), states=e.states, gripper=grip)

    pairs = [(sig(e), "success" if e.success else "failure") for e in labelled]
    a, b, c = _split3([l for _, l in pairs])
    fusion = fit_fusion(None, [pairs[i] for i in a])
    judge = HybridJudge(None, fusion=fusion)
    cal_labels = [pairs[i][1] for i in b]
    calibrated = min(cal_labels.count("success"), cal_labels.count("failure")) >= 5
    if calibrated:
        fit_calibrator(judge, [pairs[i] for i in b], target_error=0.10)
    else:
        c = np.concatenate([b, c])
    metrics = evaluate(judge, [pairs[i] for i in c])
    md = metrics.to_markdown("Judge on real episodes (action-stream features only, no VLM)")
    note = ("Conformal calibration with abstain fitted on a held-out split (target selective error 0.10)." if calibrated
            else "Too few labelled episodes per class for conformal calibration; judge evaluated uncalibrated at p = 0.5, no abstain.")
    md = md + "\n\n" + note
    (out / "judge_metrics.json").write_text(json.dumps(metrics.to_dict() if hasattr(metrics, "to_dict") else {}, indent=2, default=str))
    return md, {"n_success": n_pos, "n_failure": n_neg, "n_train": len(a), "n_cal": len(b), "n_eval": len(c),
                "balanced_accuracy": metrics.balanced_accuracy, "abstain_rate": getattr(metrics.abstain_rate, "estimate", None),
                "false_alarms_per_hour": metrics.false_alarms_per_hour}


def guard_section(eps: list[LeRobotEpisode], out: Path, fps: float, k: int) -> tuple[str, dict]:
    from robotruth.guard import GuardMetrics, OfflineReplay, calibrate_guard
    succ = [e for e in eps if e.success is True and e.actions.shape[0] > k + 5]
    fail = [e for e in eps if e.success is False and e.actions.shape[0] > k + 5]
    unl = [e for e in eps if e.success is None and e.actions.shape[0] > k + 5]
    nominal_pool = succ if len(succ) >= 6 else unl
    if len(nominal_pool) < 6:
        return "Guard skipped: fewer than 6 usable episodes.", {"skipped": True}
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(nominal_pool))
    n_cal = max(4, int(0.6 * len(nominal_pool)))
    cal_eps = [nominal_pool[i] for i in idx[:n_cal]]
    held_nom = [nominal_pool[i] for i in idx[n_cal:]]
    to_dict = lambda e: {"actions": _pseudo_chunks(e.actions, k), "timestamps": e.timestamps[: _pseudo_chunks(e.actions, k).shape[0]]}
    guard, report = calibrate_guard([to_dict(e) for e in cal_eps], alpha=0.05, method="max", dt=1.0 / (fps or 30.0), stride=1)
    replay = OfflineReplay(guard)
    for e in held_nom:
        replay.run(to_dict(e), truth=True if e.success else None, episode_id=f"nom{e.index}")
    for e in fail:
        replay.run(to_dict(e), truth=False, episode_id=f"fail{e.index}")
    m = GuardMetrics.from_episodes(guard.episodes)
    md = m.to_markdown("Guard on real episodes (chunk consistency and action stats from pseudo-chunks, conformal max threshold)")
    md += f"\n\nCalibrated on {len(cal_eps)} nominal episodes, replayed {len(held_nom)} held-out nominal and {len(fail)} failures. Nominal pool = {'successes' if nominal_pool is succ else 'unlabelled episodes'}."
    (out / "guard_report.json").write_text(json.dumps({"calibration": report, "metrics": m.to_dict() if hasattr(m, 'to_dict') else {}}, indent=2, default=str))
    return md, {"n_cal": len(cal_eps), "n_held_nominal": len(held_nom), "n_failures": len(fail)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--max-episodes", type=int, default=400); ap.add_argument("--chunk", type=int, default=10)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = ["# robotruth validation on real public datasets", "", f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} on the GPU box.", ""]
    index = {}
    for ds_dir in sorted(p for p in root.iterdir() if (p / "meta" / "info.json").exists()):
        t0 = time.time()
        name = ds_dir.name
        od = out / name
        od.mkdir(parents=True, exist_ok=True)
        doc.append(f"## {name}")
        try:
            ds = LeRobotDataset(ds_dir)
            eps = list(ds.episodes(max_episodes=args.max_episodes))
            recs = [e.to_record(f"dataset:{ds.repo_id}", ds.repo_id, ds.robot) for e in eps]
            log = EpisodeLog(od / "episodes.jsonl")
            if log.path.exists():
                log.path.unlink()
            log.extend(recs)
            fm = fleet_metrics(recs)
            doc += ["", f"repo `{ds.repo_id}`, robot `{ds.robot}`, {ds.fps:.0f} fps, action dim {ds.action_dim()}, state dim {ds.state_dim()}, cameras {ds.camera_keys()}, {len(eps)} episodes ingested.", "",
                    fm.to_markdown("Fleet metrics"), ""]
            fp_md, fp_json = fingerprint_section(eps, name)
            doc += ["### Unit fingerprint", "", fp_md, ""]
            j_md, j_json = judge_section(eps, od)
            doc += ["### Judge", "", j_md, ""]
            g_md, g_json = guard_section(eps, od, ds.fps, args.chunk)
            doc += ["### Guard", "", g_md, ""]
            index[name] = {"repo": ds.repo_id, "robot": ds.robot, "fps": ds.fps, "episodes": len(eps),
                           "success_rate": fm.success_rate.estimate if fm.success_rate else None, "interventions": fm.interventions,
                           "fingerprint": fp_json, "judge": j_json, "guard": g_json, "seconds": round(time.time() - t0, 1)}
            print(f"{name}: {len(eps)} episodes in {time.time()-t0:.0f}s")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            doc += ["", f"FAILED: {e}", "", "```", tb[-2000:], "```", ""]
            index[name] = {"error": str(e)}
            print(f"{name}: FAILED {e}")
    (out / "VALIDATION.md").write_text("\n".join(doc), encoding="utf-8")
    (out / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
