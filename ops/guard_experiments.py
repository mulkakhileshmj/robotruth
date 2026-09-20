"""Tier 1 guard experiments: tighten the bound, ramp the faults, change the policy.

Three modes, all writing markdown and json into <out_dir>:

  fa     Reuse a saved nominal calibration pool, then run many fresh nominal episodes to
         measure the false-alarm rate. The 0.1.4 bound rested on 18 held-out successes, so
         its interval reached 0.176: a true rate of 15 percent would have looked identical.
         This mode exists to shrink that interval with episodes, not adjectives.

  ramp   Faults that arrive gradually instead of all at once. Every published detection
         number so far comes from a step change in the executed action, which is the easiest
         possible case. Here each fault ramps in over a chosen number of steps and detection
         and latency are reported per ramp rate.

  combo  The full detection experiment on another policy and task. Runs a smoke test first:
         calibrating a guard on a policy that does not work produces a meaningless nominal
         distribution, so a policy that fails its smoke test is reported and skipped rather
         than silently measured.

Usage:
  python ops/guard_experiments.py fa    <out_dir> <episodes_dir> [n_nominal]
  python ops/guard_experiments.py ramp  <out_dir> <episodes_dir> [n_per_cell]
  python ops/guard_experiments.py combo <out_dir> <combo_name> [n_cal] [n_nom] [n_fault]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ONSET_STEP = 150


def load_saved_pool(ep_dir: Path, prefix: str = "cal", limit: int | None = None) -> list[dict]:
    eps = []
    for p in sorted(ep_dir.glob(f"{prefix}_*.npz")):
        d = np.load(p, allow_pickle=True)
        if not bool(d["success"]):
            continue
        eps.append({"features": d["features"], "actions": d["actions"], "timestamps": d["timestamps"]})
        if limit and len(eps) >= limit:
            break
    return eps


def _guard(pool: list[dict], fps: float, method: str = "max"):
    from robotruth.guard import calibrate_guard
    return calibrate_guard(pool, alpha=0.05, method=method, dt=1 / fps, stride=1)


def _setup(combo_name: str):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from policy_zoo import COMBOS, load_policy, make_env

    combo = COMBOS[combo_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(combo)
    policy = load_policy(combo, device)
    return combo, env, policy, device, torch


def mode_fa(out_dir: Path, ep_dir: Path, n_nominal: int) -> int:
    from policy_zoo import run_episode

    from robotruth.guard import OfflineReplay
    from robotruth.stats.intervals import wilson

    combo, env, policy, device, torch = _setup("act_aloha_transfer")
    pool = load_saved_pool(ep_dir)
    print(f"calibrating on {len(pool)} saved nominal episodes", flush=True)
    guard, report = _guard(pool, combo.fps)
    rng = np.random.default_rng(7)

    t0 = time.time()
    alarms, successes, natural_failures = 0, 0, 0
    per_ep = []
    replay = OfflineReplay(guard)
    for i in range(n_nominal):
        e = run_episode(combo, env, policy, device, torch, fault=None, rng=rng)
        rec = replay.run({"features": e["features"], "actions": e["actions"], "timestamps": e["timestamps"]},
                         truth=bool(e["success"]), episode_id=f"nom{i}").episode
        alarmed = rec.first_alert_t is not None
        if e["success"]:
            successes += 1
            alarms += alarmed
        else:
            natural_failures += 1
        per_ep.append({"i": i, "success": bool(e["success"]), "alarmed": bool(alarmed),
                       "first_alert_t": rec.first_alert_t, "steps": e["steps"]})
        if (i + 1) % 25 == 0:
            print(f"nominal {i+1}/{n_nominal} ({successes} ok, {alarms} alarms, "
                  f"{(time.time()-t0)/(i+1):.0f}s/ep)", flush=True)

    iv = wilson(alarms, successes) if successes else None
    lines = ["# Guard false-alarm rate on a large held-out nominal pool", "",
             f"Policy {combo.repo} live in {combo.env_id}. Calibration pool {len(pool)} saved nominal "
             f"episodes (from the 0.1.4 run). Held-out episodes run fresh here: {n_nominal}, of which "
             f"{successes} succeeded and {natural_failures} failed on their own. A false alarm is an "
             "alert raised during an episode the policy actually completed.", "",
             f"False alarms: {alarms}/{successes} = {iv} (95% Wilson), against a 0.05 bound.", "",
             f"For comparison, 0.1.4 measured 0/18 = {wilson(0, 18)}, an interval wide enough to hide "
             "a 17 percent true rate.", "",
             f"Total wall time {time.time()-t0:.0f} s.", "", "GUARD_FA_COMPLETE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "guard_fa.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "guard_fa.json").write_text(json.dumps(
        {"n_calibration": len(pool), "n_run": n_nominal, "n_success": successes,
         "n_natural_failure": natural_failures, "alarms": alarms,
         "false_alarm_interval": str(iv), "episodes": per_ep,
         "calibration_report": {k: v for k, v in report.items() if k != "calibrator"}},
        indent=1, default=str), encoding="utf-8")
    print("\n".join(lines[-4:]), flush=True)
    return 0


def mode_ramp(out_dir: Path, ep_dir: Path, n_cell: int) -> int:
    from policy_zoo import run_episode

    from robotruth.guard import OfflineReplay
    from robotruth.stats.intervals import wilson

    combo, env, policy, device, torch = _setup("act_aloha_transfer")
    pool = load_saved_pool(ep_dir)
    print(f"calibrating on {len(pool)} saved nominal episodes", flush=True)
    guard, _ = _guard(pool, combo.fps)
    rng = np.random.default_rng(11)
    onset_s = ONSET_STEP / combo.fps

    grid = [0, 25, 50, 100, 200]     # steps to reach full strength: 0, 0.5, 1, 2, 4 seconds at 50 Hz
    rows = []
    t0 = time.time()
    for fault in ("freeze", "offset", "noise"):
        for ramp in grid:
            replay = OfflineReplay(guard)
            broke, hits, lats = 0, 0, []
            for i in range(n_cell):
                e = run_episode(combo, env, policy, device, torch, fault=fault, rng=rng, ramp_steps=ramp)
                if e["success"]:
                    continue                      # the fault did not break the task; nothing to detect
                broke += 1
                rec = replay.run({"features": e["features"], "actions": e["actions"],
                                  "timestamps": e["timestamps"]}, truth=False,
                                 episode_id=f"{fault}{ramp}_{i}").episode
                t_alert = rec.first_alert_t
                if t_alert is not None and t_alert >= onset_s - 0.4:
                    hits += 1
                    lats.append(t_alert - onset_s)
            rows.append({"fault": fault, "ramp_steps": ramp, "ramp_s": ramp / combo.fps,
                         "broke": broke, "n": n_cell, "detected": hits,
                         "detection": str(wilson(hits, broke)) if broke else "n/a",
                         "median_latency_s": float(np.median(lats)) if lats else None})
            print(f"{fault} ramp {ramp/combo.fps:.1f}s: broke {broke}/{n_cell}, detected {hits}/{broke}, "
                  f"median latency {rows[-1]['median_latency_s']}", flush=True)

    lines = ["# Guard detection when the fault arrives gradually", "",
             f"Policy {combo.repo} live in {combo.env_id}, calibration pool {len(pool)} saved nominal "
             f"episodes, alpha 0.05, max method. Each fault begins at t = {onset_s:.1f} s and reaches "
             "full strength over the ramp time shown. A ramp of 0.0 s is the step change every "
             "previously published number used. Detection counts only episodes the fault actually broke.",
             "", "| fault | ramp | broke the task | detected | detection rate | median latency after onset |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lat = "n/a" if r["median_latency_s"] is None else f"{r['median_latency_s']:.2f} s"
        lines.append(f"| {r['fault']} | {r['ramp_s']:.1f} s | {r['broke']}/{r['n']} | "
                     f"{r['detected']}/{r['broke']} | {r['detection']} | {lat} |")
    lines += ["", f"Total wall time {time.time()-t0:.0f} s.", "", "GUARD_RAMP_COMPLETE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "guard_ramp.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "guard_ramp.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    print("GUARD_RAMP_COMPLETE", flush=True)
    return 0


def mode_combo(out_dir: Path, combo_name: str, n_cal: int, n_nom: int, n_fault: int) -> int:
    from policy_zoo import run_episode, smoke_test

    from robotruth.guard import OfflineReplay
    from robotruth.stats.intervals import wilson

    combo, env, policy, device, torch = _setup(combo_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rate, usable = smoke_test(combo, env, policy, device, torch, n=10)
    if not usable:
        msg = (f"# Guard detection on {combo.name}: not run\n\n"
               f"Policy {combo.repo} scored {rate:.0%} success in a 10-episode smoke test in "
               f"{combo.env_id}, against roughly {combo.expected_success:.0%} on its model card. "
               "Calibrating a guard on a policy that does not perform the task would describe the "
               "nominal behaviour of a broken policy, so this combination is reported as unusable "
               "rather than measured. The loading path, not the guard, is the thing to fix.\n\n"
               "COMBO_SKIPPED\n")
        (out_dir / f"guard_combo_{combo.name}.md").write_text(msg, encoding="utf-8")
        print(msg, flush=True)
        return 0

    rng = np.random.default_rng(3)
    onset_s = ONSET_STEP / combo.fps
    cal = []
    for i in range(n_cal):
        cal.append(run_episode(combo, env, policy, device, torch, rng=rng))
        if (i + 1) % 20 == 0:
            print(f"cal {i+1}/{n_cal} ({sum(e['success'] for e in cal)} ok, "
                  f"{(time.time()-t0)/(i+1):.0f}s/ep)", flush=True)
    nominal = [e for e in cal if e["success"]]
    print(f"calibration pool: {len(nominal)}/{n_cal} successes", flush=True)
    if len(nominal) < 20:
        (out_dir / f"guard_combo_{combo.name}.md").write_text(
            f"# Guard detection on {combo.name}: not run\n\nOnly {len(nominal)} of {n_cal} calibration "
            "episodes succeeded, too few to calibrate a threshold that means anything.\n\nCOMBO_SKIPPED\n",
            encoding="utf-8")
        return 0

    to_dict = lambda e: {"features": e["features"], "actions": e["actions"], "timestamps": e["timestamps"]}
    guard, report = _guard([to_dict(e) for e in nominal], combo.fps)

    eval_nom = [run_episode(combo, env, policy, device, torch, rng=rng) for _ in range(n_nom)]
    nom_succ = [e for e in eval_nom if e["success"]]
    replay = OfflineReplay(guard)
    fa = 0
    for i, e in enumerate(nom_succ):
        fa += replay.run(to_dict(e), truth=True, episode_id=f"nom{i}").episode.first_alert_t is not None

    det = {}
    lats = []
    for fault in ("freeze", "offset", "noise"):
        broke, hits = 0, 0
        for i in range(n_fault):
            e = run_episode(combo, env, policy, device, torch, fault=fault, rng=rng)
            if e["success"]:
                continue
            broke += 1
            rec = replay.run(to_dict(e), truth=False, episode_id=f"{fault}{i}").episode
            if rec.first_alert_t is not None and rec.first_alert_t >= onset_s - 0.4:
                hits += 1
                lats.append(rec.first_alert_t - onset_s)
        det[fault] = (hits, broke)
        print(f"fault {fault}: broke {broke}/{n_fault}, detected {hits}/{broke}", flush=True)

    total_hits = sum(h for h, _ in det.values())
    total_broke = sum(b for _, b in det.values())
    lines = [f"# Guard detection on {combo.name}", "",
             f"Policy {combo.repo} live in {combo.env_id} at {combo.fps:g} Hz. {combo.notes}.",
             f"Smoke test {rate:.0%} success. Calibration pool {len(nominal)} nominal episodes from "
             f"{n_cal} runs. Evaluation: {len(nom_succ)} held-out successes and {n_fault} episodes per "
             f"fault, injected into the executed action at t = {onset_s:.1f} s. Alpha 0.05, max method.",
             "", "| quantity | value |", "|---|---|",
             f"| false alarms on held-out successes | {fa}/{len(nom_succ)} = {wilson(fa, len(nom_succ)) if nom_succ else 'n/a'} |"]
    for fault, (h, b) in det.items():
        lines.append(f"| {fault} detected | {h}/{b} = {wilson(h, b) if b else 'n/a'} |")
    lines += [f"| all faults pooled | {total_hits}/{total_broke} = {wilson(total_hits, total_broke) if total_broke else 'n/a'} |",
              f"| median latency after onset | {f'{np.median(lats):.2f} s' if lats else 'n/a'} |",
              "", f"Total wall time {time.time()-t0:.0f} s.", "", "COMBO_COMPLETE"]
    (out_dir / f"guard_combo_{combo.name}.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"guard_combo_{combo.name}.json").write_text(json.dumps(
        {"combo": combo.name, "repo": combo.repo, "env": combo.env_id, "smoke_success_rate": rate,
         "n_calibration": len(nominal), "false_alarms": fa, "n_held_out_success": len(nom_succ),
         "detection": {k: list(v) for k, v in det.items()},
         "median_latency_s": float(np.median(lats)) if lats else None}, indent=1, default=str),
        encoding="utf-8")
    print("COMBO_COMPLETE", flush=True)
    return 0


def main() -> int:
    mode = sys.argv[1]
    out_dir = Path(sys.argv[2])
    if mode == "fa":
        return mode_fa(out_dir, Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 400)
    if mode == "ramp":
        return mode_ramp(out_dir, Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    if mode == "combo":
        return mode_combo(out_dir, sys.argv[3],
                          int(sys.argv[4]) if len(sys.argv) > 4 else 200,
                          int(sys.argv[5]) if len(sys.argv) > 5 else 20,
                          int(sys.argv[6]) if len(sys.argv) > 6 else 10)
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
