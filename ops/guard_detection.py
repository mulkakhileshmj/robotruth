"""Guard detection experiment: faults that actually break the policy. GPU box only.

The 2026-09-19 stability run could not measure detection because a mild observation shift
broke the policy in only 1 of 12 episodes. This experiment injects policy-side faults that
reliably fail the task, the kind a runtime monitor exists to catch, and measures detection,
detection latency and false alarms together on one large calibration pool.

Faults, injected at t = 3.0 s into the EXECUTED action (the guard sees executed actions,
exactly what a policy-server hook sees):
  freeze   hold the last commanded action for the rest of the episode (stalled policy)
  offset   add a constant 0.15 rad to every joint command (miscalibrated actuator)
  noise    add N(0, 0.2) to every joint command (erratic policy)

Design:
  - N_CAL live nominal episodes; the successes form the calibration pool (expect ~85%).
  - Evaluation pool: N_NOM held-out nominal episodes plus N_FAULT episodes per fault type.
  - Methods max and bonferroni fitted on the full pool, plus three random 45-episode
    sub-draws of the same pool to compare small-sample variance against the full pool.
  - A faulted episode counts for detection only if the task actually failed.

Outputs guard_detection.md and .json in <out_dir>; prints GUARD_DETECTION_COMPLETE.
Usage: python ops/guard_detection.py [out_dir] [n_cal] [n_nom] [n_fault_each]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ONSET_STEP = 150
FPS = 50.0
ONSET_S = ONSET_STEP / FPS


def run_episode_fault(env, policy_and_stats, device, torch, fault: str | None, rng, chunk_k: int = 10):
    """Live episode with an optional executed-action fault from ONSET_STEP onward."""
    policy, stats = policy_and_stats
    obs, _ = env.reset()
    policy.reset()
    feats, acts = [], []
    max_reward = 0.0
    steps = 0
    held = None
    for t in range(400):
        batch_img = torch.from_numpy((obs["pixels"]["top"].astype(np.float32) / 255.0)).permute(2, 0, 1).unsqueeze(0).to(device)
        batch_state = torch.from_numpy(obs["agent_pos"].astype(np.float32)).unsqueeze(0).to(device)
        batch = {
            "observation.images.top": (batch_img - stats["img_mean"]) / (stats["img_std"] + 1e-8),
            "observation.state": (batch_state - stats["state_mean"]) / (stats["state_std"] + 1e-8),
        }
        with torch.no_grad():
            action = policy.select_action(batch)
        action = action * (stats["act_std"] + 1e-8) + stats["act_mean"]
        a = action.squeeze(0).cpu().numpy().astype(np.float32)
        if fault and t >= ONSET_STEP:
            if fault == "freeze":
                a = held if held is not None else a
            elif fault == "offset":
                a = a + 0.15
            elif fault == "noise":
                a = a + rng.normal(0.0, 0.2, size=a.shape).astype(np.float32)
        held = a
        state = obs["agent_pos"].astype(np.float32)
        feats.append(np.concatenate([state, a]))
        acts.append(a)
        obs, reward, terminated, truncated, info = env.step(a)
        max_reward = max(max_reward, float(reward))
        steps = t + 1
        if terminated or truncated:
            break
    acts_arr = np.asarray(acts)
    chunks = np.stack([acts_arr[max(0, i - chunk_k + 1): i + 1] if i + 1 >= chunk_k else
                       np.pad(acts_arr[: i + 1], ((chunk_k - i - 1, 0), (0, 0)), mode="edge")
                       for i in range(len(acts_arr))])
    return {"features": np.asarray(feats), "actions": chunks, "timestamps": np.arange(steps) / FPS,
            "success": bool(max_reward >= 4.0), "steps": steps, "duration_s": steps / FPS, "fault": fault or "none"}


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "validation" / "guard_detection"
    n_cal = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_nom = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    n_fault = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_guard_test import load_policy, make_env
    from robotruth.guard import OfflineReplay, calibrate_guard
    from robotruth.stats.intervals import wilson

    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env()
    policy = load_policy(device)
    t0 = time.time()

    cal = []
    for i in range(n_cal):
        ep = run_episode_fault(env, policy, device, torch, None, rng)
        cal.append(ep)
        if (i + 1) % 10 == 0:
            print(f"cal {i+1}/{n_cal} ({sum(e['success'] for e in cal)} ok, {(time.time()-t0)/(i+1):.0f}s/ep)", flush=True)
    nominal = [e for e in cal if e["success"]]
    print(f"calibration pool: {len(nominal)}/{n_cal} successes", flush=True)

    eval_nom = [run_episode_fault(env, policy, device, torch, None, rng) for _ in range(n_nom)]
    print(f"eval nominal: {sum(e['success'] for e in eval_nom)}/{n_nom} successes", flush=True)
    eval_fault = []
    for fault in ("freeze", "offset", "noise"):
        eps = [run_episode_fault(env, policy, device, torch, fault, rng) for _ in range(n_fault)]
        eval_fault += eps
        print(f"fault {fault}: {sum(not e['success'] for e in eps)}/{n_fault} broke the task", flush=True)

    ep_dir = out_dir / "episodes"
    ep_dir.mkdir(exist_ok=True)
    for group, eps in (("cal", cal), ("evalnom", eval_nom), ("fault", eval_fault)):
        for i, e in enumerate(eps):
            np.savez_compressed(ep_dir / f"{group}_{i:04d}.npz", features=e["features"], actions=e["actions"],
                                timestamps=e["timestamps"], success=e["success"], fault=str(e.get("fault")))
    print(f"episodes saved to {ep_dir} for offline recalibration", flush=True)

    to_dict = lambda e: {"features": e["features"], "actions": e["actions"], "timestamps": e["timestamps"]}
    pools = [("full", nominal)]
    for s in range(3):
        idx = np.random.default_rng(100 + s).choice(len(nominal), size=min(45, len(nominal)), replace=False)
        pools.append((f"sub45-{s}", [nominal[i] for i in idx]))

    rows = []
    for pool_name, pool in pools:
        for method in ("max", "bonferroni"):
            guard, rep = calibrate_guard([to_dict(e) for e in pool], alpha=0.05, method=method, dt=1 / FPS, stride=1)
            replay = OfflineReplay(guard)
            fa = 0
            nom_succ = [e for e in eval_nom if e["success"]]
            for i, e in enumerate(nom_succ):
                r = replay.run(to_dict(e), truth=True, episode_id=f"nom{i}")
                fa += r.episode.first_alert_t is not None
            det = {}
            lat = []
            for fault in ("freeze", "offset", "noise"):
                broke = [e for e in eval_fault if e["fault"] == fault and not e["success"]]
                hits = 0
                for i, e in enumerate(broke):
                    r = replay.run(to_dict(e), truth=False, episode_id=f"{fault}{i}")
                    t_alert = r.episode.first_alert_t
                    if t_alert is not None and t_alert >= ONSET_S - 0.4:
                        hits += 1
                        lat.append(t_alert - ONSET_S)
                det[fault] = (hits, len(broke))
            saturated = bool(rep.get("conformal", {}).get("saturated", rep.get("saturated", False))) if isinstance(rep, dict) else False
            rows.append({"pool": pool_name, "n_pool": len(pool), "method": method,
                         "false_alarms": fa, "n_nominal": len(nom_succ),
                         "det_freeze": det["freeze"], "det_offset": det["offset"], "det_noise": det["noise"],
                         "median_latency_s": float(np.median(lat)) if lat else None,
                         "saturated": saturated})
            print(f"{pool_name} {method}: fa {fa}/{len(nom_succ)}, freeze {det['freeze']}, offset {det['offset']}, noise {det['noise']}", flush=True)

    lines = ["# Guard detection with policy-breaking faults", "",
             f"Policy lerobot/act_aloha_sim_transfer_cube_human live in gym-aloha. Calibration pool {len(nominal)} nominal episodes "
             f"(from {n_cal} runs). Evaluation: {n_nom} held-out nominal plus {n_fault} episodes per fault, faults injected into the "
             f"executed action at t = {ONSET_S:.1f} s. Alpha 0.05. An episode counts for detection only if the task actually failed.", "",
             "| pool | n | method | false alarms on successes | freeze det | offset det | noise det | median latency s |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def frac(p):
            return f"{p[0]}/{p[1]}"
        lat_s = f"{r['median_latency_s']:.2f}" if r["median_latency_s"] is not None else "n/a"
        lines.append(f"| {r['pool']} | {r['n_pool']} | {r['method']} | {r['false_alarms']}/{r['n_nominal']} | "
                     f"{frac(r['det_freeze'])} | {frac(r['det_offset'])} | {frac(r['det_noise'])} | {lat_s} |")
    full_max = next(r for r in rows if r["pool"] == "full" and r["method"] == "max")
    total_hits = sum(full_max[k][0] for k in ("det_freeze", "det_offset", "det_noise"))
    total_broke = sum(full_max[k][1] for k in ("det_freeze", "det_offset", "det_noise"))
    if total_broke:
        w = wilson(total_hits, total_broke)
        lines += ["", f"Full-pool max method, all faults pooled: detection {total_hits}/{total_broke} = {w}."]
    fa_w = wilson(full_max["false_alarms"], full_max["n_nominal"])
    lines += [f"Full-pool max method false-alarm rate on held-out successes: {fa_w} (bound: 0.05).",
              "", f"Total wall time {time.time()-t0:.0f} s.", "", "GUARD_DETECTION_COMPLETE"]
    (out_dir / "guard_detection.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "guard_detection.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
