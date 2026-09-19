"""Guard stability across calibration draws. Runs on the GPU box.

The 2026-09-18 live test showed the max-method threshold is noisy at ~20 calibration
episodes: one draw caught every policy-breaking perturbation, another caught none, while the
false-alarm bound held in both. This experiment settles it with more data and both methods.

Design:
  - one shared evaluation pool, collected live: N_LIVE nominal episodes plus N_PERT episodes
    with an observation shift at t = 3.0 s (the camera-bump surrogate);
  - N_SEEDS independent calibration pools of N_CAL live episodes each;
  - for each (seed, method in {max, bonferroni}): calibrate on that pool's successes and
    replay the SAME shared evaluation pool.
Detection then varies only through the calibration draw, which is the quantity under test.

Outputs: guard_stability.md and .json in <out_dir>.
Usage: python ops/guard_stability.py [out_dir] [n_cal] [n_live] [n_pert] [n_seeds]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "validation" / "guard_stability"
    n_cal = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    n_live = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    n_pert = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_guard_test import load_policy, make_env, run_episode
    from robotruth.guard import GuardMetrics, OfflineReplay, calibrate_guard

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env()
    policy = load_policy(device)

    def collect(n, tag, perturb_at=None):
        eps = []
        for i in range(n):
            ep = run_episode(env, policy, device, torch, perturb_at=perturb_at)
            eps.append(ep)
            print(f"{tag} {i+1}/{n}: success={ep['success']}", flush=True)
        return eps

    t0 = time.time()
    # Shared evaluation pool, collected once.
    eval_nom = collect(n_live, "eval-nominal")
    eval_pert = collect(n_pert, "eval-perturbed", perturb_at=150)
    # Independent calibration pools.
    pools = []
    for s in range(n_seeds):
        pools.append([e for e in collect(n_cal, f"cal-seed{s}") if e["success"]])
        print(f"seed {s}: {len(pools[-1])} nominal episodes", flush=True)

    to_dict = lambda e: {"features": e["features"], "actions": e["actions"], "timestamps": e["timestamps"]}
    onset_s = 150 / 50.0
    rows = []
    for s, pool in enumerate(pools):
        for method in ("max", "bonferroni"):
            guard, rep = calibrate_guard([to_dict(e) for e in pool], alpha=0.05, method=method, dt=1 / 50.0, stride=1)
            replay = OfflineReplay(guard)
            for i, e in enumerate(eval_nom):
                replay.run(to_dict(e), truth=(e["success"] if not e["perturbed"] else None), episode_id=f"nom{i}")
            pert_hits, pert_early = 0, 0
            for i, e in enumerate(eval_pert):
                r = replay.run(to_dict(e), truth=None, episode_id=f"pert{i}")
                t_alert = r.episode.first_alert_t
                if t_alert is not None and t_alert >= onset_s - 0.5:
                    pert_hits += 1
                elif t_alert is not None:
                    pert_early += 1
            m = GuardMetrics.from_episodes(guard.episodes)
            succ_eps = [e for e in eval_nom if e["success"]]
            alarms_on_succ = sum(1 for ge in guard.episodes if ge.truth is True and ge.detected)
            hours = sum(e["duration_s"] for e in succ_eps) / 3600.0
            broke = [e for e in eval_pert if not e["success"]]
            rows.append({
                "seed": s, "method": method, "n_nominal_cal": len(pool),
                "pert_alert_after_onset": pert_hits, "pert_total": len(eval_pert),
                "pert_broken": len(broke), "pert_alert_before_onset": pert_early,
                "alarms_on_nominal_successes": alarms_on_succ, "n_nominal_successes": len(succ_eps),
                "false_alarms_per_hour": (alarms_on_succ / hours) if hours > 0 else None,
                "detection_rate_true_failures": (m.detection_rate.estimate if getattr(m, "detection_rate", None) else None),
            })
            print(f"seed {s} {method}: pert {pert_hits}/{len(eval_pert)} after onset, "
                  f"alarms on successes {alarms_on_succ}/{len(succ_eps)}", flush=True)

    # Report.
    lines = ["# Guard stability across calibration draws", "",
             f"Policy lerobot/act_aloha_sim_transfer_cube_human live in gym-aloha. Shared evaluation pool: "
             f"{n_live} nominal + {n_pert} perturbed episodes (observation shift at {onset_s:.1f} s). "
             f"{n_seeds} independent calibration pools of {n_cal} live episodes; both conformal methods at alpha 0.05.", "",
             "| seed | method | nominal cal eps | perturbed alerted after onset | early alerts | alarms on successes | false alarms/h |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        fa = f"{r['false_alarms_per_hour']:.1f}" if r["false_alarms_per_hour"] is not None else "n/a"
        lines.append(f"| {r['seed']} | {r['method']} | {r['n_nominal_cal']} | {r['pert_alert_after_onset']}/{r['pert_total']} "
                     f"| {r['pert_alert_before_onset']} | {r['alarms_on_nominal_successes']}/{r['n_nominal_successes']} | {fa} |")
    det = {}
    for r in rows:
        det.setdefault(r["method"], []).append(r["pert_alert_after_onset"] / max(1, r["pert_total"]))
    lines += ["", "Spread of perturbed-episode detection across seeds:"]
    for method, vals in det.items():
        lines.append(f"- {method}: min {min(vals):.2f}, max {max(vals):.2f} over {len(vals)} draws")
    lines += ["", f"Total wall time {time.time()-t0:.0f} s.", "", "GUARD_STABILITY_COMPLETE"]
    (out_dir / "guard_stability.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "guard_stability.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
