"""Live policy-loop test of the guard and judge. Runs on the GPU box.

A real LeRobot policy (ACT, lerobot/act_aloha_sim_transfer_cube_human) runs live in its
simulator (gym-aloha AlohaTransferCube). The guard is attached to every inference step; the
judge scores each finished episode; the simulator's own success signal is ground truth.

Phases:
  1. calibration: run N_CAL live episodes, keep the successful ones as nominal, calibrate
     the guard (composite of Mahalanobis on [state, action] features and chunk consistency).
  2. live monitoring: run N_LIVE more episodes with the guard in the loop (its verdict is
     recorded, never acted on). Also run N_PERTURBED episodes with a mid-episode observation
     shift (camera bump surrogate: constant pixel gain change and state bias) to create
     controlled out-of-distribution segments.
  3. judge: fit the action-stream fusion on the calibration episodes (labels from the env),
     conformal-calibrate on half the live episodes, evaluate on the rest.

Outputs to <out_dir>: guard_metrics.md/json, judge_metrics.md/json, episodes.jsonl,
live_report.json. Usage: python ops/live_guard_test.py [out_dir] [n_cal] [n_live] [n_pert]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def make_env():
    import gymnasium as gym
    import gym_aloha  # noqa: F401
    return gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", max_episode_steps=400)


def load_policy(device: str):
    """Load the ACT checkpoint plus its normalization statistics.

    lerobot 0.6.1 drops this old checkpoint's normalize_* buffers on load ("Unexpected
    key(s)" warning) and runs the policy unnormalized: 83 percent reported success becomes
    0 percent. This is exactly the executable-policy configuration failure the robotruth
    contract module guards against ("Same Weights, Different Robot"). We reattach the
    statistics manually from the checkpoint's own safetensors.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import load_file
    policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")
    policy.to(device)
    policy.eval()
    sd = load_file(hf_hub_download("lerobot/act_aloha_sim_transfer_cube_human", "model.safetensors"))
    stats = {
        "img_mean": sd["normalize_inputs.buffer_observation_images_top.mean"].to(device),
        "img_std": sd["normalize_inputs.buffer_observation_images_top.std"].to(device),
        "state_mean": sd["normalize_inputs.buffer_observation_state.mean"].to(device),
        "state_std": sd["normalize_inputs.buffer_observation_state.std"].to(device),
        "act_mean": sd["unnormalize_outputs.buffer_action.mean"].to(device),
        "act_std": sd["unnormalize_outputs.buffer_action.std"].to(device),
    }
    return policy, stats


def obs_to_batch(obs, device, torch, gain: float = 1.0, state_bias: float = 0.0):
    img = obs["pixels"]["top"].astype(np.float32) / 255.0
    img = np.clip(img * gain, 0.0, 1.0)
    state = obs["agent_pos"].astype(np.float32) + state_bias
    return {
        "observation.images.top": torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    }


def run_episode(env, policy_and_stats, device, torch, perturb_at: int | None = None, chunk_k: int = 10):
    """Run one live episode. Returns dict with per-step features, actions, success, timing."""
    policy, stats = policy_and_stats
    obs, _ = env.reset()
    policy.reset()
    feats, acts, t_infer = [], [], []
    max_reward = 0.0
    steps = 0
    for t in range(400):
        gain, bias = (1.35, 0.06) if (perturb_at is not None and t >= perturb_at) else (1.0, 0.0)
        batch = obs_to_batch(obs, device, torch, gain, bias)
        batch["observation.images.top"] = (batch["observation.images.top"] - stats["img_mean"]) / (stats["img_std"] + 1e-8)
        batch["observation.state"] = (batch["observation.state"] - stats["state_mean"]) / (stats["state_std"] + 1e-8)
        t0 = time.time()
        with torch.no_grad():
            action = policy.select_action(batch)
        t_infer.append(time.time() - t0)
        action = action * (stats["act_std"] + 1e-8) + stats["act_mean"]
        a = action.squeeze(0).cpu().numpy()
        state = obs["agent_pos"].astype(np.float32)
        feats.append(np.concatenate([state, a]).astype(np.float32))
        acts.append(a.astype(np.float32))
        obs, reward, terminated, truncated, info = env.step(a)
        max_reward = max(max_reward, float(reward))
        steps = t + 1
        if terminated or truncated:
            break
    acts_arr = np.asarray(acts)
    chunks = np.stack([acts_arr[max(0, i - chunk_k + 1): i + 1] if i + 1 >= chunk_k else
                       np.pad(acts_arr[: i + 1], ((chunk_k - i - 1, 0), (0, 0)), mode="edge")
                       for i in range(len(acts_arr))])
    return {
        "features": np.asarray(feats), "actions": chunks, "actions_flat": acts_arr,
        "timestamps": np.arange(steps) / 50.0,
        "success": bool(max_reward >= 4.0), "max_reward": max_reward, "steps": steps,
        "duration_s": steps / 50.0, "infer_ms_mean": 1000 * float(np.mean(t_infer)),
        "infer_ms_p95": 1000 * float(np.percentile(t_infer, 95)),
        "perturbed": perturb_at is not None,
    }


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "validation" / "live_guard"
    n_cal = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    n_live = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    n_pert = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    env = make_env()
    policy = load_policy(device)  # (policy, stats) tuple

    from robotruth.guard import GuardMetrics, OfflineReplay, calibrate_guard
    from robotruth.judge import EpisodeSignals, HybridJudge, evaluate, fit_calibrator, fit_fusion

    # Phase 1: calibration rollouts (live).
    cal = []
    for i in range(n_cal):
        ep = run_episode(env, policy, device, torch)
        cal.append(ep)
        print(f"cal {i+1}/{n_cal}: success={ep['success']} steps={ep['steps']} infer={ep['infer_ms_mean']:.0f}ms", flush=True)
    nominal = [e for e in cal if e["success"]]
    print(f"calibration: {len(nominal)}/{n_cal} successes used as nominal")
    if len(nominal) < 8:
        print("not enough nominal episodes; aborting")
        return 1
    guard, cal_report = calibrate_guard(
        [{"features": e["features"], "actions": e["actions"], "timestamps": e["timestamps"]} for e in nominal],
        alpha=0.05, method="max", dt=1 / 50.0, stride=1)

    # Phase 2: live monitoring.
    live = []
    replay = OfflineReplay(guard)
    for i in range(n_live + n_pert):
        perturb_at = None if i < n_live else 150
        ep = run_episode(env, policy, device, torch, perturb_at=perturb_at)
        # Guard in the loop: step-by-step over the live episode (same path a server hook takes).
        res = replay.run({"features": ep["features"], "actions": ep["actions"], "timestamps": ep["timestamps"]},
                         truth=(None if ep["perturbed"] else ep["success"]),
                         episode_id=f"{'pert' if ep['perturbed'] else 'live'}{i}")
        ep["guard_first_alert_t"] = res.episode.first_alert_t
        ep["guard_max_level"] = res.episode.max_level
        live.append(ep)
        print(f"{'pert' if ep['perturbed'] else 'live'} {i+1}/{n_live+n_pert}: success={ep['success']} "
              f"alert={res.episode.first_alert_t} level={res.episode.max_level}", flush=True)

    live_nominal = [e for e in live if not e["perturbed"]]
    pert = [e for e in live if e["perturbed"]]
    m = GuardMetrics.from_episodes([gep for gep in guard.episodes])
    pert_detected = sum(1 for e in pert if e["guard_first_alert_t"] is not None and e["guard_first_alert_t"] >= 150 / 50.0 - 0.5)
    pert_early = sum(1 for e in pert if e["guard_first_alert_t"] is not None and e["guard_first_alert_t"] < 150 / 50.0 - 0.5)
    succ_alerts = sum(1 for e in live_nominal if e["success"] and e["guard_first_alert_t"] is not None)
    succ_hours = sum(e["duration_s"] for e in live_nominal if e["success"]) / 3600.0
    guard_md = m.to_markdown("Guard on live ACT rollouts (AlohaTransferCube)")
    guard_md += (f"\n\n- Perturbed episodes (observation shift at t=3.0 s): {pert_detected}/{len(pert)} alerted after onset, "
                 f"{pert_early} alerted before onset.\n- Alerts on successful nominal episodes: {succ_alerts} in {succ_hours:.2f} h "
                 f"({succ_alerts/succ_hours:.1f}/h)" if succ_hours > 0 else "")
    (out_dir / "guard_metrics.md").write_text(guard_md, encoding="utf-8")

    # Phase 3: judge on live outcomes.
    def sig(e, i):
        return EpisodeSignals(actions=e["actions_flat"], timestamps=e["timestamps"], instruction="transfer the cube",
                              task="aloha_transfer_cube", episode_id=str(i), gripper=e["actions_flat"][:, 6])
    train = [(sig(e, i), "success" if e["success"] else "failure") for i, e in enumerate(cal)]
    half = len(live_nominal) // 2
    calib = [(sig(e, i), "success" if e["success"] else "failure") for i, e in enumerate(live_nominal[:half])]
    test = [(sig(e, i), "success" if e["success"] else "failure") for i, e in enumerate(live_nominal[half:])]
    judge_md = "Judge skipped: not enough of both classes in live episodes."
    jm_dict = {}
    try:
        n_fail_total = sum(1 for _, l in train + calib + test if l == "failure")
        if n_fail_total >= 4 and any(l == "failure" for _, l in train):
            fusion = fit_fusion(None, train)
            judge = HybridJudge(None, fusion=fusion)
            calibrated = False
            if sum(1 for _, l in calib if l == "failure") >= 5 and sum(1 for _, l in calib if l == "success") >= 5:
                fit_calibrator(judge, calib, target_error=0.10)
                calibrated = True
            else:
                test = calib + test
            jm = evaluate(judge, test)
            judge_md = jm.to_markdown("Judge on live ACT rollouts (action stream only, env success as truth)")
            note = ("Conformal calibration on a held-out live split." if calibrated else
                    "Too few live failures for conformal calibration; judge evaluated uncalibrated (threshold 0.5, no abstain). "
                    "This is the honest small-sample mode: the policy succeeds too often to collect 5 failures per split.")
            judge_md = judge_md + chr(10) + chr(10) + note
            jm_dict = jm.to_dict()
    except Exception as e:  # noqa: BLE001
        judge_md = f"Judge phase failed non-fatally: {e}"
    (out_dir / "judge_metrics.md").write_text(judge_md, encoding="utf-8")

    report = {
        "policy": "lerobot/act_aloha_sim_transfer_cube_human", "env": "gym_aloha/AlohaTransferCube-v0",
        "device": device, "n_cal": n_cal, "n_live": n_live, "n_pert": n_pert,
        "cal_success_rate": sum(e["success"] for e in cal) / len(cal),
        "live_success_rate": sum(e["success"] for e in live_nominal) / max(1, len(live_nominal)),
        "infer_ms_mean": float(np.mean([e["infer_ms_mean"] for e in live])),
        "infer_ms_p95": float(np.mean([e["infer_ms_p95"] for e in live])),
        "guard_calibration": cal_report, "guard_metrics": m.to_dict(),
        "pert_detected_after_onset": pert_detected, "pert_total": len(pert), "pert_alert_before_onset": pert_early,
        "alerts_on_successes": succ_alerts, "success_hours": succ_hours,
        "judge": jm_dict,
    }
    (out_dir / "live_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("guard_calibration", "guard_metrics", "judge")}, indent=2))
    print("live guard test done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
