"""Record the guard watching a live ACT policy, with a stall injected mid-episode.

Calibrates the guard from the episodes that ops/guard_detection.py saved (no new
calibration rollouts needed), then runs episodes live with the guard attached to every
step and burns its verdict into the frame: level, score, per-head excess over threshold.
One nominal episode and one freeze episode, so the video shows both the quiet case and
the catch.

Usage: python ops/record_guard_video.py <episodes_dir> [out.mp4]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ONSET_STEP = 150
FPS = 50.0
LEVEL_COLOUR = {"ok": (40, 200, 40), "slow": (0, 200, 255), "handover": (30, 30, 220), "stop": (30, 30, 220)}


def load_calibration(ep_dir: Path, limit: int = 150) -> list[dict]:
    eps = []
    for p in sorted(ep_dir.glob("cal_*.npz"))[:limit]:
        d = np.load(p, allow_pickle=True)
        if not bool(d["success"]):
            continue
        eps.append({"features": d["features"], "actions": d["actions"], "timestamps": d["timestamps"]})
    return eps


def main() -> int:
    ep_dir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "validation" / "guard_video" / "guard_stall.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import imageio.v2 as imageio
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_guard_test import load_policy, make_env

    from robotruth.guard import calibrate_guard

    cal = load_calibration(ep_dir)
    print(f"calibrating guard on {len(cal)} saved nominal episodes", flush=True)
    guard, report = calibrate_guard(cal, alpha=0.05, method="max", dt=1 / FPS, stride=1)
    print(f"stall head alpha {report['stall_head']['alpha']}, saturated {report['stall_head']['saturated']}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env()
    policy, stats = load_policy(device)
    rng = np.random.default_rng(0)
    writer = imageio.get_writer(str(out), fps=50, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=1)

    for fault in (None, "freeze"):
        obs, _ = env.reset()
        policy.reset()
        guard.start_episode(episode_id=fault or "nominal")
        acts: list[np.ndarray] = []
        held = None
        max_reward, steps, caught_at = 0.0, 0, None
        for t in range(400):
            frame = obs["pixels"]["top"].copy()
            img = torch.from_numpy(obs["pixels"]["top"].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            state_t = torch.from_numpy(obs["agent_pos"].astype(np.float32)).unsqueeze(0).to(device)
            batch = {"observation.images.top": (img - stats["img_mean"]) / (stats["img_std"] + 1e-8),
                     "observation.state": (state_t - stats["state_mean"]) / (stats["state_std"] + 1e-8)}
            with torch.no_grad():
                action = policy.select_action(batch)
            action = action * (stats["act_std"] + 1e-8) + stats["act_mean"]
            a = action.squeeze(0).cpu().numpy().astype(np.float32)
            if fault == "freeze" and t >= ONSET_STEP:
                a = held if held is not None else a
            held = a
            acts.append(a)
            arr = np.asarray(acts)
            chunk = arr[-10:] if len(arr) >= 10 else np.pad(arr, ((10 - len(arr), 0), (0, 0)), mode="edge")
            state = obs["agent_pos"].astype(np.float32)
            ev = guard.step(features=np.concatenate([state, a]), action_chunk=chunk, t=t / FPS)
            if ev.level != "ok" and caught_at is None:
                caught_at = t

            title = "nominal episode" if fault is None else "stall injected at t = 3.0 s"
            lines = [f"robotruth guard  |  {title}",
                     f"t = {t / FPS:5.2f} s     level: {ev.level.upper()}"]
            for head, val in sorted(ev.components.items()):
                lines.append(f"  {head:<6} score over threshold: {val:+.2f}")
            if fault == "freeze" and t >= ONSET_STEP:
                lines.append("  actions frozen (policy stalled)")
            if caught_at is not None:
                lines.append(f"  ALARM raised at t = {caught_at / FPS:.2f} s")
            colour = LEVEL_COLOUR.get(ev.level, (255, 255, 255))
            for i, line in enumerate(lines):
                y = 24 + 22 * i
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            colour if i == 1 else (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(frame)

            obs, reward, terminated, truncated, info = env.step(a)
            max_reward = max(max_reward, float(reward))
            steps = t + 1
            if terminated or truncated:
                break

        rec = guard.end_episode(truth=bool(max_reward >= 4.0))
        ok = max_reward >= 4.0
        verdict = "TASK SUCCEEDED" if ok else "TASK FAILED"
        caught = "guard stayed quiet" if rec.first_alert_t is None else f"guard alarmed at {rec.first_alert_t:.2f} s"
        end = obs["pixels"]["top"].copy()
        for i, line in enumerate([verdict, caught]):
            y = 200 + 40 * i
            cv2.putText(end, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(end, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (40, 200, 40) if ok else (30, 30, 220), 2, cv2.LINE_AA)
        for _ in range(75):
            writer.append_data(end)
        print(f"{fault or 'nominal'}: {verdict}, {caught}, {steps} steps", flush=True)

    writer.close()
    print(f"wrote {out}")
    print("GUARD_VIDEO_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
