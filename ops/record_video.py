"""Record a video of the live ACT policy running in gym-aloha. Runs on the GPU box.

Usage: python ops/record_video.py [out.mp4] [n_episodes]
Records the top camera of each episode, stitches them with a caption strip showing episode
number, live success and step count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "validation" / "live_guard" / "act_aloha_live.mp4"
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_guard_test import load_policy, make_env, obs_to_batch
    import cv2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env()
    policy, stats = load_policy(device)
    writer = imageio.get_writer(str(out), fps=50, codec="libx264", quality=8)
    results = []
    for ep in range(n_eps):
        obs, _ = env.reset()
        policy.reset()
        max_reward, steps = 0.0, 0
        for t in range(400):
            frame = obs["pixels"]["top"].copy()
            batch = obs_to_batch(obs, device, torch)
            batch["observation.images.top"] = (batch["observation.images.top"] - stats["img_mean"]) / (stats["img_std"] + 1e-8)
            batch["observation.state"] = (batch["observation.state"] - stats["state_mean"]) / (stats["state_std"] + 1e-8)
            with torch.no_grad():
                action = policy.select_action(batch)
            action = action * (stats["act_std"] + 1e-8) + stats["act_mean"]
            a = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(a)
            max_reward = max(max_reward, float(reward))
            steps = t + 1
            label = f"ep {ep+1}  t={t}  reward={int(max_reward)}/4"
            cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(frame)
            if terminated or truncated:
                break
        ok = max_reward >= 4.0
        results.append(ok)
        end = obs["pixels"]["top"].copy()
        verdict = "SUCCESS" if ok else "FAILURE"
        colour = (40, 200, 40) if ok else (30, 30, 220)
        cv2.putText(end, f"ep {ep+1}: {verdict} in {steps} steps", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(end, f"ep {ep+1}: {verdict} in {steps} steps", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)
        for _ in range(50):
            writer.append_data(end)
        print(f"episode {ep+1}: {verdict} in {steps} steps", flush=True)
    writer.close()
    print(f"wrote {out} ({sum(results)}/{n_eps} successes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
