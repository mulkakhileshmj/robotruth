"""Why did the insertion checkpoint score 0/10? Separate a load failure from a hard task.

A policy that scores zero can be broken (normalization statistics missing or misapplied, so
it emits nonsense) or merely unsuccessful (it does the right thing and never quite finishes).
The reward trace tells them apart: gym-aloha grants partial reward for approaching, touching
and lifting, so a working-but-unsuccessful policy accumulates partial reward while a broken
one sits near zero.

Usage: python ops/diagnose_insertion.py [n_episodes]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_zoo import (COMBOS, load_policy, make_env, normalize_batch, obs_to_batch,  # noqa: E402
                        unnormalize_action)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in ("act_aloha_transfer", "act_aloha_insertion"):
        combo = COMBOS[name]
        env = make_env(combo)
        policy, stats = load_policy(combo, device)
        finite = {k: bool(torch.isfinite(s["a"]).all() and torch.isfinite(s["b"]).all())
                  for k, s in stats.items()}
        print(f"\n== {name} ({combo.repo})")
        print(f"   normalization all finite: {finite}")
        print(f"   action scheme {stats['action']['mode']}, "
              f"a norm {float(stats['action']['a'].norm()):.4f}, "
              f"b norm {float(stats['action']['b'].norm()):.4f}")
        peaks, actnorms = [], []
        for ep in range(n):
            obs, _ = env.reset()
            policy.reset()
            peak = 0.0
            for t in range(combo.max_steps):
                batch = normalize_batch(combo, obs_to_batch(combo, obs, device, torch), stats)
                with torch.no_grad():
                    a = policy.select_action(batch)
                a = unnormalize_action(a, stats)
                a = a.squeeze(0).cpu().numpy().astype(np.float32)
                actnorms.append(float(np.linalg.norm(a)))
                obs, reward, term, trunc, _ = env.step(a)
                peak = max(peak, float(reward))
                if term or trunc:
                    break
            peaks.append(peak)
        print(f"   peak reward per episode: {peaks} (4.0 is task success)")
        print(f"   executed action norm: median {np.median(actnorms):.3f}, "
              f"p95 {np.percentile(actnorms, 95):.3f}")
        env.close()
    print("\nDIAGNOSE_INSERTION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
