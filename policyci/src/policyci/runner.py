"""Policy runner: one policy over one battery, on the GPU box.

Writes:
  <out>/episodes.jsonl        one robotruth EpisodeRecord per scenario
  <out>/run_manifest.json     battery hash, pins, policy contract, evaluator version, seeds
  <out>/videos/<hash>_<policy>.mp4   replay video per failed scenario (optional passes too)

The manifest is what the regression engine and the Passport consume. Scenario order is the
battery order; the per-scenario policy seed is derived from (policy_seed, scenario hash) so
A-vs-A reruns with a different policy_seed measure the honest noise floor while the scenes
themselves stay bit-identical.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from robotruth.schema import EpisodeRecord, Provenance, Timing

from policyci.backend import SimBackend, pins_hash
from policyci.evaluator import EVALUATOR_VERSION, evaluate
from policyci.policy_iface import Policy
from policyci.scenario import Battery, content_hash

RUNNER_VERSION = "0.1.0"


def _scenario_policy_seed(policy_seed: int, scenario_hash: str) -> int:
    h = hashlib.sha256(f"{policy_seed}:{scenario_hash}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def run_battery(policy: Policy, backend: SimBackend, battery: Battery, out_dir: str | Path,
                run_id: str, policy_seed: int = 0, video_failures: bool = True,
                video_pass_every: int = 0, progress_every: int = 25) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    videos = out / "videos"
    pins = backend.pins()
    t0 = time.time()

    results_by_hash: dict[str, dict] = {}
    ep_path = out / "episodes.jsonl"
    with ep_path.open("w", encoding="utf-8") as f:
        for i, scen in enumerate(battery.scenarios):
            frames = []
            obs = backend.reset(scen)
            policy.reset(seed=_scenario_policy_seed(policy_seed, scen.hash))
            t_ep = time.time()
            done = False
            while not done:
                action = policy.act(obs)
                sr = backend.step(action)
                obs, done = sr.obs, sr.done
                fr = backend.render_frame()
                if fr is not None:
                    frames.append(fr)
            gt = backend.ground_truth()
            outcome, failure, gates = evaluate(gt, backend.control_hz)
            rec = EpisodeRecord(
                episode_id=f"{run_id}:{scen.index}:{scen.short}",
                task=backend.task,
                policy=policy.contract.name,
                robot=backend.backend_id,
                condition=scen.short,
                pair_id=scen.hash,
                outcome=outcome,
                failure=failure,
                timing=Timing(duration_s=time.time() - t_ep,
                              time_to_success_s=(gt.get("steps") or 0) / backend.control_hz if outcome.success else None,
                              timeout_s=backend.max_steps / backend.control_hz,
                              control_hz=backend.control_hz),
                provenance=Provenance(policy_fingerprint=policy.contract.hash,
                                      policy_name=policy.contract.name,
                                      policy_weights_sha256=policy.contract.weights_sha256,
                                      logger="policyci", logger_version=RUNNER_VERSION),
                tags=[f"scenario:{scen.hash}", f"battery:{battery.battery_hash[:12]}"],
            )
            f.write(rec.model_dump_json() + "\n")
            results_by_hash[scen.hash] = {"index": scen.index, "success": bool(outcome.success),
                                          "gates": gates, "max_reward": gt.get("max_reward"),
                                          "steps": gt.get("steps"),
                                          "failure": failure.failure_class.value if failure else None}
            want_video = frames and (
                (video_failures and not outcome.success)
                or (video_pass_every and outcome.success and scen.index % video_pass_every == 0))
            if want_video:
                try:
                    import imageio.v2 as imageio
                    videos.mkdir(exist_ok=True)
                    imageio.mimsave(videos / f"{scen.short}_{policy.contract.name}.mp4", frames, fps=12)
                except Exception as e:  # video is evidence, not a dependency
                    print(f"[policyci] video save failed for {scen.short}: {e}")
            if progress_every and (i + 1) % progress_every == 0:
                n_ok = sum(1 for r in results_by_hash.values() if r["success"])
                print(f"[policyci] {i + 1}/{len(battery)}  success so far {n_ok}/{i + 1}", flush=True)

    manifest = {
        "kind": "policyci.run",
        "runner_version": RUNNER_VERSION,
        "run_id": run_id,
        "task": backend.task,
        "backend_id": backend.backend_id,
        "battery_name": battery.name,
        "battery_hash": battery.battery_hash,
        "n_scenarios": len(battery),
        "policy": policy.contract.to_dict(),
        "policy_seed": policy_seed,
        "evaluator_version": EVALUATOR_VERSION,
        "pins": pins,
        "pins_hash": pins_hash(pins),
        "wall_clock_s": time.time() - t0,
        "results": results_by_hash,
    }
    manifest["manifest_hash"] = content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    n_ok = sum(1 for r in results_by_hash.values() if r["success"])
    print(f"[policyci] run {run_id} done: {n_ok}/{len(battery)} success, "
          f"{manifest['wall_clock_s']:.0f}s, manifest {manifest['manifest_hash'][:12]}")
    return out / "run_manifest.json"
