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
                video_pass_every: int = 0, progress_every: int = 25,
                shard: int = 0, num_shards: int = 1) -> Path:
    """Run `policy` over the battery, or over one shard of it.

    Sharding is a pure wall-clock optimisation and never changes a result: scenario
    identity, the per-scenario policy seed and the evaluator are all functions of the
    scenario hash alone, so shard k of n produces exactly the records it would have
    produced in a single-process run. `merge_shards` recombines them and refuses to
    emit a manifest until every scenario in the battery is covered exactly once.
    """
    if num_shards < 1 or not (0 <= shard < num_shards):
        raise ValueError(f"bad shard {shard}/{num_shards}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    videos = out / "videos"
    pins = backend.pins()
    t0 = time.time()

    my_scenarios = [s for s in battery.scenarios if s.index % num_shards == shard]
    results_by_hash: dict[str, dict] = {}
    ep_path = out / "episodes.jsonl"
    with ep_path.open("w", encoding="utf-8") as f:
        for i, scen in enumerate(my_scenarios):
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
                                          "failure": str(failure.failure_class) if failure else None}
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
                rate = (time.time() - t0) / (i + 1)
                eta = rate * (len(my_scenarios) - i - 1) / 60
                print(f"[policyci] {run_id} shard {shard}/{num_shards}: {i + 1}/{len(my_scenarios)} "
                      f"success {n_ok}/{i + 1}  {rate:.1f}s/ep  eta {eta:.0f}m", flush=True)

    manifest = {
        "kind": "policyci.run",
        "runner_version": RUNNER_VERSION,
        "run_id": run_id,
        "task": backend.task,
        "backend_id": backend.backend_id,
        "battery_name": battery.name,
        "battery_hash": battery.battery_hash,
        "n_scenarios": len(battery),
        "shard": shard,
        "num_shards": num_shards,
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
    print(f"[policyci] run {run_id} shard {shard}/{num_shards} done: {n_ok}/{len(my_scenarios)} success, "
          f"{manifest['wall_clock_s']:.0f}s, manifest {manifest['manifest_hash'][:12]}")
    return out / "run_manifest.json"


def merge_shards(shard_manifests: list[str | Path], battery: Battery, out_dir: str | Path) -> Path:
    """Recombine shard manifests into one run manifest, fail-closed.

    Refuses unless every shard agrees on battery, policy contract, evaluator and pins,
    and unless the union covers every scenario in the battery exactly once. A partial
    run must not be presentable as a complete one.
    """
    manifests = [json.loads(Path(p).read_text(encoding="utf-8")) for p in shard_manifests]
    if not manifests:
        raise ValueError("no shard manifests given")
    base = manifests[0]
    for m in manifests[1:]:
        for field in ("battery_hash", "evaluator_version", "pins_hash", "backend_id", "policy_seed"):
            if m[field] != base[field]:
                raise ValueError(f"shard disagreement on {field}: {m[field]} vs {base[field]}")
        if m["policy"]["hash"] != base["policy"]["hash"]:
            raise ValueError("shard disagreement on the policy contract hash")

    merged: dict[str, dict] = {}
    for m in manifests:
        for h, r in m["results"].items():
            if h in merged:
                raise ValueError(f"scenario {h[:12]} ran in more than one shard")
            merged[h] = r
    expected = {s.hash for s in battery.scenarios}
    missing = expected - set(merged)
    extra = set(merged) - expected
    if missing or extra:
        raise ValueError(f"incomplete merge: {len(missing)} scenarios missing, {len(extra)} unexpected")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(base)
    manifest.pop("manifest_hash", None)
    manifest["shard"] = 0
    manifest["num_shards"] = 1
    manifest["merged_from"] = sorted(m["manifest_hash"] for m in manifests)
    manifest["wall_clock_s"] = max(m["wall_clock_s"] for m in manifests)
    manifest["wall_clock_s_summed"] = sum(m["wall_clock_s"] for m in manifests)
    manifest["results"] = merged
    manifest["n_scenarios"] = len(merged)
    manifest["manifest_hash"] = content_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    with (out / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for p in shard_manifests:
            ep = Path(p).parent / "episodes.jsonl"
            if ep.exists():
                f.write(ep.read_text(encoding="utf-8"))
    n_ok = sum(1 for r in merged.values() if r["success"])
    print(f"[policyci] merged {len(manifests)} shards of {base['run_id']}: "
          f"{n_ok}/{len(merged)} success, manifest {manifest['manifest_hash'][:12]}")
    return out / "run_manifest.json"
