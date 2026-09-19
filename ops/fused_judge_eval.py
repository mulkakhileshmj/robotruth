"""Fused judge benchmarks on real labeled data. Runs on the GPU box.

Part A, ur5fail: the VLM channel plus split-conformal calibration with abstain. Scores are
computed once per episode, a stratified half calibrates, the other half is scored. Reports
uncalibrated vs calibrated-with-abstain side by side.

Part B, BotFails: the full fusion. Each nested LeRobot dataset under BotFails is an
expert (success) or anomaly (failure) recording with parquet actions and mp4 videos. We build
EpisodeSignals with both channels, fit the logistic fusion on a train split, conformal-
calibrate on a second, evaluate on a third, and report action-only vs VLM-only vs fused.

Usage: python ops/fused_judge_eval.py <out_dir> [model_id]
Outputs: ur5_calibrated.md, botfails_fused.md (+ .json) in <out_dir>. Prints FUSED_EVAL_COMPLETE.
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

SUCCESS_TOKENS = ("success", "no_failure", "no failure", "none", "correct", "ground_truth", "ground truth")


def stratified_split(labels: list[str], fractions=(0.4, 0.3, 0.3), seed=0):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in fractions]
    for cls in sorted(set(labels)):
        idx = rng.permutation([i for i, l in enumerate(labels) if l == cls])
        a = int(round(fractions[0] * len(idx)))
        b = a + int(round(fractions[1] * len(idx)))
        buckets[0] += list(idx[:a]); buckets[1] += list(idx[a:b]); buckets[2] += list(idx[b:])
    return [np.array(sorted(b)) for b in buckets]


# ------------------------------------------------------------------ Part A: ur5 calibrated
def part_a(out_dir: Path, backend) -> dict:
    import tarfile
    from huggingface_hub import snapshot_download
    from robotruth.judge import metrics_from_predictions
    from robotruth.judge.calibrate import Calibrator
    root = Path(snapshot_download("paulpacaud/ur5fail_test_dataset", repo_type="dataset"))
    extract = Path.home() / "ur5fail_records"
    if not (extract / "records").exists():
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(root / "records.tar.gz") as tf:
            tf.extractall(extract)
    metas = [json.loads(l) for l in (root / "metadata_execution.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    episodes = []
    for m in metas:
        imgs = [extract / p for p in m.get("images", [])]

        def order(p):
            head = p.name.split("_")[0]
            return (0, int(head)) if head.isdigit() else (1, 10**9)
        v0 = sorted([p for p in imgs if p.name.endswith("viewpoint_0.png") and p.exists()], key=order)
        if not v0:
            continue
        fm = str(m.get("failure_mode", "")).strip().lower()
        truth = "success" if (not fm or any(t in fm for t in SUCCESS_TOKENS)) else "failure"
        episodes.append({"frames": [str(p) for p in v0], "task": str(m.get("task_instruction", "task")), "truth": truth})
    labels = [e["truth"] for e in episodes]
    print(f"ur5: {len(episodes)} episodes, {collections.Counter(labels)}")
    scores = []
    t0 = time.time()
    for i, e in enumerate(episodes):
        scores.append(backend.judge(e["frames"], e["task"], e["task"]).success_prob)
        if (i + 1) % 20 == 0:
            print(f"ur5 {i+1}/{len(episodes)} ({(time.time()-t0)/(i+1):.1f}s/ep)", flush=True)
    scores = np.array(scores)
    cal_idx, test_idx = stratified_split(labels, fractions=(0.5, 0.5, 0.0))[:2]
    durations = [30.0] * len(test_idx)
    raw = metrics_from_predictions(["success" if scores[i] >= 0.5 else "failure" for i in test_idx],
                                   [labels[i] for i in test_idx], durations)
    cal = Calibrator().fit([scores[i] for i in cal_idx], [labels[i] for i in cal_idx], target_error=0.10)
    calibrated = metrics_from_predictions([cal.decide(scores[i]) for i in test_idx],
                                          [labels[i] for i in test_idx], durations)
    md = ["# ur5fail: VLM channel, uncalibrated vs conformal-calibrated with abstain", "",
          f"Backend {backend.name}. {len(cal_idx)} calibration / {len(test_idx)} test episodes (stratified).", "",
          raw.to_markdown("Uncalibrated (threshold 0.5, no abstain)"), "",
          calibrated.to_markdown("Calibrated with abstain (target selective error 0.10)")]
    (out_dir / "ur5_calibrated.md").write_text("\n".join(md), encoding="utf-8")
    return {"uncalibrated": raw.to_dict(), "calibrated": calibrated.to_dict()}


# ------------------------------------------------------------------ Part B: BotFails fused
def read_frames_from_video(path: Path, k: int = 6):
    """Sample k frames from a video without holding the whole decode in memory.

    Two traps this avoids: iio.imread returns one big array whose elements are VIEWS, so
    keeping a slice keeps the entire video alive (a real leak: 140 GB across 300 episodes);
    and decoding every frame of every video is slow. We stream with imiter and copy only
    the frames we keep.
    """
    import imageio.v3 as iio
    try:
        props = iio.improps(str(path), plugin="pyav")
        n = int(props.shape[0])
    except Exception:
        n = 0
    if n > 0:
        want = set(int(x) for x in np.linspace(0, n - 1, k))
    else:
        want = set(range(0, 600, 20))  # unknown length: take an even spread of the first frames
    out = []
    try:
        for i, frame in enumerate(iio.imiter(str(path), plugin="pyav")):
            if i in want:
                arr = np.array(frame, copy=True)          # copy breaks the view into the decoder buffer
                if arr.ndim == 3 and max(arr.shape[:2]) > 480:   # downscale early, the judge resizes anyway
                    step = max(1, max(arr.shape[:2]) // 480)
                    arr = arr[::step, ::step]
                out.append(arr)
            if n > 0 and i >= max(want):
                break
            if n == 0 and i > 600:
                break
    except Exception:
        return out
    return out


def part_b(out_dir: Path, backend) -> dict:
    from huggingface_hub import snapshot_download
    from robotruth.judge import EpisodeSignals, HybridJudge, evaluate, fit_calibrator, fit_fusion, metrics_from_predictions
    from robotruth.schema.lerobot_ingest import LeRobotDataset
    cached = Path.home() / "datasets" / "kantine__BotFails"
    if list(cached.rglob("meta/info.json")):
        local = cached
        print(f"using cached BotFails at {local}")
    else:
        local = Path(snapshot_download("kantine/BotFails", repo_type="dataset", local_dir=str(cached)))
    ds_dirs = sorted({p.parent.parent for p in Path(local).rglob("meta/info.json")})
    print(f"botfails: {len(ds_dirs)} nested datasets")
    pairs = []
    t0 = time.time()
    for d in ds_dirs:
        label = "failure" if "anomaly" in d.name.lower() else ("success" if "expert" in d.name.lower() else None)
        if label is None:
            continue
        try:
            ds = LeRobotDataset(d)
            cam = ds.camera_keys()[0] if ds.camera_keys() else None
            for ep in ds.episodes():
                frames = []
                if cam:
                    vid = d / "videos" / "chunk-000" / cam / f"episode_{ep.index:06d}.mp4"
                    if vid.exists():
                        frames = read_frames_from_video(vid)
                sig = EpisodeSignals(actions=ep.actions, timestamps=ep.timestamps, instruction=ep.task, task=d.name,
                                     episode_id=f"{d.name}#{ep.index}", states=ep.states,
                                     gripper=ep.actions[:, -1] if ep.actions.ndim == 2 and ep.actions.shape[1] > 1 else None,
                                     frames=frames or None)
                pairs.append((sig, label))
        except Exception as e:  # noqa: BLE001
            print(f"skip {d.name}: {e}")
    labels = [l for _, l in pairs]
    n_frames = sum(1 for s, _ in pairs if s.frames)
    print(f"botfails: {len(pairs)} episodes ({collections.Counter(labels)}), {n_frames} with video frames, {time.time()-t0:.0f}s to load")
    tr, ca, te = stratified_split(labels, seed=1)
    train = [pairs[i] for i in tr]; calib = [pairs[i] for i in ca]; test = [pairs[i] for i in te]

    results = {}
    reports = ["# BotFails: action-only vs VLM-only vs fused, all conformal-calibrated with abstain", "",
               f"{len(pairs)} real episodes across {len(ds_dirs)} tasks (expert = success, anomaly = failure). "
               f"Splits {len(train)}/{len(calib)}/{len(test)} stratified. Backend {backend.name}.", ""]
    variants = [("action-only", None), ("fused", backend)]
    for name, be in variants:
        fusion = fit_fusion(be, train)
        judge = HybridJudge(be, fusion=fusion)
        fit_calibrator(judge, calib, target_error=0.10)
        m = evaluate(judge, test)
        results[name] = m.to_dict()
        reports += [m.to_markdown(f"{name} judge"), ""]
        print(f"botfails {name}: done", flush=True)
    # VLM-only baseline (no fusion, raw threshold on test).
    vlm_dec, vlm_lab = [], []
    for sig, label in test:
        v = backend.judge(sig.frames, sig.instruction, sig.task)
        vlm_dec.append("success" if v.success_prob >= 0.5 else "failure")
        vlm_lab.append(label)
    m = metrics_from_predictions(vlm_dec, vlm_lab, [30.0] * len(vlm_dec))
    results["vlm-only"] = m.to_dict()
    reports += [m.to_markdown("VLM-only judge (uncalibrated)"), ""]
    (out_dir / "botfails_fused.md").write_text("\n".join(reports), encoding="utf-8")
    return results


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "validation" / "fused_judge"
    model_id = sys.argv[2] if len(sys.argv) > 2 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    from robotruth.judge.open_vlm import OpenVLMBackend
    backend = OpenVLMBackend(**({"model_id": model_id} if model_id else {}))
    only = sys.argv[3] if len(sys.argv) > 3 else ""
    a = {}
    if only != "b":
        a = part_a(out_dir, backend)
        print("PART_A_DONE", flush=True)
    b = part_b(out_dir, backend)
    (out_dir / "summary.json").write_text(json.dumps({"ur5": a, "botfails": b}, indent=1, default=str), encoding="utf-8")
    print("FUSED_EVAL_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
