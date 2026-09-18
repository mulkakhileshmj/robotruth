"""Open VLM judge on the Guardian UR5 execution-failure set (exact layout). GPU box only.

Layout of paulpacaud/ur5fail_test_dataset: records.tar.gz (per-episode PNG frames, several
viewpoints and timesteps) plus metadata_execution.jsonl with one record per judged episode:
task_instruction, episode_id, images (paths inside the tar), failure_mode (a class string;
success episodes carry a success-like value or no failure), detailed_subtask_name.

We take viewpoint 0 frames in timestep order, binary truth = failure_mode says success vs
anything else, and evaluate OpenVLMBackend. Usage:
    python ops/ur5fail_eval.py <out_dir> [max_episodes] [model_id]
"""

from __future__ import annotations

import collections
import json
import sys
import tarfile
from pathlib import Path

SUCCESS_TOKENS = ("success", "no_failure", "no failure", "none", "correct", "ground_truth", "ground truth")


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "validation" / "vlm_judge"
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    model_id = sys.argv[3] if len(sys.argv) > 3 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("paulpacaud/ur5fail_test_dataset", repo_type="dataset"))
    extract = Path.home() / "ur5fail_records"
    if not (extract / "records").exists():
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(root / "records.tar.gz") as tf:
            tf.extractall(extract)
        print("extracted records")
    metas = [json.loads(l) for l in (root / "metadata_execution.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(metas)} metadata records")
    label_counts = collections.Counter(str(m.get("failure_mode", "")).strip().lower() for m in metas)
    print("failure_mode values:", dict(label_counts))

    episodes = []
    for m in metas:
        imgs = [extract / p for p in m.get("images", [])]
        # viewpoint 0 only, in timestep order (paths look like <t>_img_viewpoint_<v>.png)
        def order(p):
            head = p.name.split("_")[0]
            return (0, int(head)) if head.isdigit() else (1, 10**9)  # named steps like "end" go last
        v0 = sorted([p for p in imgs if p.name.endswith("viewpoint_0.png") and p.exists()], key=order)
        if len(v0) < 2:
            v0 = sorted([p for p in imgs if p.exists()])[:6]
        if not v0:
            continue
        fm = str(m.get("failure_mode", "")).strip().lower()
        truth = "success" if (not fm or any(t in fm for t in SUCCESS_TOKENS)) else "failure"
        episodes.append({"id": str(m.get("episode_id")), "frames": [str(p) for p in v0],
                         "task": str(m.get("task_instruction", "manipulation task")), "truth": truth, "failure_mode": fm})
        if len(episodes) >= max_episodes:
            break
    counts = collections.Counter(e["truth"] for e in episodes)
    print(f"{len(episodes)} episodes usable; truth counts: {dict(counts)}")

    from robotruth.judge import metrics_from_predictions
    from robotruth.judge.open_vlm import OpenVLMBackend
    kw = {"model_id": model_id} if model_id else {}
    backend = OpenVLMBackend(**kw)
    decisions, labels, durations, verdicts = [], [], [], []
    import time
    t0 = time.time()
    for i, e in enumerate(episodes):
        v = backend.judge(e["frames"], e["task"], e["task"])
        decisions.append("success" if v.success_prob >= 0.5 else "failure")
        labels.append(e["truth"])
        durations.append(30.0)
        verdicts.append({"id": e["id"], "failure_mode": e["failure_mode"], "truth": e["truth"],
                         "success_prob": v.success_prob, "predicted_class": v.failure_class,
                         "parse_error": v.parse_error, "rationale": v.rationale[:200]})
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(episodes)} judged ({(time.time()-t0)/(i+1):.1f}s per episode)", flush=True)
    m = metrics_from_predictions(decisions, labels, durations)
    md = m.to_markdown(f"Open VLM judge ({backend.name}) on ur5fail execution test set")
    n_parse = sum(1 for v in verdicts if v["parse_error"])
    md += f"\n\nParse or backend errors: {n_parse}/{len(verdicts)}. Truth from failure_mode; success tokens: {SUCCESS_TOKENS}."
    (out_dir / "metrics.md").write_text(md, encoding="utf-8")
    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(m.to_dict(), indent=1, default=str), encoding="utf-8")
    print(md)
    print("VLM_EVAL_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
