"""Evaluate the open VLM judge on a real labeled failure dataset. Runs on the GPU box.

Dataset: the Guardian UR5 failure sets (paulpacaud/ur5fail_test_dataset and
bdv2fail_test_dataset on the HF hub): real robot episodes as frames with a fine-grained
execution outcome label. Everything that is not the success class counts as failure, giving
episode-level binary truth for the judge.

The script is defensive about layout: it downloads the repo, finds groups of frames per
episode plus a label, prints what it inferred, and evaluates the judge with
`metrics_from_predictions`. Usage:
    python ops/vlm_judge_eval.py <out_dir> [repo_id] [max_episodes] [model_id]
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

SUCCESS_TOKENS = {"success", "full_success", "successful", "no_failure", "correct"}


def download(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id, repo_type="dataset"))


def load_hf_parquet_episodes(root: Path, max_episodes: int):
    """The Guardian sets are HF datasets; try parquet with image columns first."""
    import pyarrow.parquet as pq
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return None
    tbl = pq.read_table(files[0])
    names = tbl.column_names
    print("parquet columns:", names)
    label_col = next((c for c in names if re.search(r"label|outcome|class|failure|answer", c, re.I)), None)
    img_cols = [c for c in names if re.search(r"image|frame|rgb|observation", c, re.I)]
    id_col = next((c for c in names if re.search(r"episode|traj|id$", c, re.I)), None)
    task_col = next((c for c in names if re.search(r"task|instruction|question|prompt", c, re.I)), None)
    if label_col is None or not img_cols:
        return None
    rows = []
    for f in files:
        t = pq.read_table(f)
        rows.extend(t.to_pylist())
        if len(rows) >= max_episodes * 4:
            break
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        key = r.get(id_col) if id_col else i
        groups[key].append(r)
    episodes = []
    for key, rs in list(groups.items())[:max_episodes]:
        frames = []
        for r in rs:
            for c in img_cols:
                v = r.get(c)
                if isinstance(v, dict) and "bytes" in v and v["bytes"]:
                    frames.append(v["bytes"])
                elif isinstance(v, (bytes, bytearray)):
                    frames.append(bytes(v))
        label_raw = str(rs[0].get(label_col, "")).strip().lower()
        task = str(rs[0].get(task_col, "manipulation task")) if task_col else "manipulation task"
        if frames:
            episodes.append({"id": str(key), "frames": frames, "label_raw": label_raw, "task": task})
    return episodes


def load_folder_episodes(root: Path, max_episodes: int):
    """Fallback: folders of images whose directory name carries the label."""
    exts = {".jpg", ".jpeg", ".png"}
    by_dir = collections.defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and ".cache" not in p.parts:
            by_dir[p.parent].append(p)
    episodes = []
    for d, files in sorted(by_dir.items()):
        if len(files) < 2:
            continue
        episodes.append({"id": str(d.relative_to(root)), "frames": [str(f) for f in sorted(files)],
                         "label_raw": d.name.lower(), "task": "manipulation task"})
        if len(episodes) >= max_episodes:
            break
    return episodes


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "validation" / "vlm_judge"
    repo_id = sys.argv[2] if len(sys.argv) > 2 else "paulpacaud/ur5fail_test_dataset"
    max_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    model_id = sys.argv[4] if len(sys.argv) > 4 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    root = download(repo_id)
    print("dataset at", root)
    episodes = load_hf_parquet_episodes(root, max_episodes) or load_folder_episodes(root, max_episodes)
    if not episodes:
        print("could not parse the dataset layout; listing top files for a human:")
        for p in list(root.rglob("*"))[:40]:
            print("  ", p.relative_to(root))
        return 1
    label_counts = collections.Counter(e["label_raw"] for e in episodes)
    print(f"{len(episodes)} episodes; raw labels: {dict(label_counts)}")

    from robotruth.judge import metrics_from_predictions
    from robotruth.judge.open_vlm import OpenVLMBackend
    kw = {"model_id": model_id} if model_id else {}
    backend = OpenVLMBackend(**kw)
    decisions, labels, durations, verdicts = [], [], [], []
    for i, e in enumerate(episodes):
        truth = "success" if any(tok in e["label_raw"] for tok in SUCCESS_TOKENS) else "failure"
        v = backend.judge(e["frames"], e["task"], e["task"])
        decisions.append("success" if v.success_prob >= 0.5 else "failure")
        labels.append(truth)
        durations.append(30.0)
        verdicts.append({"id": e["id"], "label_raw": e["label_raw"], "truth": truth,
                         "success_prob": v.success_prob, "progress": v.progress,
                         "failure_class": v.failure_class, "parse_error": v.parse_error,
                         "rationale": v.rationale[:200]})
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(episodes)} judged", flush=True)
    m = metrics_from_predictions(decisions, labels, durations)
    md = m.to_markdown(f"Open VLM judge ({backend.name}) on {repo_id}")
    n_parse = sum(1 for v in verdicts if v["parse_error"])
    md += f"\n\nParse or backend errors: {n_parse}/{len(verdicts)}."
    (out_dir / "metrics.md").write_text(md, encoding="utf-8")
    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(m.to_dict(), indent=1, default=str), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
