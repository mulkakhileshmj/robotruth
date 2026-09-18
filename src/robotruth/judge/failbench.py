"""Loader for a FailBench-style dataset (arXiv 2609.03611) and a one-call evaluation.

Nothing is downloaded. You point at a JSONL you already have, one episode per line:

    {"episode_id": "e1", "task": "insert_peg", "instruction": "insert the peg", "label": "success",
     "frames": ["frames/e1/000.jpg", ...], "duration_s": 42.0, "actions_npz": "actions/e1.npz"}

`frames` are paths, relative to the JSONL file unless absolute. `actions_npz` is optional;
when present it may hold `actions` [T, D] or [T, C, D], `timestamps` [T], and optionally
`states` [T, Ds] and `gripper` [T]. Missing timestamps are synthesised from duration_s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .features import EpisodeSignals
from .judge import HybridJudge, JudgeMetrics, evaluate

LABELS = {"success", "failure"}


def _resolve(base: Path, p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else (base / q))


def load_actions_npz(path: Path, duration_s: Optional[float]) -> dict:
    """Read an actions npz into arrays. Returns keys actions, timestamps, states, gripper."""
    with np.load(path, allow_pickle=False) as z:
        actions = np.asarray(z["actions"], dtype=float) if "actions" in z else None
        timestamps = np.asarray(z["timestamps"], dtype=float) if "timestamps" in z else None
        states = np.asarray(z["states"], dtype=float) if "states" in z else None
        gripper = np.asarray(z["gripper"], dtype=float) if "gripper" in z else None
    if actions is not None and timestamps is None:
        T = actions.shape[0]
        end = float(duration_s) if duration_s else float(max(T - 1, 0))
        timestamps = np.linspace(0.0, end, num=T)
    return {"actions": actions, "timestamps": timestamps, "states": states, "gripper": gripper}


def parse_failbench_record(rec: dict, base: Path) -> tuple[EpisodeSignals, str]:
    """Turn one JSONL record into (EpisodeSignals, label). Raises ValueError on a bad record."""
    label = str(rec.get("label", "")).strip().lower()
    if label not in LABELS:
        raise ValueError(f"episode {rec.get('episode_id')!r}: label must be success or failure, got {rec.get('label')!r}")
    duration = rec.get("duration_s")
    duration = None if duration is None else float(duration)
    arrays = {"actions": None, "timestamps": None, "states": None, "gripper": None}
    npz = rec.get("actions_npz")
    if npz:
        p = Path(_resolve(base, npz))
        if p.exists():
            arrays = load_actions_npz(p, duration)
    frames = rec.get("frames") or None
    if frames:
        frames = [_resolve(base, f) for f in frames]
    signals = EpisodeSignals(
        actions=arrays["actions"], timestamps=arrays["timestamps"], instruction=str(rec.get("instruction", "")),
        task=str(rec.get("task", "unknown")), episode_id=str(rec.get("episode_id", "")), states=arrays["states"],
        gripper=arrays["gripper"], frames=frames, duration_s=duration,
        meta={k: v for k, v in rec.items() if k not in ("frames", "actions_npz")},
    )
    return signals, label


def load_failbench_jsonl(path: Path | str) -> list[tuple[EpisodeSignals, str]]:
    """Load every record. Blank lines are skipped; malformed lines raise with the line number."""
    p = Path(path)
    base = p.parent
    out: list[tuple[EpisodeSignals, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{i}: invalid JSON ({e})") from e
            out.append(parse_failbench_record(rec, base))
    return out


def run_failbench(judge: HybridJudge, path: Path | str, alpha: float = 0.05) -> JudgeMetrics:
    """Load the JSONL and evaluate the judge on it."""
    return evaluate(judge, load_failbench_jsonl(path), alpha=alpha)
