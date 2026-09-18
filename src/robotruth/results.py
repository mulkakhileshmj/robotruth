"""Episode results table: the minimal record robotruth needs to compare policies.

CSV columns (header required, extra columns preserved):
    episode_id, policy, task, success, time_to_success, timeout, pair_id, unit_id, session_id, score

- success: 0 or 1
- time_to_success: seconds, empty if not succeeded
- timeout: seconds allowed (used for censoring)
- pair_id: same value for matched A/B trials (from `robotruth stats schedule`)
- score: optional graded score in [0, 1]; falls back to success
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REQUIRED = ("episode_id", "policy", "task", "success")


@dataclass
class Results:
    rows: list[dict] = field(default_factory=list)

    @classmethod
    def read_csv(cls, path: Path | str) -> "Results":
        with Path(path).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"results CSV missing required columns: {missing}")
            rows = [dict(r) for r in reader]
        for r in rows:
            r["success"] = int(float(r["success"]))
            tts = r.get("time_to_success", "")
            r["time_to_success"] = float(tts) if tts not in ("", None) else None
            to = r.get("timeout", "")
            r["timeout"] = float(to) if to not in ("", None) else None
            sc = r.get("score", "")
            r["score"] = float(sc) if sc not in ("", None) else float(r["success"])
        return cls(rows)

    def policies(self) -> list[str]:
        return sorted({r["policy"] for r in self.rows})

    def tasks(self) -> list[str]:
        return sorted({r["task"] for r in self.rows})

    def subset(self, policy: Optional[str] = None, task: Optional[str] = None) -> "Results":
        out = [r for r in self.rows if (policy is None or r["policy"] == policy) and (task is None or r["task"] == task)]
        return Results(out)

    def successes(self) -> np.ndarray:
        return np.array([r["success"] for r in self.rows], dtype=int)

    def scores(self) -> np.ndarray:
        return np.array([r["score"] for r in self.rows], dtype=float)

    def times(self) -> tuple[np.ndarray, np.ndarray]:
        """(time, succeeded) with failures placed at their timeout for censoring."""
        t, e = [], []
        for r in self.rows:
            if r["success"] == 1 and r["time_to_success"] is not None:
                t.append(r["time_to_success"]); e.append(1)
            else:
                to = r["timeout"] if r["timeout"] is not None else (r["time_to_success"] or float("nan"))
                t.append(to); e.append(0)
        return np.array(t, dtype=float), np.array(e, dtype=int)

    def paired(self, a: str, b: str, task: Optional[str] = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Match rows of policy a and b by pair_id (and task). Returns (scores_a, scores_b, pair_ids)."""
        def key(r):
            return (r.get("pair_id") or "", r["task"])
        ra = {key(r): r for r in self.subset(a, task).rows if r.get("pair_id")}
        rb = {key(r): r for r in self.subset(b, task).rows if r.get("pair_id")}
        common = sorted(set(ra) & set(rb))
        xa = np.array([ra[k]["score"] for k in common], dtype=float)
        xb = np.array([rb[k]["score"] for k in common], dtype=float)
        return xa, xb, [k[0] for k in common]

    def has_pairs(self) -> bool:
        return any(r.get("pair_id") for r in self.rows)
