"""Ingest a LeRobot dataset (v2.x or v3) into robotruth inputs.

Reads only files on disk: `meta/info.json`, `meta/episodes.jsonl` or `meta/episodes/*.parquet`,
`meta/tasks.jsonl` or `meta/tasks.parquet`, and the `data/**/*.parquet` shards. No lerobot
import, no video decoding.

Outputs:
- `EpisodeRecord` per episode: duration from timestamps, success from `next.success`
  (or `next.reward` at the last frame) when present, intervention spans from any boolean
  column whose name contains "intervention" or "is_human" when present, task string,
  provenance (repo id, episode index).
- excitation arrays per episode (`action` as commanded, `observation.state` as measured)
  for the unit fingerprint, when both have the same width.
- action arrays per episode for the judge and guard features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from robotruth.schema.episode import (
    EpisodeRecord,
    Intervention,
    InterventionSource,
    JudgeSource,
    Outcome,
    Provenance,
    Timing,
)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _read_parquet_dir(paths: list[Path], columns: Optional[list[str]] = None):
    import pyarrow.parquet as pq
    import pyarrow as pa
    tables = []
    for p in paths:
        schema_names = pq.read_schema(p).names
        cols = [c for c in columns if c in schema_names] if columns else None
        tables.append(pq.read_table(p, columns=cols))
    return pa.concat_tables(tables, promote_options="default") if tables else None


@dataclass
class LeRobotEpisode:
    index: int
    task: str
    timestamps: np.ndarray
    actions: np.ndarray            # [T, Da]
    states: Optional[np.ndarray]   # [T, Ds] or None
    success: Optional[bool]
    reward_last: Optional[float]
    intervention_mask: Optional[np.ndarray]  # [T] bool or None
    extra: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if self.timestamps.size > 1 else 0.0

    def interventions(self) -> list[Intervention]:
        if self.intervention_mask is None or not self.intervention_mask.any():
            return []
        out = []
        m = self.intervention_mask.astype(bool)
        t = self.timestamps - self.timestamps[0]
        start = None
        for i, v in enumerate(m):
            if v and start is None:
                start = i
            if (not v or i == len(m) - 1) and start is not None:
                end = i if not v else i
                out.append(Intervention(t_start=float(t[start]), t_end=float(t[end]), source=InterventionSource.TELEOP, reason="dataset intervention flag"))
                start = None
        return out

    def to_record(self, policy: str, repo_id: str, robot: str = "unknown", unit_id: Optional[str] = None,
                  timeout_s: Optional[float] = None) -> EpisodeRecord:
        succ = self.success
        tts = self.duration_s if succ else None
        return EpisodeRecord(
            episode_id=f"{repo_id}#{self.index}", task=self.task, instruction=self.task, policy=policy, robot=robot,
            unit_id=unit_id, session_id=repo_id,
            outcome=Outcome(success=succ, score=(1.0 if succ else 0.0) if succ is not None else None,
                            judged_by=JudgeSource.AUTOMATIC if succ is not None else JudgeSource.UNKNOWN),
            interventions=self.interventions(),
            timing=Timing(duration_s=self.duration_s, time_to_success_s=tts, timeout_s=timeout_s),
            provenance=Provenance(dataset_repo_id=repo_id, dataset_episode_index=self.index),
        )


class LeRobotDataset:
    """Minimal reader for a LeRobot dataset directory."""

    def __init__(self, root: Path | str, repo_id: Optional[str] = None):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.repo_id = repo_id or str(self.info.get("repo_id") or self.root.name)
        self.fps = float(self.info.get("fps") or 0.0)
        self.robot = str(self.info.get("robot_type") or "unknown")
        self.features = self.info.get("features") or {}
        self.tasks = self._load_tasks()
        self.episode_meta = self._load_episode_meta()

    # ---- metadata ---------------------------------------------------------------
    def _load_tasks(self) -> dict[int, str]:
        p = self.root / "meta" / "tasks.jsonl"
        if p.exists():
            return {int(r["task_index"]): str(r["task"]) for r in _read_jsonl(p)}
        p = self.root / "meta" / "tasks.parquet"
        if p.exists():
            import pyarrow.parquet as pq
            t = pq.read_table(p).to_pydict()
            if "task_index" in t and "task" in t:
                return {int(i): str(s) for i, s in zip(t["task_index"], t["task"])}
            # v3 stores task strings as the index
            names = list(t.get("task", t.get("__index_level_0__", [])))
            return {i: str(n) for i, n in enumerate(names)}
        return {}

    def _load_episode_meta(self) -> dict[int, dict]:
        p = self.root / "meta" / "episodes.jsonl"
        if p.exists():
            return {int(r["episode_index"]): r for r in _read_jsonl(p)}
        d = self.root / "meta" / "episodes"
        if d.exists():
            tbl = _read_parquet_dir(sorted(d.rglob("*.parquet")))
            if tbl is not None:
                rows = tbl.to_pylist()
                return {int(r["episode_index"]): r for r in rows}
        return {}

    def data_files(self) -> list[Path]:
        return sorted((self.root / "data").rglob("*.parquet"))

    def action_dim(self) -> Optional[int]:
        sh = (self.features.get("action") or {}).get("shape")
        return int(sh[0]) if sh else None

    def state_dim(self) -> Optional[int]:
        sh = (self.features.get("observation.state") or {}).get("shape")
        return int(sh[0]) if sh else None

    def camera_keys(self) -> list[str]:
        return [k for k, v in self.features.items() if str(v.get("dtype")) in ("video", "image")]

    # ---- episodes ---------------------------------------------------------------
    def episodes(self, max_episodes: Optional[int] = None) -> Iterator[LeRobotEpisode]:
        import pyarrow.parquet as pq
        files = self.data_files()
        if not files:
            return iter(())
        names = pq.read_schema(files[0]).names
        # Column aliases seen in the wild (DROID v3 splits action and state into named parts).
        action_col = "action" if "action" in names else next((c for c in ("action.joint_position", "action.cartesian_position") if c in names), None)
        state_col = "observation.state" if "observation.state" in names else next((c for c in ("observation.state.joint_position", "observation.state.cartesian_position") if c in names), None)
        success_col = next((c for c in ("next.success", "is_episode_successful", "success", "episode_success") if c in names), None)
        wanted = ["episode_index", "frame_index", "timestamp", "task_index", "task", "next.reward", "next.done"]
        inter_cols = [c for c in names if ("intervention" in c.lower() or "is_human" in c.lower() or "human" == c.lower())]
        cols = [c for c in wanted if c in names] + inter_cols + [c for c in (action_col, state_col, success_col) if c]
        tbl = _read_parquet_dir(files, columns=cols)
        if tbl is None:
            return iter(())
        d = tbl.to_pydict()
        if action_col and action_col != "action":
            d["action"] = d.pop(action_col)
        if state_col and state_col != "observation.state":
            d["observation.state"] = d.pop(state_col)
        if success_col and success_col != "next.success":
            d["next.success"] = d.pop(success_col)
        ep_idx = np.asarray(d["episode_index"])
        order = np.argsort(ep_idx, kind="stable")
        uniq = np.unique(ep_idx)
        if max_episodes:
            uniq = uniq[:max_episodes]

        def col(name):
            return d.get(name)

        return self._iter_episodes(d, ep_idx, order, uniq, inter_cols)

    def _iter_episodes(self, d, ep_idx, order, uniq, inter_cols) -> Iterator[LeRobotEpisode]:
        ep_sorted = ep_idx[order]
        for e in uniq:
            lo = int(np.searchsorted(ep_sorted, e, side="left"))
            hi = int(np.searchsorted(ep_sorted, e, side="right"))
            rows = order[lo:hi]
            if "frame_index" in d:
                rows = rows[np.argsort(np.asarray(d["frame_index"])[rows], kind="stable")]
            ts = np.asarray(d["timestamp"], dtype=float)[rows] if "timestamp" in d else np.arange(len(rows)) / (self.fps or 30.0)
            actions = np.asarray([d["action"][i] for i in rows], dtype=float) if "action" in d else np.zeros((len(rows), 0))
            states = np.asarray([d["observation.state"][i] for i in rows], dtype=float) if "observation.state" in d else None
            task = None
            if "task" in d:
                task = str(d["task"][rows[0]])
            elif "task_index" in d:
                task = self.tasks.get(int(d["task_index"][rows[0]]), str(d["task_index"][rows[0]]))
            meta = self.episode_meta.get(int(e), {})
            if task is None:
                tasks_meta = meta.get("tasks")
                task = str(tasks_meta[0]) if isinstance(tasks_meta, list) and tasks_meta else "unknown"
            success: Optional[bool] = None
            reward_last: Optional[float] = None
            if "next.success" in d:
                vals = [d["next.success"][i] for i in rows]
                success = bool(any(bool(v) for v in vals if v is not None))
            if "next.reward" in d:
                rw = [d["next.reward"][i] for i in rows if d["next.reward"][i] is not None]
                reward_last = float(rw[-1]) if rw else None
                if success is None and reward_last is not None:
                    success = reward_last >= 1.0 - 1e-6
            mask = None
            if inter_cols:
                mask = np.zeros(len(rows), dtype=bool)
                for c in inter_cols:
                    mask |= np.asarray([bool(d[c][i]) if d[c][i] is not None else False for i in rows])
            yield LeRobotEpisode(int(e), task, ts, actions, states, success, reward_last, mask,
                                 extra={"length": len(rows), "meta": {k: v for k, v in meta.items() if k in ("length", "tasks")}})

    # ---- exports ----------------------------------------------------------------
    def to_records(self, policy: Optional[str] = None, max_episodes: Optional[int] = None) -> list[EpisodeRecord]:
        pol = policy or f"dataset:{self.repo_id}"
        return [ep.to_record(pol, self.repo_id, self.robot) for ep in self.episodes(max_episodes)]

    @staticmethod
    def excitation_arrays(ep: LeRobotEpisode) -> Optional[tuple[np.ndarray, dict, dict]]:
        """(t, cmd, meas) for the unit fingerprint when action and state have the same width."""
        if ep.states is None or ep.actions.shape[1] != ep.states.shape[1] or ep.actions.shape[0] < 20:
            return None
        t = ep.timestamps - ep.timestamps[0]
        cmd = {f"j{k}": ep.actions[:, k] for k in range(ep.actions.shape[1])}
        meas = {f"j{k}": ep.states[:, k] for k in range(ep.states.shape[1])}
        return t, cmd, meas
