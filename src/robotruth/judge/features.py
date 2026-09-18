"""Cheap action-stream features for outcome judging and early warning.

Why these exist: FailBench (arXiv 2609.03611) shows video-only VLM judges top out at
0.77 balanced accuracy and sit at 0.52 on contact-rich tasks, with a bias toward calling
success. ActProbe (arXiv 2606.08508) shows that action-chunk variance and magnitude alone
give earlier and more robust failure warning than semantic judges. This module computes
that kind of signal from the policy's own actions and the robot's states, per episode and
per sliding window. No model, no GPU, microseconds per episode.

Conventions: `actions` is [T, D] (one action per step) or [T, C, D] (a chunk of C future
actions predicted at each step, as ACT, pi0 and SmolVLA emit). The executed action is the
first element of each chunk. `states` is optional [T, Ds] proprioception; when present it is
the preferred motion source, since actions can be deltas or targets while states are what
the robot actually did.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import numpy as np

FEATURE_NAMES: list[str] = [
    "has_actions",
    "duration_s",
    "n_steps",
    "control_hz",
    "action_mag_mean",
    "action_mag_max",
    "chunk_variance",
    "temporal_inconsistency",
    "motion_mean",
    "stagnation_fraction",
    "longest_stagnation_s",
    "gripper_toggles",
    "gripper_toggles_per_min",
    "gripper_closed_no_motion_fraction",
    "final_window_motion",
    "final_window_motion_ratio",
]


@dataclass
class FeatureConfig:
    """Knobs for the feature extractor. Defaults are scale-free where possible."""

    stagnation_rel: float = 0.05
    """A step is stagnant when its motion is below this fraction of the 95th percentile motion."""
    stagnation_abs: float = 1e-4
    """Floor for the stagnation threshold, so a fully frozen episode still counts as frozen."""
    final_window_s: float = 1.0
    """Length of the final window used for the end-of-episode motion features."""
    gripper_closed_is_high: bool = False
    """If True, a gripper value above the midpoint means closed. Gripper conventions differ by robot."""


@dataclass
class EpisodeSignals:
    """Everything the judge may look at for one episode.

    `frames` may be image arrays, file paths or raw encoded bytes. They are only used by
    the VLM backend and can be None when judging from actions alone.
    """

    actions: Optional[np.ndarray]
    timestamps: Optional[np.ndarray]
    instruction: str = ""
    task: str = ""
    episode_id: str = ""
    states: Optional[np.ndarray] = None
    gripper: Optional[np.ndarray] = None
    frames: Optional[Sequence[Any]] = None
    duration_s: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        if self.actions is not None:
            return int(np.asarray(self.actions).shape[0])
        if self.timestamps is not None:
            return int(np.asarray(self.timestamps).shape[0])
        return 0

    def duration(self) -> float:
        """Episode length in seconds, from timestamps when present, else the stated duration."""
        if self.timestamps is not None and len(self.timestamps) >= 2:
            ts = np.asarray(self.timestamps, dtype=float)
            return float(ts[-1] - ts[0])
        return float(self.duration_s or 0.0)

    def slice(self, idx: np.ndarray) -> "EpisodeSignals":
        """Return a copy restricted to the given step indices (or boolean mask)."""
        idx = np.asarray(idx)

        def take(x):
            return None if x is None else np.asarray(x)[idx]

        return replace(
            self,
            actions=take(self.actions),
            timestamps=take(self.timestamps),
            states=take(self.states),
            gripper=take(self.gripper),
            frames=None,
            duration_s=None,
        )


def _as_chunks(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=float)
    if a.ndim == 2:
        a = a[:, None, :]
    if a.ndim != 3:
        raise ValueError(f"actions must be [T, D] or [T, C, D], got shape {a.shape}")
    return a


def _step_motion(signals: EpisodeSignals, executed: np.ndarray) -> np.ndarray:
    """Per-step motion magnitude, length T. Uses states when available, else executed actions."""
    T = executed.shape[0]
    src = executed
    if signals.states is not None:
        st = np.asarray(signals.states, dtype=float)
        if st.ndim == 2 and st.shape[0] == T:
            src = st
    if T < 2:
        return np.zeros(T)
    d = np.linalg.norm(np.diff(src, axis=0), axis=1)
    return np.concatenate([[d[0]], d])


def _empty_features(signals: EpisodeSignals) -> dict[str, float]:
    out = {k: 0.0 for k in FEATURE_NAMES}
    out["duration_s"] = signals.duration()
    return out


def action_stream_features(signals: EpisodeSignals, config: Optional[FeatureConfig] = None) -> dict[str, float]:
    """Per-episode action-stream features. Returns a dict with the keys in FEATURE_NAMES.

    When the episode has no actions, every feature is 0.0 and `has_actions` is 0.0, so a
    fusion model can learn to lean on the VLM alone for such episodes.
    """
    cfg = config or FeatureConfig()
    if signals.actions is None or np.asarray(signals.actions).shape[0] == 0:
        return _empty_features(signals)

    a = _as_chunks(signals.actions)
    T, C, D = a.shape
    if signals.timestamps is not None and len(signals.timestamps) == T:
        ts = np.asarray(signals.timestamps, dtype=float)
    else:
        ts = np.arange(T, dtype=float)
    duration = float(ts[-1] - ts[0]) if T >= 2 else 0.0
    dt = duration / (T - 1) if T >= 2 and duration > 0 else 1.0
    hz = 1.0 / dt if dt > 0 else 0.0

    executed = a[:, 0, :]
    mags = np.linalg.norm(executed, axis=1)
    chunk_var = float(a.var(axis=1).sum(axis=-1).mean()) if C > 1 else 0.0
    if C > 1 and T > 1:
        overlap = a[:-1, 1:, :] - a[1:, :-1, :]
        temporal_inc = float(np.linalg.norm(overlap, axis=-1).mean())
    else:
        temporal_inc = 0.0

    motion = _step_motion(signals, executed)
    p95 = float(np.percentile(motion, 95)) if motion.size else 0.0
    thr = max(cfg.stagnation_abs, cfg.stagnation_rel * p95)
    stagnant = motion < thr
    stagnation_fraction = float(stagnant.mean()) if T else 0.0
    longest = 0
    run = 0
    for s in stagnant:
        run = run + 1 if s else 0
        longest = max(longest, run)
    longest_s = longest * dt

    toggles = 0
    closed_no_motion = 0.0
    if signals.gripper is not None and len(signals.gripper) == T:
        g = np.asarray(signals.gripper, dtype=float).reshape(T, -1)[:, 0]
        lo, hi = float(g.min()), float(g.max())
        if hi - lo > 1e-9:
            mid = 0.5 * (lo + hi)
            closed = g > mid if cfg.gripper_closed_is_high else g < mid
            toggles = int(np.count_nonzero(closed[1:] != closed[:-1]))
            closed_no_motion = float(np.mean(closed & stagnant))
    minutes = duration / 60.0 if duration > 0 else 0.0
    toggles_per_min = toggles / minutes if minutes > 0 else float(toggles)

    final_mask = ts >= ts[-1] - cfg.final_window_s
    final_motion = float(motion[final_mask].mean()) if final_mask.any() else 0.0
    motion_mean = float(motion.mean()) if motion.size else 0.0
    ratio = final_motion / (motion_mean + 1e-12) if motion_mean > 0 else 0.0

    return {
        "has_actions": 1.0,
        "duration_s": duration,
        "n_steps": float(T),
        "control_hz": float(hz),
        "action_mag_mean": float(mags.mean()),
        "action_mag_max": float(mags.max()),
        "chunk_variance": chunk_var,
        "temporal_inconsistency": temporal_inc,
        "motion_mean": motion_mean,
        "stagnation_fraction": stagnation_fraction,
        "longest_stagnation_s": float(longest_s),
        "gripper_toggles": float(toggles),
        "gripper_toggles_per_min": float(toggles_per_min),
        "gripper_closed_no_motion_fraction": closed_no_motion,
        "final_window_motion": final_motion,
        "final_window_motion_ratio": float(ratio),
    }


def feature_vector(features: dict[str, float], names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """Order a feature dict into a fixed vector. Missing keys become 0.0."""
    return np.asarray([float(features.get(k, 0.0)) for k in names], dtype=float)


def windowed_features(signals: EpisodeSignals, window_s: float, stride_s: Optional[float] = None,
                      config: Optional[FeatureConfig] = None) -> list[dict[str, float]]:
    """Sliding-window time series of the same features, for early warning.

    Each entry carries `t_start` and `t_end` (seconds, same clock as `timestamps`) plus the
    feature values computed on the steps inside that window. Windows advance by `stride_s`
    (default: half the window). The final window always ends at the last timestamp.
    """
    if signals.actions is None or signals.timestamps is None:
        return []
    ts = np.asarray(signals.timestamps, dtype=float)
    T = ts.shape[0]
    if T == 0 or window_s <= 0:
        return []
    stride = stride_s if stride_s and stride_s > 0 else window_s / 2.0
    t0, t_last = float(ts[0]), float(ts[-1])
    ends: list[float] = []
    t_end = t0 + window_s
    while t_end < t_last:
        ends.append(t_end)
        t_end += stride
    ends.append(t_last)
    out: list[dict[str, float]] = []
    for t_end in ends:
        mask = (ts > t_end - window_s) & (ts <= t_end)
        if mask.sum() < 2:
            continue
        sub = signals.slice(mask)
        feats = action_stream_features(sub, config)
        feats["t_start"] = float(t_end - window_s)
        feats["t_end"] = float(t_end)
        out.append(feats)
    return out
