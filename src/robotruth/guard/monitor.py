"""The Guard: a stateful per-episode runtime monitor.

One Guard wraps one policy. Call start_episode() when a rollout begins, step(...) after
every policy inference with whatever you have (the feature vector, the action chunk, the
measured state) and end_episode(truth) when it is over. Each step returns a GuardEvent
with a level:

- ok:        nothing to see.
- slow:      the score has been above the conformal threshold for `patience` consecutive
             steps. Slow the robot down, start recording, warn the operator.
- handover:  above threshold for another `patience` steps. Hand control to a human.
- stop:      a hard rule was violated (joint limit, workspace box, action magnitude).
             Immediate, no patience, latched until end_episode().

Hysteresis: once a level is raised, the score must fall below a release level
(threshold minus `hysteresis` times |threshold|) for `patience` consecutive steps before
the level drops one notch. Scores that hover around the threshold therefore hold the
current level instead of flapping between ok and slow every step.

The conformal threshold controls the probability of any false alarm on a nominal episode
at alpha. Patience only makes alarms rarer, never more frequent, so the guarantee holds at
the level of the Guard too. It costs `patience` steps of detection delay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from robotruth.guard.conformal import ContrastReport, ContrastSetCalibration, SequentialConformal
from robotruth.guard.metrics import LEVELS, GuardEpisode
from robotruth.guard.scores import CompositeScorer, Scorer, scorer_from_dict


@dataclass
class HardLimits:
    """Rules that stop the robot immediately when broken. All optional.

    joint_min, joint_max: per-dimension bounds, length D. Checked on the measured `state`
        when it is passed to step(), and on every row of the commanded action chunk when
        `check_actions_against_joints` is True (the default; set it False if actions are
        not in joint space).
    workspace_min, workspace_max: box on the Cartesian position, read from `position_dims`
        of each action row (default the first three dimensions).
    max_action_norm: largest allowed L2 norm of one action row.
    max_action_abs: largest allowed absolute value of any single action entry.
    max_step_norm: largest allowed L2 jump between consecutive action rows in a chunk.
    """

    joint_min: Optional[list[float]] = None
    joint_max: Optional[list[float]] = None
    workspace_min: Optional[list[float]] = None
    workspace_max: Optional[list[float]] = None
    position_dims: tuple[int, ...] = (0, 1, 2)
    max_action_norm: Optional[float] = None
    max_action_abs: Optional[float] = None
    max_step_norm: Optional[float] = None
    check_actions_against_joints: bool = True

    def violations(self, action_chunk: Optional[np.ndarray] = None,
                   state: Optional[np.ndarray] = None) -> list[str]:
        out: list[str] = []
        if state is not None and (self.joint_min is not None or self.joint_max is not None):
            s = np.asarray(state, dtype=np.float64).ravel()
            out += self._joint_check(s[None, :], "state")
        if action_chunk is not None:
            a = np.asarray(action_chunk, dtype=np.float64)
            if a.ndim == 1:
                a = a[None, :]
            if self.check_actions_against_joints and (self.joint_min is not None or self.joint_max is not None):
                out += self._joint_check(a, "action")
            if self.workspace_min is not None or self.workspace_max is not None:
                dims = [d for d in self.position_dims if d < a.shape[1]]
                if dims:
                    pos = a[:, dims]
                    if self.workspace_min is not None and np.any(pos < np.asarray(self.workspace_min)[: len(dims)]):
                        out.append("workspace: commanded position below workspace box")
                    if self.workspace_max is not None and np.any(pos > np.asarray(self.workspace_max)[: len(dims)]):
                        out.append("workspace: commanded position above workspace box")
            if self.max_action_norm is not None:
                worst = float(np.max(np.linalg.norm(a, axis=1)))
                if worst > self.max_action_norm:
                    out.append(f"action norm {worst:.3f} exceeds {self.max_action_norm:g}")
            if self.max_action_abs is not None:
                worst = float(np.max(np.abs(a)))
                if worst > self.max_action_abs:
                    out.append(f"action entry {worst:.3f} exceeds {self.max_action_abs:g}")
            if self.max_step_norm is not None and a.shape[0] > 1:
                worst = float(np.max(np.linalg.norm(np.diff(a, axis=0), axis=1)))
                if worst > self.max_step_norm:
                    out.append(f"within-chunk jump {worst:.3f} exceeds {self.max_step_norm:g}")
        return out

    def _joint_check(self, rows: np.ndarray, what: str) -> list[str]:
        out = []
        for name, bound, sign in (("min", self.joint_min, -1), ("max", self.joint_max, 1)):
            if bound is None:
                continue
            b = np.asarray(bound, dtype=np.float64)
            if b.size != rows.shape[1]:
                raise ValueError(f"joint_{name} has {b.size} entries but {what} has {rows.shape[1]} dims")
            bad = (rows < b) if sign < 0 else (rows > b)
            if np.any(bad):
                dims = sorted(set(np.where(bad)[1].tolist()))
                out.append(f"joint limit {name} violated on {what} dims {dims}")
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["position_dims"] = list(self.position_dims)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HardLimits":
        d = dict(d)
        if "position_dims" in d and d["position_dims"] is not None:
            d["position_dims"] = tuple(d["position_dims"])
        return cls(**d)


@dataclass
class GuardEvent:
    t: float
    step: int
    level: str                      # ok | slow | handover | stop
    score: Optional[float]
    threshold: Optional[float]
    reason: str
    components: dict[str, float] = field(default_factory=dict)
    changed: bool = False           # did the level change at this step

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Guard:
    """Stateful per-episode runtime monitor. See the module docstring for the level rules."""

    def __init__(self, scorer: Any, calibrator: Any, hard_limits: Optional[HardLimits] = None,
                 patience: int = 3, hysteresis: float = 0.2, dt: float = 1.0,
                 report: Optional[dict[str, Any]] = None) -> None:
        """scorer: anything with score_step(features, action_chunk) -> (score, components) and reset().
        calibrator: anything with threshold(step_index) -> float.
        patience: consecutive above-threshold steps per escalation (and per de-escalation).
        hysteresis: fraction of |threshold| the score must fall below the threshold to release.
        dt: seconds per step, used when step() is called without t."""
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if hysteresis < 0:
            raise ValueError("hysteresis must be non-negative")
        self.scorer = scorer
        self.calibrator = calibrator
        self.hard_limits = hard_limits
        self.patience = int(patience)
        self.hysteresis = float(hysteresis)
        self.dt = float(dt)
        self.report = report or {}
        self.episodes: list[GuardEpisode] = []
        self._episode_counter = 0
        self._in_episode = False
        self.start_episode()
        self._episode_counter = 0   # the constructor does not consume an episode id
        self._in_episode = False    # step() before start_episode() starts one

    # episode lifecycle -------------------------------------------------------------

    def start_episode(self, episode_id: Optional[str] = None) -> None:
        self._episode_counter += 1
        self._episode_id = episode_id or f"episode_{self._episode_counter:04d}"
        if hasattr(self.scorer, "reset"):
            self.scorer.reset()
        self._step = 0
        self._level = 0
        self._stopped = False
        self._above = 0
        self._below = 0
        self._t0: Optional[float] = None
        self._t_last: Optional[float] = None
        self._first_alert_t: Optional[float] = None
        self._max_level = 0
        self._n_onsets = 0
        self._max_score = float("-inf")
        self._max_excess = float("-inf")
        self.events: list[GuardEvent] = []
        self._in_episode = True

    @property
    def level(self) -> str:
        return "stop" if self._stopped else LEVELS[self._level]

    def _release_level(self, thr: float) -> float:
        return thr - self.hysteresis * abs(thr)

    def step(self, features: Optional[np.ndarray] = None, action_chunk: Optional[np.ndarray] = None,
             t: Optional[float] = None, state: Optional[np.ndarray] = None) -> GuardEvent:
        """Score one inference and return the current level."""
        if not self._in_episode:
            self.start_episode()
        step = self._step
        t_now = float(t) if t is not None else step * self.dt
        if self._t0 is None:
            self._t0 = t_now
        self._t_last = t_now
        prev_level = self.level

        # hard rules first: immediate stop, latched
        if self.hard_limits is not None:
            viol = self.hard_limits.violations(action_chunk=action_chunk, state=state)
            if viol:
                self._stopped = True
                self._max_level = 3
                ev = self._emit(t_now, step, None, None, "hard limit: " + "; ".join(viol), {}, prev_level)
                self._step += 1
                return ev
        if self._stopped:
            score, comps = self.scorer.score_step(features=features, action_chunk=action_chunk)
            ev = self._emit(t_now, step, score, None, "stopped (latched)", comps, prev_level)
            self._step += 1
            return ev

        score, comps = self.scorer.score_step(features=features, action_chunk=action_chunk)
        thr = float(self.calibrator.threshold(step))
        reason = "score within nominal range"
        if score is None:
            reason = "no scoreable input this step; level held"
        else:
            self._max_score = max(self._max_score, score)
            self._max_excess = max(self._max_excess, score - thr)
            if score > thr:
                self._above += 1
                self._below = 0
                if self._above >= self.patience and self._level < 2:
                    self._level += 1
                    self._above = 0
                    reason = f"score above threshold for {self.patience} consecutive steps"
                elif self._level > 0:
                    reason = "score above threshold; level held"
                else:
                    reason = f"score above threshold ({self._above}/{self.patience} steps)"
            elif score < self._release_level(thr):
                self._below += 1
                self._above = 0
                if self._below >= self.patience and self._level > 0:
                    self._level -= 1
                    self._below = 0
                    reason = f"score below release level for {self.patience} consecutive steps"
                elif self._level > 0:
                    reason = f"score below release level ({self._below}/{self.patience} steps)"
            else:
                self._above = 0
                self._below = 0
                if self._level > 0:
                    reason = "score in hysteresis band; level held"
        self._max_level = max(self._max_level, self._level)
        ev = self._emit(t_now, step, score, thr, reason, comps, prev_level)
        self._step += 1
        return ev

    def _emit(self, t_now: float, step: int, score, thr, reason: str, comps: dict, prev_level: str) -> GuardEvent:
        level = self.level
        changed = level != prev_level
        if changed and prev_level == "ok" and level != "ok":
            self._n_onsets += 1
            if self._first_alert_t is None:
                self._first_alert_t = t_now - (self._t0 or 0.0)
        ev = GuardEvent(t=t_now, step=step, level=level, score=score, threshold=thr, reason=reason,
                        components=dict(comps), changed=changed)
        self.events.append(ev)
        return ev

    def end_episode(self, truth: Optional[Any] = None, episode_id: Optional[str] = None) -> GuardEpisode:
        """Close the episode. truth: "success", "failure", True, False or None."""
        tv = _parse_truth(truth)
        if self._t0 is None or self._t_last is None:
            duration = 0.0
        else:
            span = self._t_last - self._t0
            step_dt = span / max(self._step - 1, 1) if self._step > 1 else self.dt
            duration = span + (step_dt if self._step > 0 else 0.0)
        rec = GuardEpisode(
            episode_id=episode_id or self._episode_id, truth=tv, duration_s=float(duration), n_steps=self._step,
            first_alert_t=self._first_alert_t, max_level="stop" if self._max_level == 3 else LEVELS[self._max_level],
            n_alert_onsets=self._n_onsets, stopped=self._stopped,
            max_score=float(self._max_score) if np.isfinite(self._max_score) else float("nan"),
            max_score_over_threshold=float(self._max_excess) if np.isfinite(self._max_excess) else float("nan"))
        self.episodes.append(rec)
        self._in_episode = False
        return rec

    # persistence -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        cal = self.calibrator.calibrator if isinstance(self.calibrator, ContrastSetCalibration) else self.calibrator
        return {"format": "robotruth.guard/0.1", "scorer": self.scorer.to_dict(), "calibrator": cal.to_dict(),
                "hard_limits": None if self.hard_limits is None else self.hard_limits.to_dict(),
                "patience": self.patience, "hysteresis": self.hysteresis, "dt": self.dt, "report": self.report}

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Guard":
        scorer = scorer_from_dict(d["scorer"])
        cal = SequentialConformal.from_dict(d["calibrator"])
        hl = HardLimits.from_dict(d["hard_limits"]) if d.get("hard_limits") else None
        return cls(scorer, cal, hl, patience=int(d.get("patience", 3)), hysteresis=float(d.get("hysteresis", 0.2)),
                   dt=float(d.get("dt", 1.0)), report=d.get("report") or {})

    @classmethod
    def load(cls, path: Path | str) -> "Guard":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        cal = self.calibrator.calibrator if isinstance(self.calibrator, ContrastSetCalibration) else self.calibrator
        s = cal.summary() if hasattr(cal, "summary") else "custom calibrator"
        return f"Guard: {s}; patience={self.patience}, hysteresis={self.hysteresis:g}, dt={self.dt:g}s"


def _parse_truth(truth: Any) -> Optional[bool]:
    if truth is None:
        return None
    if isinstance(truth, (bool, np.bool_)):
        return bool(truth)
    if isinstance(truth, (int, np.integer, float, np.floating)):
        return bool(truth)
    s = str(truth).strip().lower()
    if s in ("success", "succeeded", "ok", "pass", "1", "true"):
        return True
    if s in ("failure", "fail", "failed", "0", "false"):
        return False
    if s in ("", "none", "unknown", "null"):
        return None
    raise ValueError(f"cannot read truth label {truth!r}")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


# calibration from recorded episodes --------------------------------------------------

def calibrate_guard(nominal: Sequence[dict[str, Any]], benign: Optional[Sequence[dict[str, Any]]] = None,
                    alpha: float = 0.05, method: str = "max", n_bins: int = 4, split: float = 0.5,
                    patience: int = 3, hysteresis: float = 0.2, dt: float = 1.0,
                    hard_limits: Optional[HardLimits] = None, stride: Optional[int] = None,
                    weights: Optional[dict[str, float]] = None, scorer: Optional[Scorer] = None,
                    seed: int = 0) -> tuple[Guard, dict[str, Any]]:
    """Fit scorers and conformal thresholds from recorded nominal episodes.

    nominal: episode dicts with 'features' [T, F] and/or 'actions' [T, chunk, D] (and an
        optional 'timestamps' [T], unused here). All must be successful rollouts.
    benign: optional benign-shift episodes (SAFECAST contrast set). Thresholds are then
        calibrated on the union of held-out nominal and benign trajectories.
    split: fraction of nominal episodes used to fit the scorers; the rest calibrate the
        thresholds. Fitting and calibrating on the same episodes gives optimistic (too low)
        thresholds and more false alarms than alpha, so the split is on by default.

    Returns the Guard and a plain-dict report (also stored on guard.report).
    """
    eps = list(nominal)
    if len(eps) < 2:
        raise ValueError("need at least two nominal episodes")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eps))
    n_fit = int(round(split * len(eps)))
    n_fit = min(max(n_fit, 1), len(eps) - 1)
    fit_set = [eps[i] for i in order[:n_fit]]
    cal_set = [eps[i] for i in order[n_fit:]]
    has_f = all(e.get("features") is not None for e in eps)
    has_a = all(e.get("actions") is not None for e in eps)
    if scorer is None:
        scorer = CompositeScorer.default(has_features=has_f, has_actions=has_a, stride=stride)
        if weights:
            scorer.weights.update(weights)
    scorer.fit(fit_set)
    cal_trajs = [scorer.score_episode(e) for e in cal_set]
    horizon = max(t.size for t in cal_trajs)
    report: dict[str, Any] = {"n_nominal": len(eps), "n_fit": n_fit, "n_calibration": len(cal_set),
                              "alpha": alpha, "method": method, "split": split,
                              "inputs": [k for k, v in (("features", has_f), ("actions", has_a)) if v]}
    contrast: Optional[ContrastReport] = None
    if benign:
        ben_trajs = [scorer.score_episode(e) for e in benign]
        horizon = max(horizon, max(t.size for t in ben_trajs))
        csc = ContrastSetCalibration(alpha=alpha, method=method, n_bins=n_bins).fit(cal_trajs, ben_trajs, horizon=horizon)
        calibrator: Any = csc.calibrator
        contrast = csc.report
        report["contrast"] = contrast.to_dict()
        report["n_benign"] = len(benign)
    else:
        calibrator = SequentialConformal(alpha=alpha, method=method, n_bins=n_bins).fit(cal_trajs, horizon=horizon)
    report["calibrator"] = calibrator.to_dict()
    report["saturated"] = bool(calibrator.saturated_)
    if calibrator.saturated_:
        report["warning"] = (f"too few calibration episodes for alpha={alpha:g}; smallest certifiable alpha is "
                             f"{calibrator.achievable_alpha_:.3f}")
    guard = Guard(scorer, calibrator, hard_limits, patience=patience, hysteresis=hysteresis, dt=dt, report=report)
    return guard, report
