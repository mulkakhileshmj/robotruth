"""Calibration for the outcome judge: conformal abstain thresholds and a logistic fusion model.

Two pieces, both numpy only.

`Calibrator` turns a raw success score into a decision in {success, failure, abstain}
using split conformal prediction with class-conditional (Mondrian) quantiles. Given
calibration episodes with true labels, it computes for each class the nonconformity
quantile that covers that class with probability at least 1 - target_error. An episode is
decided only when exactly one class fits; if both fit or neither fits, the judge abstains.
This is the VLAConf idea (arXiv 2605.29605) applied to a judge instead of a policy. By
default a second pass shrinks the working error level until the empirical error among the
decided calibration episodes is at or below the target, since the conformal guarantee is
on total miscoverage while labs care about the error rate of the decisions they act on.

`FusionModel` is L2-regularised logistic regression fit by Newton steps. It combines the
VLM success probability with the action-stream features from `features.py`. Inputs are
standardised inside the model so raw feature scales do not matter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

SUCCESS = "success"
FAILURE = "failure"
ABSTAIN = "abstain"


def to_binary_labels(labels: Iterable) -> np.ndarray:
    """Accept bools, 0/1 ints, or the strings success/failure. Returns int array of 1 (success) / 0."""
    out = []
    for v in labels:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("success", "succeeded", "1", "true", "pass"):
                out.append(1)
            elif s in ("failure", "fail", "failed", "0", "false"):
                out.append(0)
            else:
                raise ValueError(f"unrecognised label {v!r}")
        else:
            out.append(1 if bool(v) else 0)
    return np.asarray(out, dtype=int)


def conformal_quantile(nonconformity: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile: the ceil((n + 1)(1 - alpha))-th order statistic.

    Returns 1.0 (the largest possible nonconformity for probabilities) when n is too small
    for the requested level, which makes the class always fit and forces abstention rather
    than a false guarantee.
    """
    nc = np.sort(np.asarray(nonconformity, dtype=float))
    n = nc.size
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return 1.0
    return float(nc[max(k, 1) - 1])


@dataclass
class Calibrator:
    """Split-conformal abstain thresholds on a success score in [0, 1]."""

    t_fail: Optional[float] = None
    """Failure fits when score <= t_fail."""
    t_succ: Optional[float] = None
    """Success fits when score >= t_succ."""
    target_error: float = 0.05
    alpha_used: float = 0.05
    n_cal: int = 0
    n_success: int = 0
    n_failure: int = 0
    cal_report: dict = field(default_factory=dict)
    abstain_on_empty: bool = False
    """What to do when the score fits neither class (an empty conformal set). Default False:
    decide the nearer class, since the truth is outside the set either way and those events
    are already inside the miscoverage budget. True: abstain there as well, which is stricter
    but makes the abstain rate non-monotone in the target error."""

    @property
    def fitted(self) -> bool:
        return self.t_fail is not None and self.t_succ is not None

    @staticmethod
    def thresholds(scores: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float]:
        nc_success = 1.0 - scores[y == 1]
        nc_failure = scores[y == 0]
        q_s = conformal_quantile(nc_success, alpha)
        q_f = conformal_quantile(nc_failure, alpha)
        return float(q_f), float(1.0 - q_s)

    @staticmethod
    def decide_with(score: float, t_fail: float, t_succ: float, abstain_on_empty: bool = False) -> str:
        """Success fits when score >= t_succ, failure when score <= t_fail. Exactly one fit
        decides; both fitting (the overlap band) abstains; neither fitting is resolved to the
        nearer class unless abstain_on_empty."""
        s = float(score)
        succ_fits = s >= t_succ
        fail_fits = s <= t_fail
        if succ_fits and not fail_fits:
            return SUCCESS
        if fail_fits and not succ_fits:
            return FAILURE
        if succ_fits and fail_fits:
            return ABSTAIN
        if abstain_on_empty:
            return ABSTAIN
        return SUCCESS if s >= 0.5 * (t_fail + t_succ) else FAILURE

    def fit(self, scores: Sequence[float], labels: Iterable, target_error: float = 0.05,
            min_per_class: int = 5, enforce_selective: bool = True, shrink: float = 0.8) -> "Calibrator":
        s = np.asarray(scores, dtype=float)
        y = to_binary_labels(labels)
        if s.shape[0] != y.shape[0]:
            raise ValueError("scores and labels must have the same length")
        if not (0 < target_error < 1):
            raise ValueError("target_error must be in (0, 1)")
        n_s, n_f = int((y == 1).sum()), int((y == 0).sum())
        if n_s < min_per_class or n_f < min_per_class:
            raise ValueError(f"need at least {min_per_class} calibration episodes per class, got {n_s} successes and {n_f} failures")
        alpha = target_error
        t_fail, t_succ = self.thresholds(s, y, alpha)
        for _ in range(60):
            t_fail, t_succ = self.thresholds(s, y, alpha)
            dec = np.asarray([self.decide_with(v, t_fail, t_succ, self.abstain_on_empty) for v in s])
            decided = dec != ABSTAIN
            n_dec = int(decided.sum())
            errors = int(((dec == SUCCESS) & (y == 0)).sum() + ((dec == FAILURE) & (y == 1)).sum())
            sel_err = errors / n_dec if n_dec else 0.0
            if not enforce_selective or sel_err <= target_error or alpha < 1e-4:
                break
            alpha *= shrink
        self.t_fail, self.t_succ = t_fail, t_succ
        self.target_error = float(target_error)
        self.alpha_used = float(alpha)
        self.n_cal, self.n_success, self.n_failure = int(s.size), n_s, n_f
        self.cal_report = {"selective_error": float(sel_err), "abstain_rate": float(1 - n_dec / s.size), "n_decided": n_dec, "errors": errors}
        return self

    def decide(self, score: float) -> str:
        """Return success, failure or abstain for one raw score."""
        if not self.fitted:
            raise RuntimeError("Calibrator is not fitted")
        return self.decide_with(score, self.t_fail, self.t_succ, self.abstain_on_empty)

    def decide_many(self, scores: Sequence[float]) -> list[str]:
        return [self.decide(v) for v in scores]

    def to_dict(self) -> dict:
        return {"t_fail": self.t_fail, "t_succ": self.t_succ, "target_error": self.target_error, "alpha_used": self.alpha_used,
                "n_cal": self.n_cal, "n_success": self.n_success, "n_failure": self.n_failure, "cal_report": self.cal_report,
                "abstain_on_empty": self.abstain_on_empty}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(t_fail=d.get("t_fail"), t_succ=d.get("t_succ"), target_error=d.get("target_error", 0.05),
                   alpha_used=d.get("alpha_used", d.get("target_error", 0.05)), n_cal=d.get("n_cal", 0),
                   n_success=d.get("n_success", 0), n_failure=d.get("n_failure", 0), cal_report=d.get("cal_report", {}),
                   abstain_on_empty=bool(d.get("abstain_on_empty", False)))

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.write_text(json.dumps({"kind": "robotruth.judge.Calibrator", **self.to_dict()}, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> "Calibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_TASK = "__default__"


@dataclass
class TaskCalibrator:
    """Per-task calibrators with a global fallback. Contact-rich tasks need looser thresholds
    than pick-and-place; FailBench shows the judge error rate differs by task family."""

    calibrators: dict[str, Calibrator] = field(default_factory=dict)
    target_error: float = 0.05
    min_per_task: int = 30

    def fit(self, scores: Sequence[float], labels: Iterable, tasks: Optional[Sequence[str]] = None,
            target_error: float = 0.05, min_per_task: int = 30, min_per_class: int = 5, **kw) -> "TaskCalibrator":
        s = np.asarray(scores, dtype=float)
        y = to_binary_labels(labels)
        self.target_error, self.min_per_task = float(target_error), int(min_per_task)
        self.calibrators = {DEFAULT_TASK: Calibrator().fit(s, y, target_error, min_per_class=min_per_class, **kw)}
        if tasks is None:
            return self
        tasks = np.asarray([str(t) for t in tasks])
        for task in sorted(set(tasks.tolist())):
            m = tasks == task
            if m.sum() < min_per_task:
                continue
            if (y[m] == 1).sum() < min_per_class or (y[m] == 0).sum() < min_per_class:
                continue
            self.calibrators[task] = Calibrator().fit(s[m], y[m], target_error, min_per_class=min_per_class, **kw)
        return self

    def for_task(self, task: Optional[str]) -> Calibrator:
        if task is not None and task in self.calibrators:
            return self.calibrators[task]
        if DEFAULT_TASK not in self.calibrators:
            raise RuntimeError("TaskCalibrator is not fitted")
        return self.calibrators[DEFAULT_TASK]

    def decide(self, score: float, task: Optional[str] = None) -> str:
        return self.for_task(task).decide(score)

    @property
    def fitted(self) -> bool:
        return DEFAULT_TASK in self.calibrators and self.calibrators[DEFAULT_TASK].fitted

    def to_dict(self) -> dict:
        return {"kind": "robotruth.judge.TaskCalibrator", "target_error": self.target_error, "min_per_task": self.min_per_task,
                "calibrators": {k: v.to_dict() for k, v in self.calibrators.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskCalibrator":
        return cls(calibrators={k: Calibrator.from_dict(v) for k, v in d.get("calibrators", {}).items()},
                   target_error=d.get("target_error", 0.05), min_per_task=d.get("min_per_task", 30))

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> "TaskCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_any_calibrator(path: Path | str) -> "Calibrator | TaskCalibrator":
    """Load whichever calibrator kind the JSON holds."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("kind") == "robotruth.judge.TaskCalibrator" or "calibrators" in d:
        return TaskCalibrator.from_dict(d)
    return Calibrator.from_dict(d)


# ----------------------------------------------------------------------------- fusion

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class FusionModel:
    """L2-regularised logistic regression on [vlm_success_prob, action features...].

    Fit by Newton's method (the problem is small and convex). Features are standardised
    with the training mean and standard deviation, stored with the model. The intercept is
    not regularised.
    """

    feature_names: list[str] = field(default_factory=list)
    l2: float = 1.0
    coef: Optional[np.ndarray] = None
    intercept: float = 0.0
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    n_train: int = 0
    train_report: dict = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return self.coef is not None

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit(self, X: np.ndarray, y: Iterable, max_iter: int = 100, tol: float = 1e-8) -> "FusionModel":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D [n, d]")
        yb = to_binary_labels(y).astype(float)
        n, d = X.shape
        if yb.shape[0] != n:
            raise ValueError("X and y length mismatch")
        if not self.feature_names:
            self.feature_names = [f"f{i}" for i in range(d)]
        elif len(self.feature_names) != d:
            raise ValueError(f"feature_names has {len(self.feature_names)} entries but X has {d} columns")
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0
        self.std = std
        Z = np.hstack([self._standardise(X), np.ones((n, 1))])
        w = np.zeros(d + 1)
        reg = np.full(d + 1, float(self.l2))
        reg[-1] = 1e-8
        for it in range(max_iter):
            p = _sigmoid(Z @ w)
            g = Z.T @ (p - yb) + reg * w
            wgt = p * (1 - p)
            H = (Z * wgt[:, None]).T @ Z + np.diag(reg)
            step = np.linalg.solve(H, g)
            w -= step
            if float(np.max(np.abs(step))) < tol:
                break
        self.coef = w[:-1]
        self.intercept = float(w[-1])
        self.n_train = n
        p = self.predict_proba(X)
        acc = float(((p >= 0.5).astype(int) == yb.astype(int)).mean())
        eps = 1e-12
        ll = float(-np.mean(yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps)))
        self.train_report = {"iterations": it + 1, "train_accuracy": acc, "train_log_loss": ll}
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FusionModel is not fitted")
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        z = self._standardise(X) @ self.coef + self.intercept
        return z[0] if single else z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of success for each row (or a float for a single vector)."""
        z = self.decision_function(X)
        return float(_sigmoid(np.asarray(z))) if np.ndim(z) == 0 else _sigmoid(z)

    def coefficients(self) -> dict[str, float]:
        """Standardised coefficients by feature name, largest magnitude first."""
        if not self.fitted:
            return {}
        pairs = sorted(zip(self.feature_names, self.coef.tolist()), key=lambda kv: -abs(kv[1]))
        return dict(pairs)

    def to_dict(self) -> dict:
        return {"kind": "robotruth.judge.FusionModel", "feature_names": list(self.feature_names), "l2": self.l2,
                "coef": None if self.coef is None else self.coef.tolist(), "intercept": self.intercept,
                "mean": None if self.mean is None else self.mean.tolist(), "std": None if self.std is None else self.std.tolist(),
                "n_train": self.n_train, "train_report": self.train_report}

    @classmethod
    def from_dict(cls, d: dict) -> "FusionModel":
        m = cls(feature_names=list(d.get("feature_names", [])), l2=float(d.get("l2", 1.0)))
        m.coef = None if d.get("coef") is None else np.asarray(d["coef"], dtype=float)
        m.intercept = float(d.get("intercept", 0.0))
        m.mean = None if d.get("mean") is None else np.asarray(d["mean"], dtype=float)
        m.std = None if d.get("std") is None else np.asarray(d["std"], dtype=float)
        m.n_train = int(d.get("n_train", 0))
        m.train_report = dict(d.get("train_report", {}))
        return m

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> "FusionModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
