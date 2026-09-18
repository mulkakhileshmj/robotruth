"""The hybrid outcome judge and its evaluation metrics.

`HybridJudge` asks a video backend for a verdict, computes action-stream features, fuses
the two with a logistic `FusionModel` when one is supplied, and turns the fused score into
success, failure or abstain through a conformal `Calibrator`. Without a fusion model it is
a plain VLM judge with abstain; without a calibrator it thresholds at 0.5.

`evaluate` scores a judge against labelled episodes and reports what FailBench reports
(balanced accuracy, failure precision and recall, success bias) plus what no paper reports:
abstain rate, coverage and false alarms per hour of robot time. Every rate carries a Wilson
interval from `robotruth.stats.intervals`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from robotruth.schema.episode import FailureClass, FailureInfo, JudgeSource, Outcome
from robotruth.stats.intervals import Interval, wilson

from .calibrate import ABSTAIN, FAILURE, SUCCESS, Calibrator, FusionModel, TaskCalibrator, to_binary_labels
from .features import FEATURE_NAMES, EpisodeSignals, FeatureConfig, action_stream_features, feature_vector
from .vlm import VLMBackend, VLMVerdict

FUSION_FEATURE_NAMES: list[str] = ["vlm_success_prob", "vlm_progress", "vlm_parse_error"] + FEATURE_NAMES

AnyCalibrator = Union[Calibrator, TaskCalibrator]


def fusion_features(features: dict[str, float], verdict: Optional[VLMVerdict]) -> np.ndarray:
    """The fusion input vector: VLM outputs first, then the action-stream features."""
    if verdict is None:
        head = [0.5, 0.5, 1.0]
    else:
        head = [verdict.success_prob, verdict.progress, 1.0 if verdict.parse_error else 0.0]
    return np.concatenate([np.asarray(head, dtype=float), feature_vector(features)])


def guess_failure_class(features: dict[str, float], verdict: Optional[VLMVerdict]) -> tuple[Optional[str], Optional[str]]:
    """Best guess at (failure_class, subclass) from the VLM and the action signature.

    The VLM wins when it named a class. Otherwise the action stream gives an execution-level
    guess: long stagnation reads as a freeze, many gripper toggles as regrasping after a
    missed grasp, inconsistent chunks as an action-head problem.
    """
    if verdict is not None and verdict.failure_class:
        return verdict.failure_class, None
    if not features.get("has_actions", 0.0):
        return FailureClass.UNKNOWN.value, None
    if features.get("stagnation_fraction", 0.0) >= 0.5:
        return FailureClass.ACTION.value, "freeze"
    if features.get("gripper_toggles", 0.0) >= 4:
        return FailureClass.GRASP.value, "missed_grasp"
    if features.get("gripper_closed_no_motion_fraction", 0.0) >= 0.3:
        return FailureClass.GRASP.value, "unstable_grasp"
    if features.get("temporal_inconsistency", 0.0) > 0 and features.get("chunk_variance", 0.0) > 0 \
            and features["temporal_inconsistency"] > 2.0 * np.sqrt(features["chunk_variance"]):
        return FailureClass.ACTION.value, "inconsistent_chunks"
    return FailureClass.UNKNOWN.value, None


@dataclass
class JudgeResult:
    """What the judge concluded about one episode."""

    episode_id: str
    task: str
    success_prob: float
    decision: str
    failure_class: Optional[str] = None
    failure_subclass: Optional[str] = None
    features: dict[str, float] = field(default_factory=dict)
    vlm: Optional[VLMVerdict] = None
    fused: bool = False
    calibrated: bool = False
    judge_model: str = "unknown"

    @property
    def abstained(self) -> bool:
        return self.decision == ABSTAIN

    @property
    def confidence(self) -> float:
        return float(max(self.success_prob, 1.0 - self.success_prob))

    def to_outcome(self) -> Outcome:
        """Module 3 Outcome record. judged_by is hybrid when actions were fused in, else vlm."""
        judged_by = JudgeSource.HYBRID if self.fused else JudgeSource.VLM
        success = None if self.abstained else self.decision == SUCCESS
        score = None if self.vlm is None else float(np.clip(self.vlm.progress, 0.0, 1.0))
        notes = None
        if self.vlm is not None and self.vlm.rationale:
            notes = self.vlm.rationale[:500]
        return Outcome(success=success, score=score, judged_by=judged_by, judge_confidence=self.confidence,
                       judge_model=self.judge_model, abstained=self.abstained, notes=notes)

    def to_failure_info(self) -> Optional[FailureInfo]:
        if self.decision != FAILURE:
            return None
        fc = self.failure_class or FailureClass.UNKNOWN.value
        evidence = []
        for k in ("stagnation_fraction", "gripper_toggles", "temporal_inconsistency"):
            if k in self.features:
                evidence.append(f"{k}={self.features[k]:.3g}")
        return FailureInfo(failure_class=FailureClass(fc), subclass=self.failure_subclass, evidence=evidence)

    def to_dict(self) -> dict:
        return {"episode_id": self.episode_id, "task": self.task, "success_prob": self.success_prob, "decision": self.decision,
                "failure_class": self.failure_class, "failure_subclass": self.failure_subclass, "fused": self.fused,
                "calibrated": self.calibrated, "judge_model": self.judge_model, "features": self.features,
                "vlm": None if self.vlm is None else self.vlm.to_dict()}


class HybridJudge:
    """Video verdict plus action-stream features, fused and calibrated, with abstain.

    backend: any VLMBackend (None means actions only; then a fusion model is required).
    fusion: FusionModel over FUSION_FEATURE_NAMES. None means use the VLM probability.
    calibrator: Calibrator or TaskCalibrator. None means threshold at 0.5, abstaining only
    when the VLM reply could not be parsed and there is nothing else to go on.
    """

    def __init__(self, backend: Optional[VLMBackend], fusion: Optional[FusionModel] = None,
                 calibrator: Optional[AnyCalibrator] = None, feature_config: Optional[FeatureConfig] = None):
        if backend is None and fusion is None:
            raise ValueError("HybridJudge needs a VLM backend, a fusion model, or both")
        self.backend = backend
        self.fusion = fusion
        self.calibrator = calibrator
        self.feature_config = feature_config

    @property
    def judge_model(self) -> str:
        parts = []
        if self.backend is not None:
            parts.append(getattr(self.backend, "model", None) or getattr(self.backend, "name", "vlm"))
        if self.fusion is not None:
            parts.append("fusion")
        return "+".join(parts) if parts else "unknown"

    def score_episode(self, signals: EpisodeSignals) -> tuple[dict[str, float], Optional[VLMVerdict], float]:
        """Features, VLM verdict and the raw (uncalibrated) success probability."""
        features = action_stream_features(signals, self.feature_config)
        verdict = None
        if self.backend is not None:
            verdict = self.backend.judge(signals.frames, signals.instruction, signals.task)
        if self.fusion is not None:
            p = float(self.fusion.predict_proba(fusion_features(features, verdict)))
        elif verdict is not None:
            p = float(verdict.success_prob)
        else:
            p = 0.5
        return features, verdict, p

    def decide(self, p: float, task: str, verdict: Optional[VLMVerdict]) -> tuple[str, bool]:
        if self.calibrator is not None:
            if isinstance(self.calibrator, TaskCalibrator):
                return self.calibrator.decide(p, task), True
            return self.calibrator.decide(p), True
        if self.fusion is None and verdict is not None and verdict.parse_error:
            return ABSTAIN, False
        return (SUCCESS if p >= 0.5 else FAILURE), False

    def judge_episode(self, signals: EpisodeSignals) -> JudgeResult:
        features, verdict, p = self.score_episode(signals)
        decision, calibrated = self.decide(p, signals.task, verdict)
        fc, sub = (None, None)
        if decision == FAILURE:
            fc, sub = guess_failure_class(features, verdict)
        return JudgeResult(episode_id=signals.episode_id, task=signals.task, success_prob=p, decision=decision,
                           failure_class=fc, failure_subclass=sub, features=features, vlm=verdict,
                           fused=self.fusion is not None, calibrated=calibrated, judge_model=self.judge_model)


LabelledEpisodes = Iterable[tuple[EpisodeSignals, Union[str, bool, int]]]


def fit_fusion(backend: Optional[VLMBackend], episodes_with_labels: LabelledEpisodes, l2: float = 1.0,
               feature_config: Optional[FeatureConfig] = None) -> FusionModel:
    """Fit a FusionModel from labelled episodes, calling the backend once per episode."""
    X, y = [], []
    for signals, label in episodes_with_labels:
        features = action_stream_features(signals, feature_config)
        verdict = backend.judge(signals.frames, signals.instruction, signals.task) if backend is not None else None
        X.append(fusion_features(features, verdict))
        y.append(label)
    return FusionModel(feature_names=list(FUSION_FEATURE_NAMES), l2=l2).fit(np.vstack(X), y)


def fit_calibrator(judge: HybridJudge, episodes_with_labels: LabelledEpisodes, target_error: float = 0.05,
                   per_task: bool = False, **kw) -> AnyCalibrator:
    """Fit a Calibrator (or TaskCalibrator) on held-out labelled episodes and attach it to the judge."""
    scores, labels, tasks = [], [], []
    for signals, label in episodes_with_labels:
        _, _, p = judge.score_episode(signals)
        scores.append(p)
        labels.append(label)
        tasks.append(signals.task)
    cal: AnyCalibrator
    if per_task:
        cal = TaskCalibrator().fit(scores, labels, tasks, target_error=target_error, **kw)
    else:
        cal = Calibrator().fit(scores, labels, target_error=target_error, **kw)
    judge.calibrator = cal
    return cal


# ----------------------------------------------------------------------------- metrics

def _rate(k: int, n: int, alpha: float) -> Optional[Interval]:
    return wilson(k, n, alpha) if n > 0 else None


def _fmt(iv: Optional[Interval], scale: float = 1.0, unit: str = "") -> str:
    if iv is None:
        return "n/a"
    return f"{iv.estimate * scale:.3f}{unit} [{iv.lower * scale:.3f}, {iv.upper * scale:.3f}] (n={iv.n})"


@dataclass
class JudgeMetrics:
    """Judge quality on labelled episodes. Rates are Wilson intervals at level 1 - alpha."""

    n: int
    n_decided: int
    n_abstain: int
    hours: float
    abstain_rate: Optional[Interval]
    coverage: Optional[Interval]
    accuracy: Optional[Interval]
    success_recall: Optional[Interval]
    failure_recall: Optional[Interval]
    failure_precision: Optional[Interval]
    balanced_accuracy: Optional[float]
    balanced_accuracy_bounds: Optional[tuple[float, float]]
    success_bias: Optional[float]
    false_alarms: int
    false_alarm_rate: Optional[Interval]
    false_alarms_per_hour: Optional[float]
    false_alarms_per_hour_bounds: Optional[tuple[float, float]]
    per_task: dict[str, dict] = field(default_factory=dict)
    alpha: float = 0.05

    def to_dict(self) -> dict:
        def iv(x):
            return None if x is None else {"estimate": x.estimate, "lower": x.lower, "upper": x.upper, "n": x.n}
        return {"n": self.n, "n_decided": self.n_decided, "n_abstain": self.n_abstain, "hours": self.hours,
                "abstain_rate": iv(self.abstain_rate), "coverage": iv(self.coverage), "accuracy": iv(self.accuracy),
                "success_recall": iv(self.success_recall), "failure_recall": iv(self.failure_recall),
                "failure_precision": iv(self.failure_precision), "balanced_accuracy": self.balanced_accuracy,
                "balanced_accuracy_bounds": self.balanced_accuracy_bounds, "success_bias": self.success_bias,
                "false_alarms": self.false_alarms, "false_alarm_rate": iv(self.false_alarm_rate),
                "false_alarms_per_hour": self.false_alarms_per_hour, "false_alarms_per_hour_bounds": self.false_alarms_per_hour_bounds,
                "per_task": self.per_task, "alpha": self.alpha}

    def to_markdown(self, title: str = "Judge evaluation") -> str:
        pct = f"{100 * (1 - self.alpha):.0f}%"
        ba = "n/a" if self.balanced_accuracy is None else f"{self.balanced_accuracy:.3f}"
        if self.balanced_accuracy_bounds:
            ba += f" [{self.balanced_accuracy_bounds[0]:.3f}, {self.balanced_accuracy_bounds[1]:.3f}]"
        fah = "n/a" if self.false_alarms_per_hour is None else f"{self.false_alarms_per_hour:.2f}"
        if self.false_alarms_per_hour_bounds:
            fah += f" [{self.false_alarms_per_hour_bounds[0]:.2f}, {self.false_alarms_per_hour_bounds[1]:.2f}]"
        bias = "n/a" if self.success_bias is None else f"{self.success_bias:+.3f}"
        lines = [
            f"# {title}", "",
            f"Episodes: {self.n} ({self.n_decided} decided, {self.n_abstain} abstained), robot time {self.hours:.2f} h. Intervals are {pct} Wilson.", "",
            "| metric | value |", "|---|---|",
            f"| abstain rate | {_fmt(self.abstain_rate)} |",
            f"| coverage | {_fmt(self.coverage)} |",
            f"| accuracy on decided | {_fmt(self.accuracy)} |",
            f"| balanced accuracy on decided | {ba} |",
            f"| success recall (TPR) | {_fmt(self.success_recall)} |",
            f"| failure recall (TNR) | {_fmt(self.failure_recall)} |",
            f"| failure precision | {_fmt(self.failure_precision)} |",
            f"| success bias (predicted minus true success rate, decided) | {bias} |",
            f"| false alarms (failure called on a true success) | {self.false_alarms} |",
            f"| false alarm rate per true success | {_fmt(self.false_alarm_rate)} |",
            f"| false alarms per hour of robot time | {fah} |",
        ]
        if self.per_task:
            lines += ["", "## Per task", "", "| task | n | abstain | balanced accuracy | false alarms |", "|---|---|---|---|---|"]
            for t, d in sorted(self.per_task.items()):
                ba_t = "n/a" if d.get("balanced_accuracy") is None else f"{d['balanced_accuracy']:.3f}"
                lines.append(f"| {t} | {d['n']} | {d['n_abstain']} | {ba_t} | {d['false_alarms']} |")
        lines += ["", "Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.",
                  "Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks."]
        return "\n".join(lines) + "\n"


def metrics_from_predictions(decisions: Sequence[str], labels: Iterable, durations_s: Sequence[float],
                             tasks: Optional[Sequence[str]] = None, alpha: float = 0.05) -> JudgeMetrics:
    """Compute JudgeMetrics from decisions (success/failure/abstain), true labels and episode durations.

    false_alarms_per_hour divides the number of true successes the judge called failures by
    the total robot time of all evaluated episodes. Its bounds scale the Wilson interval of
    the per-episode false alarm rate by episodes per hour.
    """
    dec = np.asarray([str(d) for d in decisions])
    y = to_binary_labels(labels)
    dur = np.asarray(durations_s, dtype=float)
    n = int(dec.size)
    if y.size != n or dur.size != n:
        raise ValueError("decisions, labels and durations must have equal length")
    hours = float(np.nansum(dur)) / 3600.0
    decided = dec != ABSTAIN
    n_dec = int(decided.sum())
    n_abs = n - n_dec
    pred_s = dec == SUCCESS
    pred_f = dec == FAILURE
    true_s = y == 1
    true_f = y == 0
    tp = int((pred_s & true_s).sum())
    tn = int((pred_f & true_f).sum())
    fa = int((pred_f & true_s).sum())
    fp_s = int((pred_s & true_f).sum())
    n_ts_dec = int((true_s & decided).sum())
    n_tf_dec = int((true_f & decided).sum())
    tpr = _rate(tp, n_ts_dec, alpha)
    tnr = _rate(tn, n_tf_dec, alpha)
    ba = None
    ba_bounds = None
    if tpr is not None and tnr is not None:
        ba = 0.5 * (tpr.estimate + tnr.estimate)
        ba_bounds = (0.5 * (tpr.lower + tnr.lower), 0.5 * (tpr.upper + tnr.upper))
    bias = None
    if n_dec:
        bias = float(pred_s[decided].mean() - true_s[decided].mean())
    fa_rate = _rate(fa, int(true_s.sum()), alpha)
    fah = fa / hours if hours > 0 else None
    fah_bounds = None
    if fa_rate is not None and hours > 0:
        scale = int(true_s.sum()) / hours
        fah_bounds = (fa_rate.lower * scale, fa_rate.upper * scale)
    per_task: dict[str, dict] = {}
    if tasks is not None:
        tasks_arr = np.asarray([str(t) for t in tasks])
        for t in sorted(set(tasks_arr.tolist())):
            m = tasks_arr == t
            sub = metrics_from_predictions(dec[m], y[m], dur[m], None, alpha)
            per_task[t] = {"n": sub.n, "n_abstain": sub.n_abstain, "balanced_accuracy": sub.balanced_accuracy,
                           "false_alarms": sub.false_alarms, "success_bias": sub.success_bias}
    return JudgeMetrics(
        n=n, n_decided=n_dec, n_abstain=n_abs, hours=hours,
        abstain_rate=_rate(n_abs, n, alpha), coverage=_rate(n_dec, n, alpha),
        accuracy=_rate(tp + tn, n_dec, alpha), success_recall=tpr, failure_recall=tnr,
        failure_precision=_rate(tn, tn + fa, alpha), balanced_accuracy=ba, balanced_accuracy_bounds=ba_bounds,
        success_bias=bias, false_alarms=fa, false_alarm_rate=fa_rate, false_alarms_per_hour=fah,
        false_alarms_per_hour_bounds=fah_bounds, per_task=per_task, alpha=alpha,
    )


def evaluate(judge: HybridJudge, episodes_with_labels: LabelledEpisodes, alpha: float = 0.05,
             return_results: bool = False):
    """Run the judge on labelled episodes and compute JudgeMetrics.

    With return_results=True, returns (metrics, list[JudgeResult]).
    """
    decisions, labels, durations, tasks, results = [], [], [], [], []
    for signals, label in episodes_with_labels:
        r = judge.judge_episode(signals)
        results.append(r)
        decisions.append(r.decision)
        labels.append(label)
        durations.append(signals.duration())
        tasks.append(signals.task or "unknown")
    m = metrics_from_predictions(decisions, labels, durations, tasks, alpha)
    return (m, results) if return_results else m
