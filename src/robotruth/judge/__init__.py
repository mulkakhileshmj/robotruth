"""Module 5: calibrated hybrid outcome judge with abstain.

Why: video-only VLM judges cap at 0.77 balanced accuracy and 0.52 on contact-rich tasks,
with a bias toward calling success (FailBench, arXiv 2609.03611). Cheap action-stream
signals give earlier, more robust failure warning (ActProbe, arXiv 2606.08508). Calibrated
success probabilities let a system hand uncertain cases to a human (VLAConf, arXiv
2605.29605). No paper reports false alarms per hour of robot time; this module does.

Pieces:
- features: EpisodeSignals and action-stream features, per episode and per window.
- vlm: VLMBackend protocol, MockBackend, AnthropicBackend (lazy anthropic import).
- calibrate: split-conformal Calibrator with abstain, TaskCalibrator, logistic FusionModel.
- judge: HybridJudge, JudgeResult (to Outcome), evaluate and JudgeMetrics with Wilson intervals.
- failbench: JSONL loader and run_failbench.
- cli: judge_app (typer), mounted by the main CLI.
"""

from .calibrate import ABSTAIN, FAILURE, SUCCESS, Calibrator, FusionModel, TaskCalibrator, load_any_calibrator
from .failbench import load_failbench_jsonl, parse_failbench_record, run_failbench
from .features import FEATURE_NAMES, EpisodeSignals, FeatureConfig, action_stream_features, feature_vector, windowed_features
from .judge import (
    FUSION_FEATURE_NAMES,
    HybridJudge,
    JudgeMetrics,
    JudgeResult,
    evaluate,
    fit_calibrator,
    fit_fusion,
    fusion_features,
    guess_failure_class,
    metrics_from_predictions,
)
from .vlm import DEFAULT_MODEL, AnthropicBackend, MockBackend, VLMBackend, VLMVerdict, parse_verdict, subsample_indices

__all__ = [
    "ABSTAIN",
    "FAILURE",
    "SUCCESS",
    "AnthropicBackend",
    "Calibrator",
    "DEFAULT_MODEL",
    "EpisodeSignals",
    "FEATURE_NAMES",
    "FUSION_FEATURE_NAMES",
    "FeatureConfig",
    "FusionModel",
    "HybridJudge",
    "JudgeMetrics",
    "JudgeResult",
    "MockBackend",
    "TaskCalibrator",
    "VLMBackend",
    "VLMVerdict",
    "action_stream_features",
    "evaluate",
    "feature_vector",
    "fit_calibrator",
    "fit_fusion",
    "fusion_features",
    "guess_failure_class",
    "load_any_calibrator",
    "load_failbench_jsonl",
    "metrics_from_predictions",
    "parse_failbench_record",
    "parse_verdict",
    "run_failbench",
    "subsample_indices",
    "windowed_features",
]
