"""robotruth.guard: a drop-in runtime failure monitor for learned robot policies.

The recipe exists on paper and nobody packages it. FAIL-Detect (RSS 2025, arXiv
2503.08558) frames runtime failure detection as sequential out-of-distribution detection
with conformal thresholds calibrated on successful rollouts only. VLA-FAIL (arXiv
2606.21386) gets there at near-zero overhead with last-layer Mahalanobis distance plus
action-chunk consistency. SAFECAST (arXiv 2608.04246) calibrates on contrast sets so the
detector does not fire on harmless change. None of them report false alarms per hour. This
module does.

Pieces:

- scores: MahalanobisScorer, ChunkConsistencyScorer, ActionStatsScorer, StagnationScorer,
  CompositeScorer.
- conformal: SequentialConformal (time-uniform thresholds, "max" or "bonferroni"),
  ContrastSetCalibration.
- monitor: Guard (ok, slow, handover, stop with patience and hysteresis), HardLimits,
  GuardEvent, calibrate_guard.
- metrics: GuardMetrics (detection rate with Wilson interval, warning time, false alarms
  per hour with Poisson interval).
- adapters: PolicyServerHook, attach, OfflineReplay, episode npz helpers.
- cli: guard_app (calibrate, replay, metrics).

Minimal use:

    guard, report = calibrate_guard(nominal_episodes, alpha=0.05)
    guard.start_episode()
    for features, chunk in policy_loop():
        event = guard.step(features=features, action_chunk=chunk)
        if event.level != "ok": ...
    guard.end_episode("success")
    print(GuardMetrics.from_episodes(guard.episodes).to_markdown())
"""

from robotruth.guard.adapters import (
    OfflineReplay,
    PolicyServerHook,
    ReplayResult,
    attach,
    load_episode_npz,
    save_episode_npz,
)
from robotruth.guard.cli import guard_app
from robotruth.guard.conformal import (
    ContrastReport,
    ContrastSetCalibration,
    SequentialConformal,
    conformal_quantile,
)
from robotruth.guard.metrics import GuardEpisode, GuardMetrics, poisson_rate_interval
from robotruth.guard.monitor import (Guard, GuardEvent, HardLimits, MultiHeadScorer, ZeroThreshold,
                                     calibrate_guard)
from robotruth.guard.scores import (
    ActionStatsScorer,
    ChunkConsistencyScorer,
    CompositeScorer,
    MahalanobisScorer,
    Scorer,
    StagnationScorer,
    ledoit_wolf_covariance,
    scorer_from_dict,
)

__all__ = [
    "ActionStatsScorer",
    "ChunkConsistencyScorer",
    "CompositeScorer",
    "ContrastReport",
    "ContrastSetCalibration",
    "Guard",
    "GuardEpisode",
    "GuardEvent",
    "GuardMetrics",
    "HardLimits",
    "MahalanobisScorer",
    "MultiHeadScorer",
    "OfflineReplay",
    "PolicyServerHook",
    "ReplayResult",
    "Scorer",
    "SequentialConformal",
    "StagnationScorer",
    "ZeroThreshold",
    "attach",
    "calibrate_guard",
    "conformal_quantile",
    "guard_app",
    "ledoit_wolf_covariance",
    "load_episode_npz",
    "poisson_rate_interval",
    "save_episode_npz",
    "scorer_from_dict",
]
