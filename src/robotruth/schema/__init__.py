"""Module 3: the episode record.

The gap, named by the ISO/TC 299 WG 16 convenor's group ("Data Standards for Humanoid
Robotics", arXiv 2606.19769): no log format carries provenance, calibration, policy
version, execution trace and outcome together. Without it you cannot answer why a deployed
robot failed, and you cannot turn interventions into training signal (RECAP, pi*0.6).

This module defines one record per episode, a failure taxonomy, a JSONL log, converters
to the results table module 2 consumes, an MCAP bridge, and fleet metrics.
"""

from robotruth.schema.episode import (
    EpisodeRecord,
    FailureClass,
    FailureInfo,
    Intervention,
    InterventionSource,
    JudgeSource,
    Outcome,
    Provenance,
    Timing,
    FAILURE_SUBCLASSES,
)
from robotruth.schema.log import EpisodeLog
from robotruth.schema.metrics import FleetMetrics, fleet_metrics

__all__ = [
    "EpisodeLog",
    "EpisodeRecord",
    "FAILURE_SUBCLASSES",
    "FailureClass",
    "FailureInfo",
    "FleetMetrics",
    "Intervention",
    "InterventionSource",
    "JudgeSource",
    "Outcome",
    "Provenance",
    "Timing",
    "fleet_metrics",
]
