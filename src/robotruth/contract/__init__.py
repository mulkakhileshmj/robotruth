"""Module 1: executable-policy contract checker.

A checkpoint is not a policy. The thing that runs on a robot is the tuple of weights,
normalization statistics, action semantics, control rate, chunking, preprocessing, camera
calibration and embodiment. Swap any one and the same weights become a different policy
("Same Weights, Different Robot", arXiv 2606.03724: 28/28 to 2/28).

This module extracts that tuple into an `ExecSpec` manifest and compares two manifests,
failing closed on anything it cannot verify.
"""

from robotruth.contract.spec import (
    ActionSemantics,
    CameraSpec,
    Embodiment,
    ExecSpec,
    Normalization,
    ObservationSpec,
    PolicyIdentity,
    RuntimeSpec,
)
from robotruth.contract.check import CheckResult, Finding, Severity, check_contract
from robotruth.contract.extract import extract_spec

__all__ = [
    "ActionSemantics",
    "CameraSpec",
    "CheckResult",
    "Embodiment",
    "ExecSpec",
    "Finding",
    "Normalization",
    "ObservationSpec",
    "PolicyIdentity",
    "RuntimeSpec",
    "Severity",
    "check_contract",
    "extract_spec",
]
