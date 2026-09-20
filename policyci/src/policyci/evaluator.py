"""Versioned success definition. Changing this file's logic must bump EVALUATOR_VERSION;
the regression engine refuses to diff runs evaluated under different versions.

v1, ALOHA transfer-cube, from simulator ground truth (dm-control reward stages):
  success           max_reward >= 4 within the step budget
  grasp failure     never lifted the cube (max_reward < 2)
  action failure    lifted but never completed the transfer
  timeout           budget exhausted while still progressing

Collision and placement-error gates are not exposed by this env and are recorded as
unavailable rather than silently passed: an absent gate is not a passed gate.
"""

from __future__ import annotations

from robotruth.schema import FailureClass, FailureInfo, Outcome

EVALUATOR_VERSION = "0.1.0"
JUDGED_BY = "environment"


def evaluate(gt: dict, control_hz: float) -> tuple[Outcome, FailureInfo | None, dict]:
    """Map backend ground truth to a robotruth Outcome + FailureInfo + gate table."""
    success = bool(gt["success"])
    steps = int(gt["steps"])
    gates = {
        "grasp": "pass" if gt.get("lifted") else "fail",
        "transfer": "pass" if success else "fail",
        "time_budget": "fail" if gt.get("timed_out") else "pass",
        "collision": "unavailable",
        "placement_error": "unavailable",
    }
    outcome = Outcome(success=success, score=min(1.0, float(gt.get("max_reward", 0)) / 4.0),
                      judged_by=JUDGED_BY)
    failure = None
    if not success:
        if not gt.get("touched"):
            failure = FailureInfo(failure_class=FailureClass.GRASP, subclass="missed_grasp",
                                  evidence=[f"never touched the cube in {steps} steps"])
        elif not gt.get("lifted"):
            failure = FailureInfo(failure_class=FailureClass.GRASP, subclass="unstable_grasp",
                                  evidence=["touched but never lifted (max_reward < 2)"])
        elif gt.get("timed_out"):
            failure = FailureInfo(failure_class=FailureClass.TIMEOUT, subclass="no_progress",
                                  t_first_evidence=(gt.get("t_lift") or 0) / control_hz,
                                  evidence=[f"lifted at step {gt.get('t_lift')} but budget exhausted"])
        else:
            failure = FailureInfo(failure_class=FailureClass.ACTION, subclass="inconsistent_chunks",
                                  evidence=["lifted but transfer not completed"])
    return outcome, failure, gates
