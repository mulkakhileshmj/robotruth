"""Module 4: cell and unit fingerprints, and drift between them.

Why: setup drift dwarfs model differences. Moving a camera or tote shifted task completion
by 22 points (PhAIL). Two identical Franka FR3 arms replaying the same commands showed
6.3 mm versus 32.6 mm tracking error and a policy went from 98 to 18 percent (SPACE,
arXiv 2606.24049). Weights degrade within days from lighting change (1X). Nobody
fingerprints the cell or the unit before an evaluation, so nobody can tell "the model got
worse" from "the camera moved 4 mm".

Two fingerprints:
- unit: from a fixed excitation trajectory (commanded versus measured joint positions):
  per-joint tracking error, lag, backlash, steady-state error.
- cell: from a camera frame of the workspace with fiducial markers: marker positions,
  exposure, sharpness, colour balance, and (with intrinsics) marker pose.

`diff` compares two fingerprints against tolerances and returns findings the episode
record can carry.
"""

from robotruth.fingerprint.unit import UnitFingerprint, unit_fingerprint, diff_unit
from robotruth.fingerprint.cell import CellFingerprint, cell_fingerprint, diff_cell
from robotruth.fingerprint.common import DriftFinding, DriftReport

__all__ = [
    "CellFingerprint",
    "DriftFinding",
    "DriftReport",
    "UnitFingerprint",
    "cell_fingerprint",
    "diff_cell",
    "diff_unit",
    "unit_fingerprint",
]
