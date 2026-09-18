"""Robot unit fingerprint from a fixed excitation trajectory.

Input: a log of the robot replaying a fixed, known trajectory. Columns: `t` (seconds), and for
each joint j: `cmd_j` (commanded position) and `meas_j` (measured position). Radians or
metres, any consistent unit. Run the same trajectory on every unit and on the same unit every
session; the fingerprint is what should stay constant.

Per joint we estimate:
- rmse: tracking error between command and measurement (after lag alignment)
- lag_s: command-to-response delay (sub-sample) that minimises tracking error
- backlash: hysteresis width, the mean gap between measured position on rising versus
  falling command passes through the same commanded position
- steady_error: mean signed error during low-velocity segments (offset, calibration)
- gain: slope of measured versus commanded velocity (under 1 means sluggish or saturating)

These are cheap proxies for the actuator parameters SPACE and the actuator-ID paper fit with
a differentiable simulator. They are enough to flag a second unit or a worn one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from robotruth.fingerprint.common import DriftFinding, DriftReport, grade, stable_hash


@dataclass
class JointStats:
    rmse: float
    lag_s: float
    backlash: float
    steady_error: float
    gain: float
    max_abs_error: float


@dataclass
class UnitFingerprint:
    unit_id: str
    joints: dict[str, JointStats]
    n_samples: int
    duration_s: float
    sample_hz: float
    trajectory_sha256: str
    notes: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return stable_hash({"unit": self.unit_id, "traj": self.trajectory_sha256,
                            "joints": {k: asdict(v) for k, v in self.joints.items()}})

    def to_json(self, path: Path | str) -> Path:
        p = Path(path)
        p.write_text(json.dumps({"unit_id": self.unit_id, "n_samples": self.n_samples, "duration_s": self.duration_s,
                                 "sample_hz": self.sample_hz, "trajectory_sha256": self.trajectory_sha256,
                                 "fingerprint": self.fingerprint, "notes": self.notes,
                                 "joints": {k: asdict(v) for k, v in self.joints.items()}}, indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_json(cls, path: Path | str) -> "UnitFingerprint":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["unit_id"], {k: JointStats(**v) for k, v in d["joints"].items()}, d["n_samples"],
                   d["duration_s"], d["sample_hz"], d["trajectory_sha256"], d.get("notes", []))


def read_excitation_csv(path: Path | str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "t" not in rows[0]:
        raise ValueError("excitation CSV needs a 't' column plus cmd_<joint> and meas_<joint> columns")
    t = np.array([float(r["t"]) for r in rows])
    joints = sorted({k[4:] for k in rows[0] if k.startswith("cmd_")})
    cmd = {j: np.array([float(r[f"cmd_{j}"]) for r in rows]) for j in joints}
    meas = {j: np.array([float(r[f"meas_{j}"]) for r in rows]) for j in joints}
    return t, cmd, meas


def _align(t: np.ndarray, cmd: np.ndarray, meas: np.ndarray, max_lag_s: float = 0.5):
    """Jointly estimate lag and backlash.

    Step 1: integer-sample lag L minimising tracking RMSE.
    Step 2: at that lag, regress err = meas - cmd(t - L) on [1, v, sign(v)]. A residual lag delta
    shows up as err = -delta * v, hysteresis as err = -(b/2) * sign(v). So lag = L - k_v and
    backlash = 2 |k_sign|. Returns (lag_s, backlash, cmd_aligned, meas_aligned).
    """
    dt = float(np.median(np.diff(t)))
    n = len(t)
    max_lag = max(1, int(max_lag_s / dt))
    tail = max(1, int(0.05 * n))
    sl = slice(max_lag + tail, n - tail)

    def rmse_at(lag_s: float) -> float:
        e = (meas - np.interp(t - lag_s, t, cmd))[sl]
        return float(np.sqrt(np.mean(e ** 2)))

    coarse = np.arange(0, max_lag + 1) * dt
    L = float(coarse[int(np.argmin([rmse_at(l) for l in coarse]))])
    c = np.interp(t - L, t, cmd)
    v = np.gradient(c, dt)
    err = (meas - c)[sl]
    vv = v[sl]
    thr = 0.05 * (np.abs(vv).max() + 1e-12)
    sg = np.where(vv > thr, 1.0, np.where(vv < -thr, -1.0, 0.0))
    X = np.column_stack([np.ones_like(vv), vv, sg])
    coef, *_ = np.linalg.lstsq(X, err, rcond=None)
    _, k_v, k_s = coef
    lag = max(0.0, L - float(k_v))
    backlash = 2.0 * abs(float(k_s))
    c_fin = np.interp(t - lag, t, cmd)
    return lag, backlash, c_fin[sl], meas[sl]


def _joint_stats(t: np.ndarray, cmd: np.ndarray, meas: np.ndarray) -> JointStats:
    dt = float(np.median(np.diff(t)))
    lag, backlash, c, m = _align(t, cmd, meas)
    err = m - c
    vc = np.gradient(c, dt)
    vm = np.gradient(m, dt)
    slow = np.abs(vc) < (0.05 * (np.abs(vc).max() + 1e-12))
    steady = float(err[slow].mean()) if slow.sum() >= 3 else float(err.mean())
    mask = np.abs(vc) > 1e-9
    gain = float(np.dot(vc[mask], vm[mask]) / (np.dot(vc[mask], vc[mask]) + 1e-12)) if mask.sum() > 3 else 1.0
    return JointStats(rmse=float(np.sqrt(np.mean(err ** 2))), lag_s=lag, backlash=backlash,
                      steady_error=steady, gain=gain, max_abs_error=float(np.abs(err).max()))


def unit_fingerprint(t: np.ndarray, cmd: dict[str, np.ndarray], meas: dict[str, np.ndarray], unit_id: str = "unknown") -> UnitFingerprint:
    joints = {j: _joint_stats(t, cmd[j], meas[j]) for j in sorted(cmd)}
    traj_hash = stable_hash({j: np.round(cmd[j], 4).tolist() for j in sorted(cmd)})
    dt = float(np.median(np.diff(t)))
    return UnitFingerprint(unit_id, joints, int(t.size), float(t[-1] - t[0]), 1.0 / dt if dt > 0 else 0.0, traj_hash)


DEFAULT_UNIT_TOL = {
    # (warn, fail) as ratios of the reference value, with absolute floors below.
    "rmse": (0.5, 1.5), "backlash": (0.5, 1.5), "max_abs_error": (0.5, 1.5),
    # absolute
    "lag_s": (0.02, 0.05), "steady_error_abs": (0.005, 0.02), "gain": (0.05, 0.15),
}


def diff_unit(ref: UnitFingerprint, cur: UnitFingerprint, tol: Optional[dict] = None,
              floors: Optional[dict] = None) -> DriftReport:
    tol = {**DEFAULT_UNIT_TOL, **(tol or {})}
    floors = {"rmse": 0.002, "backlash": 0.001, "max_abs_error": 0.005, **(floors or {})}
    rep = DriftReport("unit")
    if ref.trajectory_sha256 != cur.trajectory_sha256:
        rep.findings.append(DriftFinding("trajectory", ref.trajectory_sha256[:12], cur.trajectory_sha256[:12], 1.0, 0.0, "fail",
                                         "Different excitation trajectories; fingerprints are not comparable."))
        return rep
    for j in sorted(set(ref.joints) | set(cur.joints)):
        if j not in ref.joints or j not in cur.joints:
            rep.findings.append(DriftFinding(f"{j}", None, None, 1.0, 0.0, "fail", "Joint present on one side only."))
            continue
        a, b = ref.joints[j], cur.joints[j]
        for m in ("rmse", "backlash", "max_abs_error"):
            va, vb = getattr(a, m), getattr(b, m)
            base = max(abs(va), floors[m])
            rel = (vb - va) / base
            warn, fail = tol[m]
            sev = grade(rel, warn, fail) if vb > va else "ok"  # getting better is not drift
            rep.findings.append(DriftFinding(f"{j}.{m}", va, vb, float(rel), fail, sev,
                                             f"{m} changed {100*rel:+.0f}% relative to reference."))
        d = b.lag_s - a.lag_s
        warn, fail = tol["lag_s"]
        rep.findings.append(DriftFinding(f"{j}.lag_s", a.lag_s, b.lag_s, float(d), fail, grade(d, warn, fail),
                                         f"command-to-response lag changed {1000*d:+.0f} ms."))
        d = b.steady_error - a.steady_error
        warn, fail = tol["steady_error_abs"]
        rep.findings.append(DriftFinding(f"{j}.steady_error", a.steady_error, b.steady_error, float(d), fail, grade(d, warn, fail),
                                         f"steady-state offset changed {d:+.4f} (calibration or encoder drift)."))
        d = b.gain - a.gain
        warn, fail = tol["gain"]
        rep.findings.append(DriftFinding(f"{j}.gain", a.gain, b.gain, float(d), fail, grade(d, warn, fail),
                                         f"velocity gain changed {d:+.3f} (sluggish or saturating actuator if negative)."))
    return rep
