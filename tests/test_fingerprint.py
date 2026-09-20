import json

import numpy as np
import pytest

from robotruth.fingerprint import (
    CellFingerprint,
    DriftDetector,
    UnitFingerprint,
    cell_fingerprint,
    diff_cell,
    diff_unit,
    unit_fingerprint,
)


def _excitation(seed=0, lag_s=0.02, backlash=0.002, noise=0.0005, gain=1.0, offset=0.0, hz=100.0, dur=20.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / hz)
    cmd = {}
    meas = {}
    for j, f in (("j1", 0.3), ("j2", 0.5)):
        c = 0.5 * np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * 1.3 * f * t)
        shift = int(round(lag_s * hz))
        m = np.concatenate([np.full(shift, c[0]), c[:-shift]]) if shift else c.copy()
        m = gain * m + offset
        # backlash: measured lags behind on direction reversals
        v = np.gradient(c)
        m = m - 0.5 * backlash * np.sign(v)
        m = m + rng.normal(0, noise, size=t.size)
        cmd[j], meas[j] = c, m
    return t, cmd, meas


def test_unit_fingerprint_recovers_lag_and_backlash():
    t, cmd, meas = _excitation(lag_s=0.03, backlash=0.004)
    fp = unit_fingerprint(t, cmd, meas, "u1")
    js = fp.joints["j1"]
    assert abs(js.lag_s - 0.03) <= 0.011
    assert 0.002 < js.backlash < 0.007
    assert js.rmse < 0.01
    assert abs(js.gain - 1.0) < 0.05


def test_diff_unit_flags_a_worse_second_unit_and_passes_same_unit(tmp_path):
    t, cmd, meas = _excitation(seed=1)
    ref = unit_fingerprint(t, cmd, meas, "u1")
    t2, cmd2, meas2 = _excitation(seed=2)
    same = unit_fingerprint(t2, cmd2, meas2, "u1")
    rep_same = diff_unit(ref, same)
    assert rep_same.passed, rep_same.summary()
    # SPACE-style second unit: 5x tracking error, more lag, more backlash, an offset
    t3, cmd3, meas3 = _excitation(seed=3, lag_s=0.08, backlash=0.01, noise=0.003, gain=0.85, offset=0.03)
    other = unit_fingerprint(t3, cmd3, meas3, "u2")
    rep = diff_unit(ref, other)
    assert not rep.passed
    failed = {f.metric for f in rep.findings if f.severity == "fail"}
    assert any(m.endswith(".lag_s") for m in failed)
    assert any(m.endswith(".rmse") or m.endswith(".backlash") for m in failed)
    assert any(m.endswith(".steady_error") for m in failed)
    # roundtrip
    p = ref.to_json(tmp_path / "u1.json")
    again = UnitFingerprint.from_json(p)
    assert again.fingerprint == ref.fingerprint


def test_diff_unit_refuses_different_trajectories():
    t, cmd, meas = _excitation()
    a = unit_fingerprint(t, cmd, meas, "u1")
    cmd2 = {k: v * 1.1 for k, v in cmd.items()}
    b = unit_fingerprint(t, cmd2, meas, "u1")
    rep = diff_unit(a, b)
    assert not rep.passed and rep.findings[0].metric == "trajectory"


def _scene(shift=(0, 0), brightness=0, blur=0, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), 120, np.uint8)
    img[:] = np.clip(img.astype(int) + rng.integers(-10, 10, size=img.shape), 0, 255).astype(np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid, (x, y) in {3: (80, 80), 7: (420, 90), 11: (250, 320)}.items():
        m = cv2.aruco.generateImageMarker(d, mid, 80)
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        x, y = x + shift[0], y + shift[1]
        img[y:y + 80, x:x + 80] = m
    if brightness:
        img = np.clip(img.astype(int) + brightness, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def test_cell_fingerprint_detects_markers_and_camera_bump(tmp_path):
    ref = cell_fingerprint(_scene(), "top")
    assert set(ref.markers) == {3, 7, 11}
    same = cell_fingerprint(_scene(seed=1), "top")
    rep = diff_cell(ref, same)
    assert rep.passed, rep.summary()
    bumped = cell_fingerprint(_scene(shift=(9, 4)), "top")
    rep = diff_cell(ref, bumped)
    assert not rep.passed
    assert any(f.metric.endswith("shift_px") and f.severity == "fail" for f in rep.findings)
    p = ref.to_json(tmp_path / "cell.json")
    again = CellFingerprint.from_json(p)
    assert again.fingerprint == ref.fingerprint


def test_cell_fingerprint_flags_lighting_and_blur():
    ref = cell_fingerprint(_scene(), "top")
    dark = cell_fingerprint(_scene(brightness=-60), "top")
    rep = diff_cell(ref, dark)
    assert any(f.metric == "mean_luma" and f.severity == "fail" for f in rep.findings)
    blurry = cell_fingerprint(_scene(blur=3), "top")
    rep = diff_cell(ref, blurry)
    assert any(f.metric == "sharpness" and f.severity in ("warn", "fail") for f in rep.findings)


# calibrated drift detection ---------------------------------------------------------

def _population(n, seed0=100, **kw):
    return [unit_fingerprint(*_excitation(seed=seed0 + i, **kw), f"u{i}") for i in range(n)]


def test_drift_detector_holds_its_false_alarm_rate_and_catches_offset():
    nominal = _population(60)
    det = DriftDetector(alpha=0.10, fields=("steady_error",)).fit(nominal)
    assert det.fitted and not det.saturated_, det.summary()
    held = _population(80, seed0=900)
    fa = np.mean([det.alarms(fp) for fp in held])
    assert fa <= 0.10 + 0.06, f"false alarm rate {fa:.3f}"
    drifted = _population(30, seed0=500, offset=0.05)
    assert np.mean([det.alarms(fp) for fp in drifted]) >= 0.9
    worst = det.explain(drifted[0])[0]
    assert worst[0].endswith("steady_error") and worst[2] > 1.0


def test_drift_detector_reports_saturation_instead_of_an_uncertifiable_rate():
    det = DriftDetector(alpha=0.01, fields=("steady_error",)).fit(_population(8))
    assert det.saturated_ and det.achievable_alpha_ > 0.01
    assert "SATURATED" in det.summary()


def test_drift_detector_round_trips_and_needs_fitting():
    det = DriftDetector(alpha=0.05, fields=("steady_error", "gain")).fit(_population(40))
    fp = _population(1, seed0=777)[0]
    again = DriftDetector.from_dict(json.loads(json.dumps(det.to_dict())))
    assert again.score(fp) == pytest.approx(det.score(fp))
    assert again.alarms(fp) == det.alarms(fp)
    with pytest.raises(RuntimeError):
        DriftDetector().score(fp)
