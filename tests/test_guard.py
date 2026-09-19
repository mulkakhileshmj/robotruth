"""Tests for robotruth.guard: synthetic nominal and failure episodes, conformal false alarm
control, detection, hard limits, hysteresis, contrast sets, metrics, replay, persistence, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from robotruth.guard import (
    ActionStatsScorer,
    ChunkConsistencyScorer,
    CompositeScorer,
    ContrastSetCalibration,
    Guard,
    GuardEpisode,
    GuardMetrics,
    HardLimits,
    MahalanobisScorer,
    OfflineReplay,
    SequentialConformal,
    StagnationScorer,
    attach,
    calibrate_guard,
    conformal_quantile,
    guard_app,
    ledoit_wolf_covariance,
    load_episode_npz,
    save_episode_npz,
    scorer_from_dict,
)
from robotruth.guard.scores import _torch_cuda

F, CHUNK, D, STRIDE, T = 6, 8, 4, 4, 60
DT = 0.1


def _world(seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(F, F))
    cov = a @ a.T / F + 0.5 * np.eye(F)
    mean = rng.normal(size=F)
    return mean, cov


MEAN, COV = _world()
SIG = np.sqrt(np.diag(COV))


def make_episode(rng, kind: str = "nominal", t_steps: int = T) -> dict:
    """kind: nominal | failure | benign.

    nominal: Gaussian features, smooth consistent action chunks.
    failure: from the midpoint the features drift off-distribution and the chunks stop
        agreeing with each other.
    benign: a constant harmless shift of the features (new lighting), chunks unchanged.
    """
    feats = rng.multivariate_normal(MEAN, COV, size=t_steps)
    phase = rng.uniform(0, 2 * np.pi, size=D)
    freq = rng.uniform(0.02, 0.05, size=D)
    s = np.arange(t_steps * STRIDE + CHUNK)
    traj = np.sin(np.outer(s, freq) + phase) * 0.5
    chunks = np.stack([traj[i * STRIDE: i * STRIDE + CHUNK] for i in range(t_steps)])
    chunks = chunks + rng.normal(scale=0.02, size=chunks.shape)
    if kind == "failure":
        half = t_steps // 2
        ramp = np.linspace(0.5, 1.0, t_steps - half)[:, None]
        feats[half:] += ramp * 5.0 * SIG[None, :] * np.array([1, 1, 1, 0, 0, 0])
        chunks[half:] += rng.normal(scale=0.5, size=chunks[half:].shape)
    elif kind == "benign":
        feats += 1.5 * SIG
    ts = np.arange(t_steps) * DT
    truth = {"nominal": "success", "benign": "success", "failure": "failure"}[kind]
    return {"features": feats, "actions": chunks, "timestamps": ts, "truth": truth}


def episodes(rng, kind: str, n: int) -> list[dict]:
    return [make_episode(rng, kind) for _ in range(n)]


# scorers ---------------------------------------------------------------------------

def test_ledoit_wolf_shrinkage_is_sane():
    rng = np.random.default_rng(1)
    x = rng.multivariate_normal(MEAN, COV, size=2000)
    cov, shrink = ledoit_wolf_covariance(x)
    assert 0.0 <= shrink <= 1.0
    assert np.allclose(cov, COV, atol=0.25)
    # few samples, many dims: shrinkage must kick in and keep the matrix well conditioned
    y = rng.normal(size=(12, 40))
    cov2, shrink2 = ledoit_wolf_covariance(y)
    assert shrink2 > 0.0
    assert np.linalg.cond(cov2) < 1e6


def test_mahalanobis_single_equals_batch_and_reports_backend():
    rng = np.random.default_rng(2)
    x = rng.multivariate_normal(MEAN, COV, size=500)
    m = MahalanobisScorer(device="cpu").fit(x)
    q = rng.multivariate_normal(MEAN, COV, size=7)
    batch = m.score(q)
    singles = np.array([m.score(v) for v in q])
    assert np.allclose(batch, singles)
    assert m.last_backend == "numpy"
    # a far point scores far
    assert m.score(MEAN + 10 * SIG) > np.max(batch)


def test_mahalanobis_cuda_path_matches_numpy():
    torch = _torch_cuda()
    if torch is None:
        pytest.skip("torch with CUDA not available on this box; numpy path only")
    rng = np.random.default_rng(3)
    x = rng.multivariate_normal(MEAN, COV, size=500)
    cpu = MahalanobisScorer(device="cpu").fit(x)
    gpu = MahalanobisScorer(device="auto").fit(x)
    q = rng.multivariate_normal(MEAN, COV, size=64)
    a, b = cpu.score(q), gpu.score(q)
    assert gpu.last_backend == "torch_cuda", gpu.last_backend
    assert np.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_chunk_consistency_and_action_stats_behave():
    rng = np.random.default_rng(4)
    noms = episodes(rng, "nominal", 5)
    cc = ChunkConsistencyScorer(stride=STRIDE).fit([e["actions"] for e in noms])
    z_nom = cc.score_episode(noms[0])
    fail = make_episode(rng, "failure")
    z_fail = cc.score_episode(fail)
    assert np.abs(z_nom[1:]).mean() < 2.0
    assert z_fail[T // 2 + 1:].mean() > 5.0
    # stride >= chunk falls back to the boundary jump and still works
    cc2 = ChunkConsistencyScorer(stride=CHUNK).fit([e["actions"] for e in noms])
    assert np.isfinite(cc2.score_episode(noms[1])).all()
    st = ActionStatsScorer().fit(np.concatenate([e["actions"] for e in noms]))
    frozen = np.zeros((CHUNK, D))
    assert st.score(frozen) > st.score(noms[0]["actions"][5])


def make_frozen(rng, t_steps: int = T) -> dict:
    """A nominal episode whose policy stalls at the midpoint: every later chunk repeats."""
    ep = make_episode(rng, "nominal", t_steps)
    half = t_steps // 2
    ep["actions"] = ep["actions"].copy()
    ep["actions"][half:] = ep["actions"][half]
    ep["truth"] = "failure"
    return ep


def test_stagnation_scorer_sees_a_stall_and_the_others_do_not():
    rng = np.random.default_rng(5)
    noms = episodes(rng, "nominal", 8)
    st = StagnationScorer(window=5).fit([e["actions"] for e in noms])
    z_nom = st.score_episode(noms[0])
    frozen = make_frozen(rng)
    z_frozen = st.score_episode(frozen)
    assert z_nom.mean() < 1.5 and z_nom.min() >= 0.0
    assert z_frozen[T // 2 + 5:].min() > 5.0
    # streaming and episode scoring agree
    st.reset()
    stream = np.array([st.score(c) for c in frozen["actions"]])
    assert np.allclose(stream, z_frozen)
    d = st.to_dict()
    st2 = scorer_from_dict(json.loads(json.dumps(d)))
    assert np.allclose(st2.score_episode(frozen), z_frozen)


def test_guard_detects_a_stalled_policy_with_no_extra_false_alarms():
    rng = np.random.default_rng(12)
    guard, report = calibrate_guard(episodes(rng, "nominal", 80), alpha=0.05, patience=3, dt=DT, stride=STRIDE)
    assert "stall" in guard.scorer.heads
    assert not report["stall_head"]["saturated"]
    replay = OfflineReplay(guard)
    frozen = [replay.run(make_frozen(rng)).episode for _ in range(30)]
    m = GuardMetrics.from_episodes(frozen)
    assert m.detection_rate.estimate >= 0.9, m.to_markdown()
    onset = (T // 2) * DT
    firsts = np.array([r.first_alert_t for r in frozen if r.detected])
    assert np.median(firsts) >= onset - 1e-9
    assert np.median(firsts) <= onset + 15 * DT
    held = [replay.run(e).episode for e in episodes(rng, "nominal", 100)]
    assert np.mean([r.detected for r in held]) <= 0.10


# conformal -------------------------------------------------------------------------

def test_conformal_quantile_rank_and_saturation():
    v = np.arange(1, 20, dtype=float)          # n=19
    thr, sat, ach = conformal_quantile(v, 0.1)  # rank ceil(20*0.9)=18
    assert thr == 18.0 and not sat and ach == 0.1
    thr, sat, ach = conformal_quantile(v, 0.01)  # rank 20 > 19
    assert thr == 19.0 and sat and abs(ach - 1 / 20) < 1e-12


@pytest.mark.parametrize("method", ["max", "bonferroni"])
def test_false_alarm_rate_on_held_out_nominal_at_most_alpha(method):
    rng = np.random.default_rng(10)
    alpha, tol = 0.10, 0.05
    guard, report = calibrate_guard(episodes(rng, "nominal", 120), alpha=alpha, method=method, n_bins=3,
                                    patience=1, dt=DT, stride=STRIDE, seed=0)
    assert not report["saturated"], report.get("warning")
    held = episodes(rng, "nominal", 300)
    replay = OfflineReplay(guard)
    recs = [replay.run(e).episode for e in held]
    guard_rate = np.mean([r.detected for r in recs])
    raw_rate = np.mean([guard.calibrator.alarms(guard.scorer.score_episode(e)) for e in held])
    assert raw_rate <= alpha + tol, f"raw any-time false alarm rate {raw_rate:.3f} ({method})"
    assert guard_rate <= raw_rate + 1e-9
    assert guard_rate <= alpha + tol
    # in-sample the calibration set itself must sit at or below alpha too
    assert report["calibrator"]["calibration_alarm_fraction"] <= alpha + 1e-9


def test_detects_failures_before_episode_end():
    rng = np.random.default_rng(11)
    guard, _ = calibrate_guard(episodes(rng, "nominal", 80), alpha=0.05, patience=3, dt=DT, stride=STRIDE)
    fails = episodes(rng, "failure", 40)
    replay = OfflineReplay(guard)
    recs = [replay.run(e).episode for e in fails]
    m = GuardMetrics.from_episodes(recs)
    assert m.detection_rate.estimate >= 0.9, m.to_markdown()
    assert all(r.lead_time_s > 0 for r in recs if r.detected)
    onset = (T // 2) * DT
    firsts = np.array([r.first_alert_t for r in recs if r.detected])
    assert np.median(firsts) >= onset - 1e-9        # alarms come after the drift starts
    assert np.median(firsts) <= onset + 10 * DT     # and not long after
    assert all(r.max_level in ("slow", "handover") for r in recs if r.detected)


def test_hard_limit_is_an_immediate_latched_stop():
    rng = np.random.default_rng(12)
    guard, _ = calibrate_guard(episodes(rng, "nominal", 20), alpha=0.1, dt=DT, stride=STRIDE,
                               hard_limits=HardLimits(max_action_abs=1.0, joint_min=[-2] * D, joint_max=[2] * D))
    ep = make_episode(rng, "nominal")
    guard.start_episode("hl")
    ev = guard.step(features=ep["features"][0], action_chunk=ep["actions"][0])
    assert ev.level == "ok"
    bad = ep["actions"][1].copy()
    bad[3, 0] = 5.0
    ev = guard.step(features=ep["features"][1], action_chunk=bad)
    assert ev.level == "stop" and ev.changed and "hard limit" in ev.reason
    ev = guard.step(features=ep["features"][2], action_chunk=ep["actions"][2])
    assert ev.level == "stop" and "latched" in ev.reason
    rec = guard.end_episode("failure")
    assert rec.stopped and rec.max_level == "stop" and rec.first_alert_t == pytest.approx(1 * DT)
    # joint limit on the measured state alone
    guard.start_episode("state")
    ev = guard.step(features=ep["features"][0], action_chunk=ep["actions"][0], state=np.array([0, 0, 0, 3.0]))
    assert ev.level == "stop" and "joint limit" in ev.reason
    guard.end_episode(None)


class _StubScorer:
    def __init__(self, seq):
        self.seq, self.i = list(seq), 0

    def reset(self):
        self.i = 0

    def score_step(self, features=None, action_chunk=None):
        s = self.seq[self.i]
        self.i += 1
        return s, {"stub": s}


class _ConstThreshold:
    def threshold(self, t):
        return 1.0


def _run(guard: Guard, n: int) -> list[str]:
    guard.start_episode()
    return [guard.step(features=np.zeros(1)).level for _ in range(n)]


def test_hysteresis_prevents_flapping_and_releases_on_sustained_drop():
    seq = [1.2, 0.9] * 20 + [0.3] * 4
    flappy = Guard(_StubScorer(seq), _ConstThreshold(), patience=1, hysteresis=0.0)
    steady = Guard(_StubScorer(seq), _ConstThreshold(), patience=1, hysteresis=0.3)
    lv_flappy, lv_steady = _run(flappy, len(seq)), _run(steady, len(seq))
    changes = lambda lv: sum(a != b for a, b in zip(lv, lv[1:]))  # noqa: E731
    assert changes(lv_flappy) >= 20            # oscillates ok/slow every step
    assert changes(lv_steady[:40]) <= 2        # ok -> slow -> handover, then holds
    assert lv_steady[39] == "handover"
    assert lv_steady[-1] == "ok"               # a real drop below the release band releases
    rec = steady.end_episode("success")
    assert rec.n_alert_onsets == 1             # one false alarm, not twenty
    flappy.end_episode("success")
    assert flappy.episodes[-1].n_alert_onsets >= 10


def test_contrast_set_raises_threshold_and_stops_benign_alarms():
    rng = np.random.default_rng(13)
    noms = episodes(rng, "nominal", 100)
    scorer = CompositeScorer.default(stride=STRIDE).fit(noms[:50])
    nom_traj = [scorer.score_episode(e) for e in noms[50:]]
    ben_traj = [scorer.score_episode(e) for e in episodes(rng, "benign", 50)]
    csc = ContrastSetCalibration(alpha=0.1, method="max").fit(nom_traj, ben_traj)
    rep = csc.report
    assert rep.union.threshold(0) >= rep.nominal_only.threshold(0)
    assert rep.threshold_shift > 0
    assert rep.benign_alarm_fraction_under_nominal > rep.benign_alarm_fraction_under_union
    # the guarantee is on the union mixture: fresh nominal plus benign episodes pooled
    fresh = [scorer.score_episode(e) for e in episodes(rng, "nominal", 50) + episodes(rng, "benign", 50)]
    pooled = np.mean([rep.union.alarms(t) for t in fresh])
    assert pooled <= 0.1 + 0.05, f"pooled alarm rate under union threshold {pooled:.3f}"
    assert "Threshold moved by +" in rep.to_markdown()
    # the same through calibrate_guard
    guard, report = calibrate_guard(noms, benign=episodes(rng, "benign", 30), alpha=0.1, stride=STRIDE)
    assert report["contrast"]["threshold_shift"] > 0
    # the composite lives in the guard's main head; its calibrator carries the union threshold
    main_cal = guard.scorer.heads["main"][1]
    assert main_cal.threshold(0) == pytest.approx(report["contrast"]["union"]["thresholds"][0])


# metrics ---------------------------------------------------------------------------

def _ep(eid, truth, dur, first, onsets, level="slow", stopped=False):
    return GuardEpisode(eid, truth, dur, int(dur / DT), first, level if first is not None else "ok",
                        onsets, stopped, 1.0, 0.0)


def test_metrics_false_alarms_per_hour_and_detection_on_hand_built_case():
    eps = [
        _ep("n1", True, 1800.0, 500.0, 1),      # one false alarm in half an hour
        _ep("n2", True, 1800.0, None, 0),       # clean half hour
        _ep("f1", False, 100.0, 90.0, 1),       # detected 10 s before the end
        _ep("f2", False, 100.0, 80.0, 1),       # detected 20 s before the end
        _ep("f3", False, 100.0, None, 0),       # missed
        _ep("u1", None, 50.0, None, 0),         # unlabelled, ignored
    ]
    m = GuardMetrics.from_episodes(eps, alpha=0.05)
    assert m.nominal_hours == pytest.approx(1.0)
    assert m.n_false_alarms == 1 and m.false_alarms_per_hour == pytest.approx(1.0)
    lo, hi = m.false_alarms_per_hour_ci
    assert lo < 1.0 < hi and lo > 0
    assert m.detection_rate.estimate == pytest.approx(2 / 3) and m.detection_rate.method == "Wilson"
    assert m.lead_time_mean_s == pytest.approx(15.0) and m.lead_time_median_s == pytest.approx(15.0)
    assert m.nominal_episode_alarm_rate.estimate == pytest.approx(0.5)
    assert m.n_unlabelled == 1
    md = m.to_markdown()
    assert "False alarms per hour: 1.00 [" in md and "Detection rate: 0.667" in md
    assert "1 episodes without a truth label" in md
    json.dumps(m.to_dict())


# adapters --------------------------------------------------------------------------

def test_offline_replay_runs_from_npz(tmp_path: Path):
    rng = np.random.default_rng(14)
    guard, _ = calibrate_guard(episodes(rng, "nominal", 20), alpha=0.1, dt=DT, stride=STRIDE)
    ep = make_episode(rng, "nominal")
    p = save_episode_npz(tmp_path / "ep_nom.npz", ep["features"], ep["actions"], ep["timestamps"], truth="success")
    loaded = load_episode_npz(p)
    assert loaded["truth"] == "success" and loaded["features"].shape == (T, F)
    res = OfflineReplay(guard).run(p)
    assert res.episode.episode_id == "ep_nom" and res.episode.truth is True
    assert res.episode.n_steps == T and res.episode.duration_s == pytest.approx(T * DT)
    assert len(res.events) == T and all(np.isfinite(e.score) for e in res.events)
    assert "truth=success" in res.verdict
    # the hook path used by policy servers accepts a [1, chunk, D] batch and tensor-like inputs
    hook = attach(guard)
    guard.start_episode()
    ev = hook(ep["features"][0], ep["actions"][0][None], t=0.0)
    assert ev.level in ("ok", "slow") and ev.threshold is not None


def test_guard_json_round_trips(tmp_path: Path):
    rng = np.random.default_rng(15)
    guard, _ = calibrate_guard(episodes(rng, "nominal", 30), alpha=0.1, method="bonferroni", n_bins=2,
                               patience=2, hysteresis=0.1, dt=DT, stride=STRIDE,
                               hard_limits=HardLimits(max_action_norm=3.0, workspace_min=[-1, -1, -1]))
    p = guard.save(tmp_path / "guard.json")
    g2 = Guard.load(p)
    assert g2.to_dict() == json.loads(json.dumps(guard.to_dict()))
    assert np.allclose(g2.calibrator.thresholds(T), guard.calibrator.thresholds(T))
    ep = make_episode(rng, "failure")
    assert np.allclose(g2.scorer.score_episode(ep), guard.scorer.score_episode(ep))
    assert g2.hard_limits.max_action_norm == 3.0 and g2.patience == 2 and g2.hysteresis == 0.1
    r1, r2 = OfflineReplay(guard).run(ep).episode, OfflineReplay(g2).run(ep).episode
    assert r1.first_alert_t == r2.first_alert_t and r1.max_level == r2.max_level


# cli -------------------------------------------------------------------------------

def test_cli_calibrate_replay_metrics(tmp_path: Path):
    rng = np.random.default_rng(16)
    nom_dir, ben_dir, lab_dir = tmp_path / "nominal", tmp_path / "benign", tmp_path / "labelled"
    for d in (nom_dir, ben_dir, lab_dir):
        d.mkdir()
    for i, e in enumerate(episodes(rng, "nominal", 30)):
        save_episode_npz(nom_dir / f"nom_{i:03d}.npz", e["features"], e["actions"], e["timestamps"], truth="success")
    for i, e in enumerate(episodes(rng, "benign", 10)):
        save_episode_npz(ben_dir / f"ben_{i:03d}.npz", e["features"], e["actions"], e["timestamps"], truth="success")
    for i, e in enumerate(episodes(rng, "nominal", 10) + episodes(rng, "failure", 10)):
        save_episode_npz(lab_dir / f"ep_{i:03d}.npz", e["features"], e["actions"], e["timestamps"], truth=e["truth"])
    runner = CliRunner()
    gj = tmp_path / "guard.json"
    r = runner.invoke(guard_app, ["calibrate", str(nom_dir), "--out", str(gj), "--alpha", "0.1",
                                  "--benign-dir", str(ben_dir), "--stride", str(STRIDE)])
    assert r.exit_code == 0, r.output
    assert gj.exists() and "contrast set" in r.output
    fail_ep = sorted(lab_dir.glob("*.npz"))[-1]
    r = runner.invoke(guard_app, ["replay", str(gj), str(fail_ep)])
    assert r.exit_code == 1, r.output
    assert fail_ep.with_suffix(".events.jsonl").exists() and "first alert at" in r.output
    r = runner.invoke(guard_app, ["metrics", str(gj), str(lab_dir), "--out", str(tmp_path / "m.md")])
    assert r.exit_code == 0, r.output
    assert "False alarms per hour" in r.output and "Detection rate" in r.output
    assert (tmp_path / "m.md").exists()
