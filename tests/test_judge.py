"""Tests for module 5, the calibrated hybrid outcome judge. Synthetic episodes only."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from robotruth.judge import (
    ABSTAIN,
    FAILURE,
    FUSION_FEATURE_NAMES,
    SUCCESS,
    AnthropicBackend,
    Calibrator,
    EpisodeSignals,
    FusionModel,
    HybridJudge,
    MockBackend,
    TaskCalibrator,
    action_stream_features,
    evaluate,
    fit_calibrator,
    fit_fusion,
    load_failbench_jsonl,
    metrics_from_predictions,
    parse_verdict,
    run_failbench,
    subsample_indices,
    windowed_features,
)
from robotruth.judge.cli import judge_app
from robotruth.judge.vlm import SYSTEM_PROMPT, encode_png
from robotruth.schema import FailureInfo

HZ = 20
T = 200
D = 6
CHUNK = 8


def make_episode(rng: np.random.Generator, success: bool, i: int, task: str = "pick_cup") -> EpisodeSignals:
    """Successes move smoothly and close the gripper once. Failures freeze for a stretch,
    toggle the gripper many times and emit inconsistent action chunks."""
    t = np.arange(T) / HZ
    phase = rng.uniform(0, 2 * np.pi, D)
    amp = rng.uniform(0.2, 0.5, D)
    pos = amp * np.sin(2 * np.pi * 0.15 * t[:, None] + phase)
    gripper = np.ones(T)
    gripper[T // 2:] = 0.0
    if not success:
        f0 = int(rng.integers(50, 80))
        f1 = f0 + int(rng.integers(60, 90))
        pos[f0:f1] = pos[f0]
        n_tog = int(rng.integers(5, 10))
        idx = np.sort(rng.choice(np.arange(20, T - 20), n_tog, replace=False))
        g = np.ones(T)
        state = 1.0
        for k in idx:
            state = 1.0 - state
            g[k:] = state
        gripper = g
    states = pos + rng.normal(0, 1e-4, pos.shape)
    a = np.stack([pos[np.minimum(np.arange(T) + c, T - 1)] for c in range(CHUNK)], axis=1)
    a = a + rng.normal(0, 0.002 if success else 0.02, a.shape)
    return EpisodeSignals(actions=a, timestamps=t, instruction=f"pick up cup number {i}", task=task,
                          episode_id=f"e{i}", states=states, gripper=gripper, frames=None)


def make_dataset(seed: int, n: int, task: str = "pick_cup", offset: int = 0) -> list[tuple[EpisodeSignals, str]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        success = bool(i % 2 == 0)
        out.append((make_episode(rng, success, offset + i, task), "success" if success else "failure"))
    return out


def weak_vlm(episodes: list[tuple[EpisodeSignals, str]], seed: int) -> MockBackend:
    """A mock VLM that is right more often than not but far from perfect, like FailBench says."""
    rng = np.random.default_rng(seed)
    table = {}
    for sig, label in episodes:
        centre = 0.62 if label == "success" else 0.45
        table[sig.instruction] = float(np.clip(centre + rng.normal(0, 0.18), 0, 1))
    return MockBackend(fn=lambda frames, instr, task: table.get(instr, 0.5))


# ----------------------------------------------------------------------------- features

def test_action_features_separate_failures_from_successes():
    data = make_dataset(0, 40)
    feats = {label: [] for label in ("success", "failure")}
    for sig, label in data:
        feats[label].append(action_stream_features(sig))
    stag_s = np.mean([f["stagnation_fraction"] for f in feats["success"]])
    stag_f = np.mean([f["stagnation_fraction"] for f in feats["failure"]])
    assert stag_s < 0.1 and stag_f > 0.3
    assert all(f["gripper_toggles"] == 1 for f in feats["success"])
    assert all(f["gripper_toggles"] >= 5 for f in feats["failure"])
    inc_s = np.mean([f["temporal_inconsistency"] for f in feats["success"]])
    inc_f = np.mean([f["temporal_inconsistency"] for f in feats["failure"]])
    assert inc_f > 3 * inc_s
    f0 = feats["success"][0]
    assert f0["has_actions"] == 1.0 and f0["n_steps"] == T
    assert f0["duration_s"] == pytest.approx((T - 1) / HZ)
    assert f0["control_hz"] == pytest.approx(HZ)
    assert f0["chunk_variance"] > 0


def test_features_without_actions_and_2d_actions():
    empty = EpisodeSignals(actions=None, timestamps=None, duration_s=33.0, task="x")
    f = action_stream_features(empty)
    assert f["has_actions"] == 0.0 and f["duration_s"] == 33.0 and f["stagnation_fraction"] == 0.0
    rng = np.random.default_rng(3)
    sig = make_episode(rng, True, 0)
    flat = EpisodeSignals(actions=sig.actions[:, 0, :], timestamps=sig.timestamps, states=sig.states, gripper=sig.gripper)
    f2 = action_stream_features(flat)
    assert f2["chunk_variance"] == 0.0 and f2["temporal_inconsistency"] == 0.0
    assert f2["action_mag_mean"] > 0


def test_windowed_features_flag_the_frozen_span():
    rng = np.random.default_rng(7)
    sig = make_episode(rng, False, 1)
    windows = windowed_features(sig, window_s=2.0, stride_s=1.0)
    assert len(windows) >= 5
    ends = [w["t_end"] for w in windows]
    assert ends == sorted(ends) and ends[-1] == pytest.approx(sig.timestamps[-1])
    assert max(w["stagnation_fraction"] for w in windows) > 0.8
    assert min(w["stagnation_fraction"] for w in windows) < 0.2


# ----------------------------------------------------------------------------- fusion

def test_fusion_model_learns_the_failure_signature():
    train = make_dataset(10, 120)
    test = make_dataset(11, 120, offset=1000)
    backend = weak_vlm(train + test, seed=5)
    model = fit_fusion(backend, train, l2=1.0)
    assert model.fitted and model.feature_names == FUSION_FEATURE_NAMES
    judge = HybridJudge(backend, fusion=model)
    correct = 0
    for sig, label in test:
        r = judge.judge_episode(sig)
        correct += (r.decision == label)
    assert correct / len(test) >= 0.95
    coef = model.coefficients()
    assert coef["stagnation_fraction"] < 0 or coef["gripper_toggles"] < 0 or coef["temporal_inconsistency"] < 0
    # the VLM alone is much weaker on the same episodes
    vlm_only = HybridJudge(backend)
    vlm_correct = sum(vlm_only.judge_episode(sig).decision == label for sig, label in test)
    assert vlm_correct < correct


def test_fusion_model_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.3, 200) > 0).astype(int)
    m = FusionModel(feature_names=["a", "b", "c"], l2=0.5).fit(X, y)
    p = m.predict_proba(X)
    m.save(tmp_path / "fusion.json")
    m2 = FusionModel.load(tmp_path / "fusion.json")
    assert np.allclose(m2.predict_proba(X), p)
    assert m2.feature_names == ["a", "b", "c"] and m2.n_train == 200
    assert isinstance(m2.predict_proba(X[0]), float)
    assert m.train_report["train_accuracy"] > 0.85


# ----------------------------------------------------------------------------- conformal

def _scores(rng, n):
    y = rng.integers(0, 2, n)
    s = np.where(y == 1, rng.normal(0.72, 0.15, n), rng.normal(0.30, 0.15, n))
    return np.clip(s, 0, 1), y


def test_conformal_calibration_hits_target_error_on_held_out():
    rng = np.random.default_rng(1)
    s_cal, y_cal = _scores(rng, 600)
    s_test, y_test = _scores(rng, 600)
    target = 0.05
    cal = Calibrator().fit(s_cal, y_cal, target_error=target)
    assert cal.fitted and cal.cal_report["selective_error"] <= target
    dec = np.asarray(cal.decide_many(s_test))
    decided = dec != ABSTAIN
    errors = ((dec == SUCCESS) & (y_test == 0)).sum() + ((dec == FAILURE) & (y_test == 1)).sum()
    sel_err = errors / decided.sum()
    assert sel_err <= target + 0.03
    abstain_rate = 1 - decided.mean()
    assert 0.02 < abstain_rate < 0.5
    # every decided success sits above every decided failure
    assert s_test[dec == SUCCESS].min() > s_test[dec == FAILURE].max()
    # a looser target abstains less
    loose = Calibrator().fit(s_cal, y_cal, target_error=0.20)
    loose_abstain = np.mean(np.asarray(loose.decide_many(s_test)) == ABSTAIN)
    assert loose_abstain < abstain_rate
    # the strict variant abstains on the empty band too, so it can only abstain more
    strict = Calibrator(abstain_on_empty=True).fit(s_cal, y_cal, target_error=0.20)
    assert np.mean(np.asarray(strict.decide_many(s_test)) == ABSTAIN) >= loose_abstain
    assert Calibrator.from_dict(strict.to_dict()).abstain_on_empty is True
    # save and load give identical decisions
    cal.save(_tmp := __import__("pathlib").Path(__file__).parent / "_cal_tmp.json")
    try:
        again = Calibrator.load(_tmp)
        assert again.decide_many(s_test) == dec.tolist()
    finally:
        _tmp.unlink(missing_ok=True)


def test_calibrator_rejects_tiny_calibration_sets_and_abstains_on_overlap():
    with pytest.raises(ValueError):
        Calibrator().fit([0.9, 0.1, 0.8], ["success", "failure", "success"])
    rng = np.random.default_rng(4)
    # scores carry no information: the calibrator must abstain almost everywhere
    s = rng.uniform(0, 1, 400)
    y = rng.integers(0, 2, 400)
    cal = Calibrator().fit(s, y, target_error=0.05)
    abstain = np.mean(np.asarray(cal.decide_many(rng.uniform(0, 1, 400))) == ABSTAIN)
    assert abstain > 0.8


def test_task_calibrator_falls_back_to_global(tmp_path):
    rng = np.random.default_rng(9)
    s_a, y_a = _scores(rng, 300)
    s_b, y_b = _scores(rng, 12)
    s = np.concatenate([s_a, s_b])
    y = np.concatenate([y_a, y_b])
    tasks = ["insert"] * 300 + ["stack"] * 12
    tc = TaskCalibrator().fit(s, y, tasks, target_error=0.1, min_per_task=30)
    assert "insert" in tc.calibrators and "stack" not in tc.calibrators
    assert tc.decide(0.95, "stack") == SUCCESS and tc.decide(0.02, "unknown_task") == FAILURE
    tc.save(tmp_path / "tc.json")
    again = TaskCalibrator.load(tmp_path / "tc.json")
    assert again.for_task("insert").t_succ == pytest.approx(tc.for_task("insert").t_succ)
    assert set(again.calibrators) == set(tc.calibrators)


# ----------------------------------------------------------------------------- metrics

def test_metrics_on_hand_built_case():
    labels = ["success"] * 4 + ["failure"] * 4
    decisions = [SUCCESS, SUCCESS, FAILURE, ABSTAIN, FAILURE, FAILURE, SUCCESS, ABSTAIN]
    durations = [60.0] * 8  # 8 minutes of robot time
    m = metrics_from_predictions(decisions, labels, durations, tasks=["t1"] * 4 + ["t2"] * 4)
    assert m.n == 8 and m.n_decided == 6 and m.n_abstain == 2
    assert m.hours == pytest.approx(8 * 60 / 3600)
    assert m.abstain_rate.estimate == pytest.approx(0.25)
    assert m.coverage.estimate == pytest.approx(0.75)
    assert m.success_recall.estimate == pytest.approx(2 / 3)
    assert m.failure_recall.estimate == pytest.approx(2 / 3)
    assert m.balanced_accuracy == pytest.approx(2 / 3)
    assert m.failure_precision.estimate == pytest.approx(2 / 3)
    assert m.accuracy.estimate == pytest.approx(4 / 6)
    assert m.success_bias == pytest.approx(0.0)
    assert m.false_alarms == 1
    assert m.false_alarm_rate.estimate == pytest.approx(1 / 4)
    assert m.false_alarms_per_hour == pytest.approx(7.5)
    lo, hi = m.false_alarms_per_hour_bounds
    assert lo <= 7.5 <= hi
    for iv in (m.abstain_rate, m.success_recall, m.failure_recall, m.failure_precision):
        assert iv.lower <= iv.estimate <= iv.upper and iv.method == "Wilson"
    assert set(m.per_task) == {"t1", "t2"} and m.per_task["t1"]["false_alarms"] == 1
    md = m.to_markdown()
    assert "false alarms per hour" in md and "7.50" in md and "| t1 |" in md
    d = m.to_dict()
    assert d["false_alarms_per_hour"] == pytest.approx(7.5) and d["abstain_rate"]["n"] == 8


def test_metrics_success_bias_is_positive_for_a_success_happy_judge():
    labels = ["success"] * 5 + ["failure"] * 5
    decisions = [SUCCESS] * 9 + [FAILURE]
    m = metrics_from_predictions(decisions, labels, [30.0] * 10)
    assert m.success_bias == pytest.approx(0.4)
    assert m.false_alarms == 0 and m.false_alarms_per_hour == 0.0


# ----------------------------------------------------------------------------- hybrid judge

def test_mock_backend_and_hybrid_judge_produce_outcomes():
    rng = np.random.default_rng(21)
    sig_ok = make_episode(rng, True, 1)
    sig_bad = make_episode(rng, False, 2)

    vlm_only = HybridJudge(MockBackend(success_prob=0.9))
    r = vlm_only.judge_episode(sig_ok)
    out = r.to_outcome()
    assert out.judged_by == "vlm" and out.success is True and not out.abstained
    assert out.judge_confidence == pytest.approx(0.9) and out.judge_model == "mock"

    train = make_dataset(30, 80)
    backend = weak_vlm(train + [(sig_ok, "success"), (sig_bad, "failure")], seed=1)
    fusion = fit_fusion(backend, train)
    hybrid = HybridJudge(backend, fusion=fusion)
    r_bad = hybrid.judge_episode(sig_bad)
    out_bad = r_bad.to_outcome()
    assert out_bad.judged_by == "hybrid" and out_bad.success is False
    assert r_bad.failure_class in ("action", "grasp")
    fi = r_bad.to_failure_info()
    assert isinstance(fi, FailureInfo) and fi.failure_class == r_bad.failure_class
    assert "fusion" in out_bad.judge_model
    assert hybrid.judge_episode(sig_ok).to_outcome().success is True

    # a calibrator whose class bands overlap on [0.3, 0.7] (both fit there) abstains in the middle
    cal = Calibrator(t_fail=0.7, t_succ=0.3, target_error=0.05)
    unsure = HybridJudge(MockBackend(success_prob=0.5), calibrator=cal)
    out_unsure = unsure.judge_episode(sig_ok).to_outcome()
    assert out_unsure.abstained and out_unsure.success is None and out_unsure.judged_by == "vlm"
    assert cal.decide(0.9) == SUCCESS and cal.decide(0.1) == FAILURE
    # the empty-band shape (neither fits between 0.3 and 0.7) resolves to the nearer class
    gap = Calibrator(t_fail=0.3, t_succ=0.7)
    assert gap.decide(0.45) == FAILURE and gap.decide(0.55) == SUCCESS
    assert Calibrator(t_fail=0.3, t_succ=0.7, abstain_on_empty=True).decide(0.55) == ABSTAIN

    # a VLM parse error with nothing else to go on abstains even without a calibrator
    broken = MockBackend(verdicts={sig_ok.instruction: parse_verdict("no json here", "mock")})
    assert HybridJudge(broken).judge_episode(sig_ok).decision == ABSTAIN

    with pytest.raises(ValueError):
        HybridJudge(None)


def test_full_pipeline_fit_calibrate_evaluate():
    train = make_dataset(40, 100)
    calib = make_dataset(41, 100, offset=500)
    test = make_dataset(42, 100, offset=900)
    backend = weak_vlm(train + calib + test, seed=3)
    judge = HybridJudge(backend, fusion=fit_fusion(backend, train))
    cal = fit_calibrator(judge, calib, target_error=0.05)
    assert judge.calibrator is cal and cal.fitted
    metrics, results = evaluate(judge, test, return_results=True)
    assert len(results) == 100 and all(r.calibrated for r in results)
    assert metrics.balanced_accuracy is not None and metrics.balanced_accuracy > 0.9
    assert metrics.abstain_rate.estimate < 0.5
    assert metrics.hours == pytest.approx(100 * (T - 1) / HZ / 3600)
    assert metrics.false_alarms_per_hour is not None
    assert "Wilson" in metrics.to_markdown()


# ----------------------------------------------------------------------------- vlm backend

def test_parse_verdict_handles_good_fenced_and_bad_replies():
    good = parse_verdict('{"success_prob": 0.82, "progress": 0.9, "failure_class": null, "final_state_evidence": "cup on plate", "rationale": "done"}', "m")
    assert good.success_prob == pytest.approx(0.82) and good.progress == pytest.approx(0.9)
    assert good.failure_class is None and not good.parse_error and "cup on plate" in good.rationale
    fenced = parse_verdict('Here you go:\n```json\n{"success_prob": 0.1, "progress": 0.3, "failure_class": "grasp"}\n```', "m")
    assert fenced.success_prob == pytest.approx(0.1) and fenced.failure_class == "grasp" and not fenced.parse_error
    odd = parse_verdict('{"success_prob": 1.7, "progress": -2, "failure_class": "made_up"}')
    assert odd.success_prob == 1.0 and odd.progress == 0.0 and odd.failure_class is None
    for bad in ("", "I cannot tell.", '{"progress": 0.2}', '{"success_prob": "high"}', "{not json}"):
        v = parse_verdict(bad, "m")
        assert v.parse_error and v.success_prob == 0.5 and v.progress == 0.5


def test_subsample_indices_and_png_encoder():
    assert subsample_indices(0, 8) == []
    assert subsample_indices(5, 8) == [0, 1, 2, 3, 4]
    idx = subsample_indices(100, 8)
    assert idx[0] == 0 and idx[-1] == 99 and len(idx) == 8 and idx == sorted(idx)
    assert subsample_indices(100, 1) == [99]
    png = encode_png(np.zeros((6, 9, 3), dtype=np.uint8))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    assert (w, h) == (9, 6)
    grey = encode_png(np.zeros((4, 4), dtype=np.uint8))
    assert grey[:8] == b"\x89PNG\r\n\x1a\n"


class _FakeClient:
    """Stands in for anthropic.Anthropic(). Records the request and returns a canned response."""

    def __init__(self, response):
        self.calls = []

        def create(**kw):
            self.calls.append(kw)
            return response

        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def _response(text, stop_reason="end_turn"):
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=1234, output_tokens=56)
    return SimpleNamespace(content=[block], stop_reason=stop_reason, model="claude-opus-5", usage=usage,
                           stop_details=SimpleNamespace(category="other", explanation="x") if stop_reason == "refusal" else None)


def test_anthropic_backend_with_fake_client_parses_and_subsamples():
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(20)]
    fake = _FakeClient(_response('{"success_prob": 0.15, "progress": 0.4, "failure_class": "grasp", "rationale": "gripper empty"}'))
    backend = AnthropicBackend(model="claude-opus-5", n_frames=6, client=fake)
    v = backend.judge(frames, "put the cup on the plate", "cup_plate")
    assert v.success_prob == pytest.approx(0.15) and v.failure_class == "grasp" and not v.parse_error
    assert v.raw["usage"]["input_tokens"] == 1234 and v.model == "claude-opus-5"
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5" and kw["system"] == SYSTEM_PROMPT
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]
    images = [c for c in kw["messages"][0]["content"] if c["type"] == "image"]
    assert len(images) == 6 and images[0]["source"]["media_type"] == "image/png"
    assert "bias toward calling success" in SYSTEM_PROMPT and "FINAL STATE" in SYSTEM_PROMPT
    texts = " ".join(c["text"] for c in kw["messages"][0]["content"] if c["type"] == "text")
    assert "put the cup on the plate" in texts and "frame 20 of 20" in texts

    refused = AnthropicBackend(client=_FakeClient(_response("", stop_reason="refusal")), fallbacks=False)
    r = refused.judge(frames, "x", "t")
    assert r.parse_error and r.success_prob == 0.5 and r.raw["stop_reason"] == "refusal"
    garbage = AnthropicBackend(client=_FakeClient(_response("Looks like a success to me!")))
    g = garbage.judge(frames, "x", "t")
    assert g.parse_error and g.success_prob == 0.5
    assert AnthropicBackend(client=fake).judge(None, "x", "t").parse_error

    class _Boom:
        def __init__(self):
            def create(**kw):
                raise RuntimeError("network down")
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))
            self.messages = SimpleNamespace(create=create)

    e = AnthropicBackend(client=_Boom()).judge(frames, "x", "t")
    assert e.parse_error and "RuntimeError" in e.raw["error"]


# ----------------------------------------------------------------------------- failbench loader

def _write_failbench(tmp_path, n_ok=3, n_bad=3):
    rng = np.random.default_rng(77)
    rows = []
    (tmp_path / "actions").mkdir(exist_ok=True)
    for i in range(n_ok + n_bad):
        success = i < n_ok
        sig = make_episode(rng, success, i, task="insert_peg" if i % 2 else "pick_cup")
        rec = {"episode_id": f"fb{i}", "task": sig.task, "instruction": sig.instruction,
               "label": "success" if success else "failure", "frames": [f"frames/fb{i}/{k:03d}.jpg" for k in range(4)],
               "duration_s": float(sig.timestamps[-1])}
        if i % 3 != 2:
            np.savez(tmp_path / "actions" / f"fb{i}.npz", actions=sig.actions, timestamps=sig.timestamps, states=sig.states, gripper=sig.gripper)
            rec["actions_npz"] = f"actions/fb{i}.npz"
        rows.append(rec)
    path = tmp_path / "failbench.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write("\n")
    return path, rows


def test_failbench_loader_parses_synthetic_jsonl(tmp_path):
    path, rows = _write_failbench(tmp_path)
    eps = load_failbench_jsonl(path)
    assert len(eps) == 6
    labels = [lab for _, lab in eps]
    assert labels == ["success"] * 3 + ["failure"] * 3
    sig0, _ = eps[0]
    assert sig0.episode_id == "fb0" and sig0.actions.shape == (T, CHUNK, D) and sig0.gripper.shape == (T,)
    assert sig0.frames[0].endswith("000.jpg") and str(tmp_path) in sig0.frames[0]
    sig2, _ = eps[2]
    assert sig2.actions is None and sig2.duration() == pytest.approx(rows[2]["duration_s"])
    assert action_stream_features(sig2)["has_actions"] == 0.0
    assert action_stream_features(sig0)["has_actions"] == 1.0

    judge = HybridJudge(MockBackend(success_prob=0.7))
    m = run_failbench(judge, path)
    assert m.n == 6 and m.hours == pytest.approx(sum(r["duration_s"] for r in rows) / 3600)
    assert m.n_abstain == 0 and m.success_bias == pytest.approx(0.5)

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"episode_id": "z", "label": "maybe"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_failbench_jsonl(bad)
    bad.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_failbench_jsonl(bad)


# ----------------------------------------------------------------------------- cli

def test_cli_features_calibrate_run_and_evaluate(tmp_path):
    runner = CliRunner()
    path, rows = _write_failbench(tmp_path)
    npz = tmp_path / "actions" / "fb0.npz"
    res = runner.invoke(judge_app, ["features", str(npz), "--out", str(tmp_path / "f.json"), "--window-s", "2"])
    assert res.exit_code == 0, res.output
    payload = json.loads((tmp_path / "f.json").read_text())
    assert payload["features"]["has_actions"] == 1.0 and len(payload["windows"]) > 3

    rng = np.random.default_rng(5)
    s, y = _scores(rng, 200)
    with (tmp_path / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["score", "label", "task"])
        for i, (si, yi) in enumerate(zip(s, y)):
            w.writerow([f"{si:.4f}", "success" if yi else "failure", "pick_cup" if i % 2 else "insert_peg"])
    res = runner.invoke(judge_app, ["calibrate", str(tmp_path / "scores.csv"), "--out", str(tmp_path / "cal.json"), "--target-error", "0.1"])
    assert res.exit_code == 0, res.output
    cal = json.loads((tmp_path / "cal.json").read_text())
    assert cal["kind"] == "robotruth.judge.TaskCalibrator" and "__default__" in cal["calibrators"]

    res = runner.invoke(judge_app, ["run", str(path), "--backend", "mock", "--calibrator", str(tmp_path / "cal.json"),
                                    "--out", str(tmp_path / "results.jsonl")])
    assert res.exit_code == 0, res.output
    lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 6 and json.loads(lines[0])["decision"] in ("success", "failure", "abstain")

    res = runner.invoke(judge_app, ["evaluate", str(path), "--backend", "mock", "--mock-success-prob", "0.8",
                                    "--out", str(tmp_path / "metrics.md"), "--json-out", str(tmp_path / "metrics.json")])
    assert res.exit_code == 0, res.output
    md = (tmp_path / "metrics.md").read_text()
    assert "false alarms per hour" in md and "Wilson" in md
    assert json.loads((tmp_path / "metrics.json").read_text())["n"] == 6

    res = runner.invoke(judge_app, ["run", str(path), "--backend", "nope"])
    assert res.exit_code != 0
