import json
from pathlib import Path

import pytest

from robotruth.contract import ExecSpec, Severity, check_contract, extract_spec


def _fake_lerobot(dir_: Path, weight_bytes: bytes = b"w" * 1024, chunk: int = 50, stats: str = "s1") -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    cfg = {
        "type": "act", "chunk_size": chunk, "n_action_steps": chunk,
        "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "MEAN_STD"},
        "input_features": {
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [6]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }
    (dir_ / "config.json").write_text(json.dumps(cfg))
    (dir_ / "train_config.json").write_text(json.dumps({"dataset": {"fps": 30, "repo_id": "lab/so101_pick"}, "policy": {"type": "act"}}))
    # minimal safetensors: 8-byte header length + header json + data
    header = json.dumps({"__metadata__": {}}).encode()
    (dir_ / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + weight_bytes)
    (dir_ / "stats.json").write_text(json.dumps({"action": {"mean": [0] * 6, "std": [1] * 6, "tag": stats}}))
    return dir_


def test_extract_lerobot_populates_fields(tmp_path):
    d = _fake_lerobot(tmp_path / "ckpt")
    spec = extract_spec(d)
    assert spec.policy.family == "lerobot"
    assert spec.action.dim == 6
    assert spec.action.chunk_size == 50
    assert spec.action.control_hz == 30
    assert spec.action.normalization.type == "mean_std"
    assert spec.action.normalization.stats_sha256
    assert spec.observation.cameras[0].name == "observation.images.top"
    assert spec.observation.state_dim == 6
    assert "action.gripper_convention" in spec.unverified


def test_identical_checkpoint_fails_closed_on_unknown_then_passes_with_overlay(tmp_path):
    d = _fake_lerobot(tmp_path / "ckpt")
    ev = extract_spec(d, role="evaluated")
    de = extract_spec(d, role="deployed")
    res = check_contract(ev, de)
    assert not res.passed  # gripper convention unknown => fail closed
    fatal_fields = {f.field for f in res.by_severity(Severity.FATAL)}
    assert "action.gripper_convention" in fatal_fields
    overlay = {"action": {"gripper_convention": "0_open_1_closed"}, "embodiment": {"robot": "so101"}}
    ev = extract_spec(d, role="evaluated", overlay=overlay)
    de = extract_spec(d, role="deployed", overlay=overlay)
    res = check_contract(ev, de)
    assert res.passed, res.summary()


def test_weights_change_is_fatal(tmp_path):
    ov = {"action": {"gripper_convention": "0_open_1_closed"}, "embodiment": {"robot": "so101"}}
    ev = extract_spec(_fake_lerobot(tmp_path / "a", b"a" * 1024), overlay=ov)
    de = extract_spec(_fake_lerobot(tmp_path / "b", b"b" * 1024), overlay=ov)
    res = check_contract(ev, de)
    assert not res.passed
    assert any(f.field == "policy.weights_sha256" and f.severity == Severity.FATAL for f in res.findings)


def test_normalizer_stats_change_is_fatal(tmp_path):
    ov = {"action": {"gripper_convention": "0_open_1_closed"}, "embodiment": {"robot": "so101"}}
    ev = extract_spec(_fake_lerobot(tmp_path / "a", stats="s1"), overlay=ov)
    de = extract_spec(_fake_lerobot(tmp_path / "b", stats="s2"), overlay=ov)
    res = check_contract(ev, de)
    assert any(f.field == "action.normalization.stats_sha256" and f.severity == Severity.FATAL for f in res.findings)


def test_chunk_size_change_is_fatal_and_unit_change_is_warn(tmp_path):
    ov = {"action": {"gripper_convention": "0_open_1_closed"}, "embodiment": {"robot": "so101", "unit_id": "u1"}}
    ev = extract_spec(_fake_lerobot(tmp_path / "a", chunk=50), overlay=ov)
    ov2 = {"action": {"gripper_convention": "0_open_1_closed"}, "embodiment": {"robot": "so101", "unit_id": "u2"}}
    de = extract_spec(_fake_lerobot(tmp_path / "b", chunk=25), overlay=ov2)
    res = check_contract(ev, de)
    sev = {f.field: f.severity for f in res.findings}
    assert sev["action.chunk_size"] == Severity.FATAL
    assert sev["embodiment.unit_id"] == Severity.WARN


def test_spec_roundtrip_and_fingerprint_stability(tmp_path):
    d = _fake_lerobot(tmp_path / "ckpt")
    spec = extract_spec(d)
    p = spec.to_json(tmp_path / "spec.json")
    again = ExecSpec.from_json(p)
    assert again.fingerprint() == spec.fingerprint()
    assert again.policy.weights_sha256 == spec.policy.weights_sha256


def test_custom_family_marks_everything_unverified(tmp_path):
    d = tmp_path / "mystery"
    d.mkdir()
    (d / "policy.pt").write_bytes(b"x" * 100)
    spec = extract_spec(d)
    assert spec.policy.family == "custom"
    assert spec.policy.weights_sha256
    assert "action.chunk_size" in spec.unverified
