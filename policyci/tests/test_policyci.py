"""policyci core tests. No simulator or GPU needed: these cover the parts whose failure
would silently corrupt evidence (identity, comparability, regression math)."""

import json

import numpy as np
import pytest

from policyci.scenario import Scenario, Battery, sample_seed_battery, content_hash
from policyci.regression import (ComparabilityError, check_comparable, count_flips,
                                 diff_runs, load_manifest)
from policyci.report import render_diff


def test_scenario_hash_is_deterministic_and_order_insensitive():
    a = Scenario(index=0, params={"reset_seed": 7, "x": 1})
    b = Scenario(index=5, params={"x": 1, "reset_seed": 7})  # index is not identity
    assert a.hash == b.hash
    c = Scenario(index=0, params={"reset_seed": 8, "x": 1})
    assert a.hash != c.hash


def test_battery_is_reproducible_and_tamperproof(tmp_path):
    b1 = sample_seed_battery("b", "t", "backend", n=50, base_seed=3)
    b2 = sample_seed_battery("b", "t", "backend", n=50, base_seed=3)
    assert b1.battery_hash == b2.battery_hash
    b3 = sample_seed_battery("b", "t", "backend", n=50, base_seed=4)
    assert b1.battery_hash != b3.battery_hash

    p = b1.save(tmp_path / "bat.jsonl")
    loaded = Battery.load(p)
    assert loaded.battery_hash == b1.battery_hash

    # tamper with one scenario -> load refuses
    lines = p.read_text(encoding="utf-8").splitlines()
    d = json.loads(lines[1])
    d["params"]["reset_seed"] += 1
    lines[1] = json.dumps(d)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Battery.load(p)


def _manifest(name, outcomes, battery_hash="bh", pins=None, evaluator="0.1.0"):
    pins = pins or {"mujoco": "3.1", "gym_aloha": "0.1"}
    results = {f"h{i:04d}": {"index": i, "success": bool(o), "gates": {}, "max_reward": 4 if o else 1,
                             "steps": 100, "failure": None if o else "grasp"}
               for i, o in enumerate(outcomes)}
    m = {"kind": "policyci.run", "runner_version": "0.1.0", "run_id": name,
         "task": "t", "backend_id": "backend", "battery_name": "b",
         "battery_hash": battery_hash, "n_scenarios": len(outcomes),
         "policy": {"name": name, "hash": "ph_" + name}, "policy_seed": 0,
         "evaluator_version": evaluator, "pins": pins, "pins_hash": content_hash(pins),
         "wall_clock_s": 1.0, "results": results}
    m["manifest_hash"] = content_hash({k: v for k, v in m.items() if k != "manifest_hash"})
    return m


def test_pin_mismatch_fails_closed():
    a = _manifest("a", [1, 1, 0])
    b = _manifest("b", [1, 0, 0], pins={"mujoco": "3.2", "gym_aloha": "0.1"})
    with pytest.raises(ComparabilityError, match="pins differ"):
        check_comparable(a, b)
    warnings = check_comparable(a, b, allow_pin_mismatch=True)
    assert warnings and "OVERRIDE" in warnings[0]


def test_battery_and_evaluator_mismatch_fail_closed():
    a = _manifest("a", [1, 1])
    with pytest.raises(ComparabilityError, match="battery"):
        check_comparable(a, _manifest("b", [1, 1], battery_hash="other"))
    with pytest.raises(ComparabilityError, match="evaluator"):
        check_comparable(a, _manifest("b", [1, 1], evaluator="0.2.0"))


def test_diff_detects_a_clear_regression():
    rng = np.random.default_rng(0)
    xa = (rng.random(400) < 0.90).astype(int)
    xb = xa.copy()
    # break 15% of previously-passing scenarios
    broken = np.flatnonzero(xa == 1)[:54]
    xb[broken] = 0
    a = _manifest("v18", xa.tolist())
    b = _manifest("v19", xb.tolist())
    d = diff_runs(a, b)
    assert len(d.newly_broken) == 54
    assert len(d.fixed) == 0
    assert d.sequential.decision == "A_better"


def test_noise_floor_contextualizes_flips():
    rng = np.random.default_rng(1)
    xa = (rng.random(400) < 0.9).astype(int)
    # a rerun of the same policy flips a few outcomes both ways
    xa2 = xa.copy()
    flip = rng.choice(400, size=12, replace=False)
    xa2[flip] = 1 - xa2[flip]
    a1, a2 = _manifest("v18", xa.tolist()), _manifest("v18b", xa2.tolist())
    d = diff_runs(a1, a2, noise_ref=(a1, a2))
    assert d.noise_flips == count_flips(a1, a2) // 2
    assert d.sequential.decision in ("undecided", "no_difference_within_margin")
    # observed one-directional flips minus the floor is small
    assert d.significant_regressions <= len(d.newly_broken)


def test_report_renders(tmp_path):
    a = _manifest("v18", [1, 1, 1, 0] * 25)
    b = _manifest("v19", [1, 1, 0, 0] * 25)
    d = diff_runs(a, b, noise_ref=(a, a))
    out = render_diff(d, a, b, tmp_path / "r.md")
    text = out.read_text(encoding="utf-8")
    assert "noise floor" in text
    assert "interval is not a result" in text
    assert "v19 vs v18" in text


# --- evaluator: these build the REAL robotruth objects, which is what a synthetic
# --- manifest test cannot do. The first version of this file lacked them and an
# --- invalid judged_by value reached the GPU box.

@pytest.mark.parametrize("gt,expect_success,expect_class", [
    ({"success": True, "steps": 210, "max_reward": 4, "touched": True, "lifted": True,
      "t_touch": 40, "t_lift": 90, "timed_out": False}, True, None),
    ({"success": False, "steps": 400, "max_reward": 0, "touched": False, "lifted": False,
      "t_touch": None, "t_lift": None, "timed_out": True}, False, "grasp"),
    ({"success": False, "steps": 400, "max_reward": 1, "touched": True, "lifted": False,
      "t_touch": 60, "t_lift": None, "timed_out": True}, False, "grasp"),
    ({"success": False, "steps": 400, "max_reward": 3, "touched": True, "lifted": True,
      "t_touch": 40, "t_lift": 95, "timed_out": True}, False, "timeout"),
    ({"success": False, "steps": 300, "max_reward": 3, "touched": True, "lifted": True,
      "t_touch": 40, "t_lift": 95, "timed_out": False}, False, "action"),
])
def test_evaluate_builds_valid_robotruth_objects(gt, expect_success, expect_class):
    from policyci.evaluator import evaluate
    outcome, failure, gates = evaluate(gt, control_hz=50.0)
    assert outcome.success is expect_success
    assert 0.0 <= outcome.score <= 1.0
    if expect_class is None:
        assert failure is None
    else:
        # the schema sets use_enum_values=True, so this field is a plain string
        assert failure is not None
        assert str(failure.failure_class) == expect_class
        # the subclass must be in the robotruth taxonomy or the model rejects it
        assert failure.subclass is not None
    # an absent gate is never reported as a passed gate
    assert gates["collision"] == "unavailable"
    assert gates["placement_error"] == "unavailable"


def test_evaluator_output_survives_a_full_episode_record():
    """The runner wraps evaluate() in an EpisodeRecord; validate that whole path."""
    from robotruth.schema import EpisodeRecord, Provenance, Timing
    from policyci.evaluator import evaluate
    gt = {"success": False, "steps": 400, "max_reward": 1, "touched": True, "lifted": False,
          "t_touch": 60, "t_lift": None, "timed_out": True}
    outcome, failure, _ = evaluate(gt, 50.0)
    rec = EpisodeRecord(
        episode_id="r:0:abc123def456", task="aloha_transfer_cube", policy="act_v18",
        robot="gym_aloha.transfer_cube.v0", condition="abc123def456", pair_id="abc123def456",
        outcome=outcome, failure=failure,
        timing=Timing(duration_s=33.0, timeout_s=8.0, control_hz=50.0),
        provenance=Provenance(policy_name="act_v18", logger="policyci"),
    )
    assert "grasp" in rec.model_dump_json()
    assert rec.outcome.success is False


# --- passport

def test_passport_is_tamper_evident_and_scoped(tmp_path):
    from policyci.passport import build_passport, write_passport, verify_passport
    a = _manifest("v18", [1] * 180 + [0] * 20)
    b = _manifest("v19", [1] * 150 + [0] * 50)
    d = diff_runs(a, b, noise_ref=(a, a))
    p = build_passport(a, b, d)
    path = write_passport(p, tmp_path / "passport.json")
    assert verify_passport(path)

    # it must never imply real-world performance
    assert p["scope"]["evidence"] == "simulation_only"
    assert "do not estimate real-world success" in p["scope"]["statement"]

    # editing a result invalidates the digest
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["results"]["candidate_success"]["estimate"] = 0.99
    path.write_text(json.dumps(edited), encoding="utf-8")
    assert not verify_passport(path)


def test_passport_blocks_a_worse_candidate_and_withholds_approval_without_a_floor():
    from policyci.passport import build_passport
    rng = np.random.default_rng(7)
    xa = (rng.random(300) < 0.9).astype(int)
    xb = xa.copy()
    xb[np.flatnonzero(xa == 1)[:60]] = 0
    a, b = _manifest("v18", xa.tolist()), _manifest("v19", xb.tolist())

    worse = build_passport(a, b, diff_runs(a, b, noise_ref=(a, a)))
    assert worse["decision"]["recommendation"] == "block"

    # no noise floor -> nothing is approved, whatever the counts say
    no_floor = build_passport(a, b, diff_runs(a, b))
    assert no_floor["decision"]["recommendation"] == "insufficient_evidence"
    assert no_floor["results"]["noise_floor_measured"] is False
