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
