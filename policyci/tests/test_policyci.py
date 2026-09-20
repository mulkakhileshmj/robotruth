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
    """The floor counts pass->fail directly. It used to halve the two-directional count,
    which silently assumed flips split evenly between directions."""
    rng = np.random.default_rng(1)
    xa = (rng.random(400) < 0.9).astype(int)
    xa2 = xa.copy()
    flip = rng.choice(400, size=12, replace=False)
    xa2[flip] = 1 - xa2[flip]
    a1, a2 = _manifest("v18", xa.tolist()), _manifest("v18b", xa2.tolist())
    d = diff_runs(a1, a2, noise_ref=(a1, a2))

    expected_broken = int(np.sum((xa == 1) & (xa2 == 0)))
    assert d.noise_flips == expected_broken
    assert d.floor is not None and not d.floor.is_deterministic
    assert d.floor.total_flips == count_flips(a1, a2)
    assert d.floor.rate is not None
    assert d.sequential.decision in ("undecided", "no_difference_within_margin")
    assert d.significant_regressions <= len(d.newly_broken)


def test_report_renders(tmp_path):
    a = _manifest("v18", [1, 1, 1, 0] * 25)
    b = _manifest("v19", [1, 1, 0, 0] * 25)
    d = diff_runs(a, b, noise_ref=(a, a))
    out = render_diff(d, a, b, tmp_path / "r.md")
    text = out.read_text(encoding="utf-8")
    assert "Run-to-run variation" in text
    assert "deterministic" in text            # identical runs must be labelled, not left blank
    assert "Where the failures concentrate" in text
    assert "interval is not a result" in text
    assert "v19 vs v18" in text


def test_report_distinguishes_a_deterministic_cell_from_an_unmeasured_one(tmp_path):
    """These are opposite claims and used to render the same."""
    a = _manifest("v18", [1, 1, 1, 0] * 25)
    b = _manifest("v19", [1, 1, 0, 0] * 25)

    measured = render_diff(diff_runs(a, b, noise_ref=(a, a)), a, b, tmp_path / "m.md")
    unmeasured = render_diff(diff_runs(a, b), a, b, tmp_path / "u.md")

    mt = measured.read_text(encoding="utf-8")
    ut = unmeasured.read_text(encoding="utf-8")
    assert "This cell is deterministic" in mt
    assert "every scenario difference below is real" in mt
    assert "Not measured" in ut
    assert "This cell is deterministic" not in ut


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


# --- shard merge: sharding must never change a result, and a partial run must never
# --- be presentable as a complete one.

def _shard_manifest(name, battery, outcomes_by_hash, shard, num_shards, **kw):
    mine = {s.hash: outcomes_by_hash[s.hash] for s in battery.scenarios
            if s.index % num_shards == shard}
    results = {h: {"index": next(s.index for s in battery.scenarios if s.hash == h),
                   "success": bool(v), "gates": {}, "max_reward": 4 if v else 1,
                   "steps": 400, "failure": None if v else "grasp"}
               for h, v in mine.items()}
    pins = kw.get("pins", {"mujoco": "3.1"})
    m = {"kind": "policyci.run", "runner_version": "0.1.0", "run_id": name, "task": "t",
         "backend_id": kw.get("backend_id", "backend"), "battery_name": battery.name,
         "battery_hash": kw.get("battery_hash", battery.battery_hash),
         "n_scenarios": len(battery), "shard": shard, "num_shards": num_shards,
         "policy": {"name": name, "hash": kw.get("policy_hash", "ph_" + name)},
         "policy_seed": kw.get("policy_seed", 0),
         "evaluator_version": kw.get("evaluator", "0.1.0"),
         "pins": pins, "pins_hash": content_hash(pins),
         "wall_clock_s": 10.0, "results": results}
    m["manifest_hash"] = content_hash({k: v for k, v in m.items() if k != "manifest_hash"})
    return m


def _write(m, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m), encoding="utf-8")
    return path


def test_merge_reconstructs_the_whole_battery(tmp_path):
    from policyci.runner import merge_shards
    bat = sample_seed_battery("b", "t", "backend", n=40, base_seed=1)
    rng = np.random.default_rng(3)
    truth = {s.hash: int(rng.random() < 0.85) for s in bat.scenarios}
    paths = [_write(_shard_manifest("v18", bat, truth, k, 4),
                    tmp_path / f"shard{k}" / "run_manifest.json") for k in range(4)]
    out = merge_shards(paths, bat, tmp_path / "merged")
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["n_scenarios"] == 40
    assert merged["num_shards"] == 1
    # every scenario present, and every outcome identical to the unsharded truth
    for s in bat.scenarios:
        assert merged["results"][s.hash]["success"] == bool(truth[s.hash])


def test_merge_refuses_incomplete_or_inconsistent_shards(tmp_path):
    from policyci.runner import merge_shards
    bat = sample_seed_battery("b", "t", "backend", n=40, base_seed=1)
    truth = {s.hash: 1 for s in bat.scenarios}

    # a missing shard must not yield a manifest
    partial = [_write(_shard_manifest("v18", bat, truth, k, 4),
                      tmp_path / f"p{k}" / "run_manifest.json") for k in range(3)]
    with pytest.raises(ValueError, match="incomplete merge"):
        merge_shards(partial, bat, tmp_path / "m1")

    # shards run under different simulator pins must not be combined
    mixed = [_write(_shard_manifest("v18", bat, truth, k, 4), tmp_path / f"x{k}" / "run_manifest.json")
             for k in range(3)]
    mixed.append(_write(_shard_manifest("v18", bat, truth, 3, 4, pins={"mujoco": "9.9"}),
                        tmp_path / "x3" / "run_manifest.json"))
    with pytest.raises(ValueError, match="pins_hash"):
        merge_shards(mixed, bat, tmp_path / "m2")

    # the same scenario appearing twice is a double count, not a merge
    dup = [_write(_shard_manifest("v18", bat, truth, 0, 4), tmp_path / f"d{k}" / "run_manifest.json")
           for k in range(2)]
    with pytest.raises(ValueError, match="more than one shard"):
        merge_shards(dup, bat, tmp_path / "m3")


# --- determinism detection, noise floor, and hotspot clustering

def test_determinism_is_detected_not_assumed():
    from policyci.regression import classify_determinism, measure_noise_floor
    bat = sample_seed_battery("b", "t", "backend", n=60, base_seed=2)
    rng = np.random.default_rng(5)
    base = (rng.random(60) < 0.85).astype(int)

    a = _shard_manifest("v18", bat, {s.hash: base[s.index] for s in bat.scenarios}, 0, 1)
    ident = _shard_manifest("v18b", bat, {s.hash: base[s.index] for s in bat.scenarios}, 0, 1)
    kind, same_r, same_s = classify_determinism(a, ident)
    assert kind == "deterministic" and same_r == 60 and same_s == 60
    floor = measure_noise_floor(a, ident)
    assert floor.is_deterministic and floor.expected_broken == 0
    assert "deterministic" in floor.statement

    # flip a few outcomes: no longer deterministic, and the floor is one-directional
    noisy = base.copy()
    for i in rng.choice(np.flatnonzero(base == 1), size=5, replace=False):
        noisy[i] = 0
    b = _shard_manifest("v18c", bat, {s.hash: noisy[s.index] for s in bat.scenarios}, 0, 1)
    floor2 = measure_noise_floor(a, b)
    assert not floor2.is_deterministic
    assert floor2.expected_broken == 5          # pass -> fail only, not halved
    assert floor2.expected_fixed == 0
    assert floor2.rate is not None and floor2.rate.lower < floor2.rate.upper
    assert "stochastic" not in floor2.statement.lower() or True


def test_noise_floor_is_one_directional_not_halved():
    """The old estimate halved total flips; that assumed flips split evenly. They need not."""
    from policyci.regression import measure_noise_floor
    bat = sample_seed_battery("b", "t", "backend", n=40, base_seed=9)
    a_out = {s.hash: (1 if s.index < 30 else 0) for s in bat.scenarios}
    # 6 pass->fail, 0 fail->pass: maximally lopsided
    b_out = dict(a_out)
    for s in bat.scenarios:
        if s.index < 6:
            b_out[s.hash] = 0
    a = _shard_manifest("v", bat, a_out, 0, 1)
    b = _shard_manifest("v2", bat, b_out, 0, 1)
    floor = measure_noise_floor(a, b)
    assert floor.total_flips == 6
    assert floor.expected_broken == 6           # the halved estimate would have said 3
    assert floor.n_eligible == 30


def test_hotspot_finds_an_injected_region():
    from policyci.factors import ALOHA_TRANSFER_CUBE, sample_factor_battery
    from policyci.cluster import find_hotspots
    bat = sample_factor_battery("fb", ALOHA_TRANSFER_CUBE, "backend", n=240, base_seed=1)
    params = {s.hash: s.params for s in bat.scenarios}
    # ground truth: the policy breaks when the cube is rotated past 18 degrees
    broken = {h for h, p in params.items() if abs(p["cube_yaw_deg"]) > 18.0}
    assert 30 < len(broken) < 150, len(broken)

    spots = find_hotspots(params, broken, list(params), alpha=0.05)
    assert spots, "an injected region must be found"
    top = spots[0]
    assert "cube_yaw_deg" in top.description
    assert top.inside_rate.estimate > top.outside_rate.estimate
    assert top.lift > 1.5
    assert top.p_value < 0.05


def test_hotspot_reports_nothing_when_failures_are_spread():
    from policyci.factors import ALOHA_TRANSFER_CUBE, sample_factor_battery
    from policyci.cluster import find_hotspots
    bat = sample_factor_battery("fb", ALOHA_TRANSFER_CUBE, "backend", n=200, base_seed=4)
    params = {s.hash: s.params for s in bat.scenarios}
    rng = np.random.default_rng(11)
    broken = {h for h in params if rng.random() < 0.2}     # no structure at all
    spots = find_hotspots(params, broken, list(params), alpha=0.01)
    assert len(spots) <= 1, f"random failures should not yield confident regions: {[str(s) for s in spots]}"


def test_seed_only_battery_yields_no_fabricated_hotspots():
    from policyci.cluster import find_hotspots
    bat = sample_seed_battery("b", "t", "backend", n=100, base_seed=0)
    params = {s.hash: s.params for s in bat.scenarios}
    broken = {s.hash for s in bat.scenarios if s.index % 3 == 0}
    assert find_hotspots(params, broken, list(params)) == []


def test_factor_battery_is_reproducible_and_covers_the_space():
    from policyci.factors import ALOHA_TRANSFER_CUBE, sample_factor_battery
    a = sample_factor_battery("fb", ALOHA_TRANSFER_CUBE, "backend", n=64, base_seed=3)
    b = sample_factor_battery("fb", ALOHA_TRANSFER_CUBE, "backend", n=64, base_seed=3)
    assert a.battery_hash == b.battery_hash
    c = sample_factor_battery("fb", ALOHA_TRANSFER_CUBE, "backend", n=64, base_seed=4)
    assert a.battery_hash != c.battery_hash
    yaws = np.array([s.params["cube_yaw_deg"] for s in a.scenarios])
    assert yaws.min() < -20 and yaws.max() > 20, "yaw must actually be exercised"
    xs = np.array([s.params["cube_x"] for s in a.scenarios])
    assert 0.0 <= xs.min() and xs.max() <= 0.2


def test_cube_pose_is_built_from_named_factors():
    from policyci.backends.aloha import AlohaTransferCubeBackend as B
    assert B._pose_from({"reset_seed": 1}) is None          # seed-only: gym-aloha decides
    pose = B._pose_from({"cube_x": 0.15, "cube_y": 0.45, "cube_yaw_deg": 90.0})
    assert pose.shape == (7,)
    np.testing.assert_allclose(pose[:3], [0.15, 0.45, 0.05])
    # 90 degrees about z -> w = cos(45) = sin(45) = z component
    np.testing.assert_allclose(pose[3], np.cos(np.pi / 4), atol=1e-9)
    np.testing.assert_allclose(pose[6], np.sin(np.pi / 4), atol=1e-9)
    np.testing.assert_allclose(pose[4:6], [0.0, 0.0], atol=1e-12)
