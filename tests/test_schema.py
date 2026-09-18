import pytest
from typer.testing import CliRunner

from robotruth.cli import app
from robotruth.schema import (
    EpisodeLog,
    EpisodeRecord,
    FailureClass,
    FailureInfo,
    Intervention,
    InterventionSource,
    Outcome,
    Timing,
    fleet_metrics,
)

runner = CliRunner()


def _rec(i: int, policy: str = "ckpt_a", success: bool = True, interventions=None, fail=None, dur=60.0) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=f"e{i}", task="pick_mug", policy=policy, robot="so101", unit_id=f"u{i % 2}",
        session_id="s1", pair_id=f"p{i}",
        outcome=Outcome(success=success, score=1.0 if success else 0.2, judged_by="human"),
        failure=fail, interventions=interventions or [],
        timing=Timing(duration_s=dur, time_to_success_s=20.0 if success else None, timeout_s=90.0,
                      t_start_utc=f"2026-09-18T10:{i:02d}:00+00:00"),
    )


def test_record_roundtrip_and_taxonomy_validation():
    r = _rec(1, fail=FailureInfo(failure_class=FailureClass.GRASP, subclass="slip", t_first_evidence=12.5))
    again = EpisodeRecord.model_validate_json(r.model_dump_json())
    assert again.failure.subclass == "slip"
    with pytest.raises(ValueError):
        FailureInfo(failure_class=FailureClass.GRASP, subclass="camera_moved_or_drift")


def test_autonomous_flag_and_result_row():
    r = _rec(2, interventions=[Intervention(t_start=10, t_end=15, source=InterventionSource.TELEOP)])
    assert not r.autonomous and r.intervention_seconds == 5.0
    row = r.to_result_row()
    assert row["success"] == 1 and row["pair_id"] == "p2" and row["autonomous"] == 0
    assert _rec(3, interventions=[Intervention(t_start=1, source=InterventionSource.VERBAL)]).autonomous


def test_log_append_iterate_validate_and_export(tmp_path):
    log = EpisodeLog(tmp_path / "ep.jsonl")
    log.extend([_rec(i, success=i % 3 != 0) for i in range(9)])
    assert len(log.records()) == 9
    ok, errors = log.validate()
    assert ok == 9 and not errors
    out = log.to_results_csv(tmp_path / "res.csv")
    text = out.read_text()
    assert text.startswith("episode_id,policy,task,success")
    assert text.count("\n") == 10
    with (tmp_path / "ep.jsonl").open("a") as f:
        f.write('{"episode_id": "bad"}\n')
    ok, errors = log.validate()
    assert ok == 9 and len(errors) == 1


def test_fleet_metrics_numbers():
    recs = [_rec(i, success=(i % 4 != 0), dur=600.0,
                 interventions=[Intervention(t_start=5, t_end=35, source=InterventionSource.TELEOP)] if i % 3 == 0 else [],
                 fail=FailureInfo(failure_class=FailureClass.GRASP, subclass="missed_grasp") if i % 4 == 0 else None)
            for i in range(12)]
    m = fleet_metrics(recs)
    assert m.n_episodes == 12 and m.robot_hours == pytest.approx(2.0)
    assert m.interventions == 4 and m.interventions_per_hour == pytest.approx(2.0)
    assert m.interventions_per_hour_ci[0] < 2.0 < m.interventions_per_hour_ci[1]
    assert m.mean_time_between_interventions_s == pytest.approx(1800.0)
    assert m.autonomous_fraction.estimate == pytest.approx(8 / 12)
    assert m.failure_pareto[0][0] == "grasp"
    md = m.to_markdown()
    assert "Interventions per hour" in md and "u0" in md


def test_mcap_roundtrip(tmp_path):
    pytest.importorskip("mcap")
    log = EpisodeLog(tmp_path / "ep.jsonl")
    log.extend([_rec(i) for i in range(5)])
    mcap_path = log.to_mcap(tmp_path / "ep.mcap")
    back = EpisodeLog.from_mcap(mcap_path, tmp_path / "back.jsonl")
    assert [r.episode_id for r in back] == [f"e{i}" for i in range(5)]


def test_cli_episodes(tmp_path):
    log = EpisodeLog(tmp_path / "ep.jsonl")
    log.extend([_rec(i, success=i % 2 == 0) for i in range(6)])
    r = runner.invoke(app, ["episodes", "validate", str(log.path)])
    assert r.exit_code == 0 and "6 valid" in r.output
    r = runner.invoke(app, ["episodes", "metrics", str(log.path)])
    assert r.exit_code == 0 and "Autonomous fraction" in r.output
    r = runner.invoke(app, ["episodes", "to-results", str(log.path), "-o", str(tmp_path / "r.csv")])
    assert r.exit_code == 0 and (tmp_path / "r.csv").exists()
    r = runner.invoke(app, ["episodes", "taxonomy"])
    assert r.exit_code == 0 and "grasp" in r.output
