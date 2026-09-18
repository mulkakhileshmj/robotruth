import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from robotruth.schema.lerobot_ingest import LeRobotDataset  # noqa: E402
from robotruth.schema import EpisodeLog, fleet_metrics  # noqa: E402
from robotruth.fingerprint import unit_fingerprint  # noqa: E402


def _fake_lerobot_dataset(root, n_episodes=4, T=60, fps=30, with_success=True, with_intervention=True):
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v2.1", "robot_type": "so100", "fps": fps, "repo_id": "test/so100_demo",
            "features": {"action": {"dtype": "float32", "shape": [6]}, "observation.state": {"dtype": "float32", "shape": [6]},
                         "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]}, "timestamp": {"dtype": "float32", "shape": [1]}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text('{"task_index": 0, "task": "pick the cube"}\n{"task_index": 1, "task": "place the cube"}\n')
    with (root / "meta" / "episodes.jsonl").open("w") as f:
        for e in range(n_episodes):
            f.write(json.dumps({"episode_index": e, "tasks": ["pick the cube" if e % 2 == 0 else "place the cube"], "length": T}) + "\n")
    rng = np.random.default_rng(0)
    cols = {"episode_index": [], "frame_index": [], "timestamp": [], "action": [], "observation.state": [], "task_index": []}
    if with_success:
        cols["next.success"] = []
    if with_intervention:
        cols["is_intervention"] = []
    for e in range(n_episodes):
        t = np.arange(T) / fps
        base = np.sin(2 * np.pi * 0.3 * t)[:, None] * np.linspace(0.2, 0.8, 6)[None, :]
        act = base
        st = np.concatenate([base[:1], base[:-1]]) + rng.normal(0, 0.002, base.shape)
        for i in range(T):
            cols["episode_index"].append(e); cols["frame_index"].append(i); cols["timestamp"].append(float(t[i]))
            cols["action"].append(act[i].astype(np.float32).tolist()); cols["observation.state"].append(st[i].astype(np.float32).tolist())
            cols["task_index"].append(e % 2)
            if with_success:
                cols["next.success"].append(bool(i == T - 1 and e != 1))
            if with_intervention:
                cols["is_intervention"].append(bool(e == 2 and 20 <= i < 30))
    pq.write_table(pa.table(cols), root / "data" / "chunk-000" / "episode_000000.parquet")
    return root


def test_ingest_records_metrics_and_fingerprint(tmp_path):
    root = _fake_lerobot_dataset(tmp_path / "ds")
    ds = LeRobotDataset(root)
    assert ds.repo_id == "test/so100_demo" and ds.fps == 30 and ds.action_dim() == 6
    assert ds.camera_keys() == ["observation.images.top"]
    eps = list(ds.episodes())
    assert len(eps) == 4
    assert eps[0].task == "pick the cube" and eps[1].task == "place the cube"
    assert eps[0].success is True and eps[1].success is False
    assert len(eps[2].interventions()) == 1 and not eps[2].to_record("p", ds.repo_id).autonomous
    recs = ds.to_records()
    m = fleet_metrics(recs)
    assert m.n_judged == 4 and m.success_rate.estimate == pytest.approx(0.75)
    assert m.interventions == 1
    log = EpisodeLog(tmp_path / "ep.jsonl")
    log.extend(recs)
    assert log.validate()[0] == 4
    arrays = LeRobotDataset.excitation_arrays(eps[0])
    assert arrays is not None
    t, cmd, meas = arrays
    fp = unit_fingerprint(t, cmd, meas, "so100_a")
    assert set(fp.joints) == {f"j{k}" for k in range(6)}
    assert fp.joints["j5"].lag_s == pytest.approx(1 / 30, abs=0.02)


def test_ingest_without_labels(tmp_path):
    root = _fake_lerobot_dataset(tmp_path / "ds", with_success=False, with_intervention=False)
    eps = list(LeRobotDataset(root).episodes(max_episodes=2))
    assert len(eps) == 2 and eps[0].success is None and eps[0].intervention_mask is None
    rec = eps[0].to_record("p", "r")
    assert rec.outcome.success is None and rec.autonomous
