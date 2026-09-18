import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
from tests.test_lerobot_ingest import _fake_lerobot_dataset  # noqa: E402


def _labelled_dataset(root, n_episodes=48, T=80, fps=30):
    """Fake dataset where failures freeze midway, so action features carry signal."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = _fake_lerobot_dataset(root, n_episodes=1, T=T, fps=fps)
    # rewrite data with more episodes and a failure signature
    rng = np.random.default_rng(1)
    cols = {"episode_index": [], "frame_index": [], "timestamp": [], "action": [], "observation.state": [], "task_index": [], "next.success": []}
    for e in range(n_episodes):
        succ = e % 3 != 0
        t = np.arange(T) / fps
        base = np.sin(2 * np.pi * 0.3 * t)[:, None] * np.linspace(0.2, 0.8, 6)[None, :]
        if not succ:
            base[T // 2:] = base[T // 2]  # freeze
        st = np.concatenate([base[:1], base[:-1]]) + rng.normal(0, 0.002, base.shape)
        for i in range(T):
            cols["episode_index"].append(e); cols["frame_index"].append(i); cols["timestamp"].append(float(t[i]))
            cols["action"].append(base[i].astype(np.float32).tolist()); cols["observation.state"].append(st[i].astype(np.float32).tolist())
            cols["task_index"].append(0); cols["next.success"].append(bool(succ and i == T - 1))
    pq.write_table(pa.table(cols), root / "data" / "chunk-000" / "episode_000000.parquet")
    with (root / "meta" / "episodes.jsonl").open("w") as f:
        for e in range(n_episodes):
            f.write(json.dumps({"episode_index": e, "tasks": ["pick the cube"], "length": T}) + "\n")
    return root


def test_validate_real_runner_end_to_end(tmp_path):
    _labelled_dataset(tmp_path / "datasets" / "fake_so100")
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "ops" / "validate_real.py"
    sys.argv = [str(script), str(tmp_path / "datasets"), str(out), "--chunk", "8"]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        assert e.code in (0, None)
    md = (out / "VALIDATION.md").read_text(encoding="utf-8")
    assert "fake_so100" in md and "Fleet metrics" in md and "Unit fingerprint" in md
    assert "Judge on real episodes" in md and "Guard on real episodes" in md
    assert "FAILED" not in md, md
    idx = json.loads((out / "index.json").read_text())
    assert idx["fake_so100"]["episodes"] == 48
    assert idx["fake_so100"]["judge"]["balanced_accuracy"] is not None
