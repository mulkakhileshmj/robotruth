import csv
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from robotruth.cli import app
from robotruth.compare import compare_policies
from robotruth.results import Results

runner = CliRunner()


def _write_results(path: Path, n_pairs: int = 80, pa: float = 0.7, pb: float = 0.9, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        task = f"task{i % 3}"
        for pol, p in (("base", pa), ("cand", pb)):
            s = int(rng.uniform() < p)
            rows.append(dict(episode_id=f"{pol}-{i}", policy=pol, task=task, success=s,
                             time_to_success=f"{rng.uniform(5, 25):.1f}" if s else "", timeout=30, pair_id=f"p{i}",
                             unit_id="u1", session_id="s1"))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return path


def test_results_reader_and_pairing(tmp_path):
    res = Results.read_csv(_write_results(tmp_path / "r.csv"))
    assert res.policies() == ["base", "cand"]
    xa, xb, ids = res.paired("base", "cand")
    assert xa.size == xb.size == 80


def test_compare_report_has_intervals_everywhere(tmp_path):
    res = Results.read_csv(_write_results(tmp_path / "r.csv"))
    rep = compare_policies(res, "base", "cand")
    md = rep.to_markdown()
    assert "Wilson" in md and "[" in md
    assert rep.verdict in ("PASS", "WARN", "FAIL")
    assert "Anytime-valid" in md
    html = rep.to_html()
    assert "<table" in html


def test_cli_compare_and_plan_and_interval(tmp_path):
    csv_path = _write_results(tmp_path / "r.csv")
    r = runner.invoke(app, ["stats", "compare", str(csv_path), "--a", "base", "--b", "cand", "--out-dir", str(tmp_path / "rep")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "rep" / "compare_base_vs_cand.html").exists()
    r = runner.invoke(app, ["stats", "plan", "--p-a", "0.8", "--p-b", "0.9"])
    assert r.exit_code == 0 and "trials per policy" in r.output
    r = runner.invoke(app, ["stats", "interval", "45", "50"])
    assert r.exit_code == 0 and "Wilson" in r.output


def test_cli_compare_fails_when_candidate_worse(tmp_path):
    csv_path = _write_results(tmp_path / "r.csv", n_pairs=150, pa=0.9, pb=0.6)
    r = runner.invoke(app, ["stats", "compare", str(csv_path), "--a", "base", "--b", "cand", "--out-dir", str(tmp_path / "rep")])
    assert r.exit_code == 1


def test_cli_schedule(tmp_path):
    r = runner.invoke(app, ["stats", "schedule", "--policies", "base,cand", "--conditions", "c1,c2",
                            "--out", str(tmp_path / "s.csv"), "--key-out", str(tmp_path / "k.json")])
    assert r.exit_code == 0
    assert (tmp_path / "s.csv").exists() and (tmp_path / "k.json").exists()
