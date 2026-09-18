from pathlib import Path

from typer.testing import CliRunner

from robotruth.audit import Claim, audit_claims, newcombe_diff_ci, read_claims
from robotruth.cli import app

runner = CliRunner()


def test_newcombe_interval_behaves():
    d, lo, hi = newcombe_diff_ci(45, 50, 48, 50)
    assert abs(d - 0.06) < 1e-9 and lo < 0 < hi  # 3 more successes out of 50 is noise
    d, lo, hi = newcombe_diff_ci(53, 300, 131, 300)
    assert lo > 0.15  # RoboChallenge gap is real


def test_audit_counts_resolved_claims(tmp_path):
    claims = [Claim("A", "t", 9, 10), Claim("B", "t", 7, 10), Claim("A", "u", 53, 300), Claim("B", "u", 131, 300)]
    rep = audit_claims(claims)
    md = rep.to_markdown()
    assert "2 pairwise comparisons" in md and "1 resolved" in md


def test_cli_audit_on_example(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "claims_public_2026.csv"
    claims = read_claims(example)
    assert len(claims) == 10
    r = runner.invoke(app, ["stats", "audit", str(example), "--out-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "Pairwise comparisons" in r.output
    assert (tmp_path / "claims_audit.md").exists()
