"""robotruth command line."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from robotruth import __version__
from robotruth.contract import ExecSpec, Severity, check_contract, extract_spec
from robotruth.results import Results

app = typer.Typer(help="Robot CI: contract checks, honest statistics and drift fingerprints for learned robot policies.", no_args_is_help=True)
contract_app = typer.Typer(help="Module 1: executable-policy contract manifests.", no_args_is_help=True)
stats_app = typer.Typer(help="Module 2: honest evaluation statistics.", no_args_is_help=True)
app.add_typer(contract_app, name="contract")
app.add_typer(stats_app, name="stats")
console = Console()


@app.callback()
def _version(version: bool = typer.Option(False, "--version", help="Show version and exit.")):
    if version:
        console.print(f"robotruth {__version__}")
        raise typer.Exit()


@contract_app.command("extract")
def contract_extract(path: Path = typer.Argument(..., help="Checkpoint directory."),
                     out: Path = typer.Option(Path("execspec.json"), "--out", "-o"),
                     family: Optional[str] = typer.Option(None, help="lerobot | openpi | gr00t | custom (auto-detected if omitted)"),
                     role: str = typer.Option("unspecified", help="evaluated | deployed"),
                     overlay: Optional[Path] = typer.Option(None, help="JSON with fields the extractor cannot read (control_hz, gripper_convention, cameras...).")):
    """Build an ExecSpec manifest from a checkpoint directory."""
    ov = json.loads(overlay.read_text(encoding="utf-8")) if overlay else None
    spec = extract_spec(path, family=family, overlay=ov, role=role)
    spec.to_json(out)
    console.print(f"[green]wrote[/] {out}  family={spec.policy.family}  fingerprint={spec.fingerprint()[:16]}")
    if spec.unverified:
        console.print(f"[yellow]unverified fields ({len(spec.unverified)}):[/] " + ", ".join(spec.unverified))
        console.print("Supply them with --overlay or a robotruth.spec.json next to the checkpoint, or the check will fail closed.")


@contract_app.command("check")
def contract_check(evaluated: Path, deployed: Path,
                   allow_unknown: bool = typer.Option(False, help="Downgrade unknown fatal fields to warnings (not recommended)."),
                   out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write a JSON report here.")):
    """Compare the evaluated manifest against the deployed one. Exit code 1 on FAIL."""
    res = check_contract(ExecSpec.from_json(evaluated), ExecSpec.from_json(deployed), allow_unknown=allow_unknown)
    table = Table(title=res.summary())
    for col in ("severity", "field", "evaluated", "deployed", "why"):
        table.add_column(col, overflow="fold")
    colours = {Severity.FATAL: "red", Severity.WARN: "yellow", Severity.INFO: "blue", Severity.OK: "green"}
    for f in sorted(res.findings, key=lambda f: list(Severity).index(f.severity)):
        table.add_row(f"[{colours[f.severity]}]{f.severity.value}[/]", f.field, _short(f.evaluated), _short(f.deployed), f.message)
    console.print(table)
    if out:
        out.write_text(json.dumps({"passed": res.passed, "summary": res.summary(),
                                   "findings": [f.__dict__ | {"severity": f.severity.value} for f in res.findings]},
                                  indent=2, default=str), encoding="utf-8")
    raise typer.Exit(code=0 if res.passed else 1)


def _short(v) -> str:
    s = str(v)
    return s if len(s) <= 24 else s[:10] + "..." + s[-10:]


@stats_app.command("plan")
def stats_plan(p_a: float = typer.Option(..., help="Baseline success rate, e.g. 0.80"),
               p_b: float = typer.Option(..., help="Rate you hope to detect, e.g. 0.90"),
               alpha: float = 0.05, power: float = 0.8,
               rho: float = typer.Option(0.4, help="Within-pair outcome correlation for the paired design.")):
    """How many trials you need before you start."""
    from robotruth.stats.planning import required_trials_two_proportions
    console.print(str(required_trials_two_proportions(p_a, p_b, alpha, power, paired=False)))
    console.print(str(required_trials_two_proportions(p_a, p_b, alpha, power, paired=True, rho=rho)))


@stats_app.command("schedule")
def stats_schedule(policies: str = typer.Option(..., help="Comma-separated policy names."),
                   conditions: str = typer.Option(..., help="Comma-separated condition labels (object pose, lighting...)."),
                   repeats: int = 1, seed: int = 0,
                   out: Path = typer.Option(Path("schedule.csv")),
                   key_out: Path = typer.Option(Path("schedule_key.json"), help="Blinding key; keep it away from the operator.")):
    """Write a blinded, interleaved A/B/n schedule."""
    from robotruth.stats.schedule import interleaved_schedule
    s = interleaved_schedule([p.strip() for p in policies.split(",")], [c.strip() for c in conditions.split(",")], repeats, seed)
    rows = s.to_rows()
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    key_out.write_text(json.dumps(s.key, indent=2), encoding="utf-8")
    console.print(f"[green]wrote[/] {out} ({len(rows)} slots) and blinding key {key_out}")


@stats_app.command("compare")
def stats_compare(results: Path = typer.Argument(..., help="Results CSV (see robotruth.results)."),
                  a: str = typer.Option(..., help="Baseline policy name."),
                  b: str = typer.Option(..., help="Candidate policy name."),
                  alpha: float = 0.05,
                  margin: float = typer.Option(0.0, help="Minimum improvement that counts, e.g. 0.02."),
                  out_dir: Path = typer.Option(Path("reports"))):
    """Is B better than A? Writes a Markdown and HTML report. Exit 1 if B is worse."""
    from robotruth.compare import compare_policies
    res = Results.read_csv(results)
    rep = compare_policies(res, a, b, alpha=alpha, margin=margin)
    md, html = rep.write(out_dir, stem=f"compare_{a}_vs_{b}")
    console.print(rep.to_markdown())
    console.print(f"[green]wrote[/] {md} and {html}")
    raise typer.Exit(code=1 if rep.verdict == "FAIL" else 0)


@stats_app.command("interval")
def stats_interval(k: int, n: int, alpha: float = 0.05):
    """Interval for k successes out of n. Because a rate without an interval is not a result."""
    from robotruth.stats.intervals import clopper_pearson, wilson
    console.print(str(wilson(k, n, alpha)))
    console.print(str(clopper_pearson(k, n, alpha)))


if __name__ == "__main__":
    app()
