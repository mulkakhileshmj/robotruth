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
episodes_app = typer.Typer(help="Module 3: episode records, fleet metrics, MCAP bridge.", no_args_is_help=True)
fingerprint_app = typer.Typer(help="Module 4: cell and robot-unit fingerprints and drift.", no_args_is_help=True)
app.add_typer(contract_app, name="contract")
app.add_typer(stats_app, name="stats")
app.add_typer(episodes_app, name="episodes")
app.add_typer(fingerprint_app, name="fingerprint")
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


@stats_app.command("audit")
def stats_audit(claims_csv: Path = typer.Argument(..., help="CSV: policy, task, successes, trials"),
                alpha: float = 0.05, out_dir: Path = typer.Option(Path("reports"))):
    """Which reported comparisons survive an interval? Pairwise Newcombe intervals and Fisher tests within each task."""
    from robotruth.audit import audit_claims, read_claims
    rep = audit_claims(read_claims(claims_csv), alpha)
    md, html = rep.write(out_dir, stem="claims_audit")
    console.print(rep.to_markdown())
    console.print(f"[green]wrote[/] {md} and {html}")


@stats_app.command("interval")
def stats_interval(k: int, n: int, alpha: float = 0.05):
    """Interval for k successes out of n. Because a rate without an interval is not a result."""
    from robotruth.stats.intervals import clopper_pearson, wilson
    console.print(str(wilson(k, n, alpha)))
    console.print(str(clopper_pearson(k, n, alpha)))


@episodes_app.command("validate")
def episodes_validate(log: Path):
    """Validate a JSONL episode log against the schema."""
    from robotruth.schema import EpisodeLog
    ok, errors = EpisodeLog(log).validate()
    console.print(f"{ok} valid records, {len(errors)} errors")
    for line, err in errors[:20]:
        console.print(f"  [red]line {line}[/]: {err}")
    raise typer.Exit(code=1 if errors else 0)


@episodes_app.command("to-results")
def episodes_to_results(log: Path, out: Path = typer.Option(Path("results.csv"), "--out", "-o")):
    """Export episodes with an outcome to the results CSV used by `robotruth stats compare`."""
    from robotruth.schema import EpisodeLog
    p = EpisodeLog(log).to_results_csv(out)
    console.print(f"[green]wrote[/] {p}")


@episodes_app.command("metrics")
def episodes_metrics(log: Path, alpha: float = 0.05, out: Optional[Path] = typer.Option(None, "--out", "-o")):
    """Fleet metrics: success, autonomous fraction, interventions per hour, MTBI, failure Pareto."""
    from robotruth.schema import EpisodeLog, fleet_metrics
    md = fleet_metrics(EpisodeLog(log), alpha).to_markdown()
    console.print(md)
    if out:
        out.write_text(md, encoding="utf-8")


@episodes_app.command("to-mcap")
def episodes_to_mcap(log: Path, out: Path = typer.Option(Path("episodes.mcap"), "--out", "-o")):
    """Write the log as JSON messages on /robotruth/episode for Foxglove or Rerun."""
    from robotruth.schema import EpisodeLog
    console.print(f"[green]wrote[/] {EpisodeLog(log).to_mcap(out)}")


@episodes_app.command("from-mcap")
def episodes_from_mcap(mcap_file: Path, out: Path = typer.Option(Path("episodes.jsonl"), "--out", "-o")):
    """Extract /robotruth/episode messages from an MCAP file into a JSONL log."""
    from robotruth.schema import EpisodeLog
    log = EpisodeLog.from_mcap(mcap_file, out)
    console.print(f"[green]wrote[/] {log.path} ({len(log.records())} records)")


@episodes_app.command("from-lerobot")
def episodes_from_lerobot(dataset_dir: Path, out: Path = typer.Option(Path("episodes.jsonl"), "--out", "-o"),
                          policy: Optional[str] = typer.Option(None, help="Policy label; defaults to dataset:<repo_id>."),
                          max_episodes: Optional[int] = None):
    """Convert a LeRobot dataset directory into an episode log (success from next.success or next.reward, interventions from flag columns)."""
    from robotruth.schema import EpisodeLog
    from robotruth.schema.lerobot_ingest import LeRobotDataset
    ds = LeRobotDataset(dataset_dir)
    recs = ds.to_records(policy, max_episodes)
    if out.exists():
        out.unlink()
    log = EpisodeLog(out)
    log.extend(recs)
    labelled = sum(r.outcome.success is not None for r in recs)
    console.print(f"[green]wrote[/] {out}: {len(recs)} episodes from {ds.repo_id} ({ds.robot}, {ds.fps:.0f} fps, {len(ds.camera_keys())} cameras); {labelled} with success labels")


@episodes_app.command("taxonomy")
def episodes_taxonomy():
    """Print the failure taxonomy."""
    from robotruth.schema import FAILURE_SUBCLASSES
    for cls, subs in FAILURE_SUBCLASSES.items():
        console.print(f"[bold]{cls}[/]: " + ", ".join(subs))


@fingerprint_app.command("unit")
def fingerprint_unit(excitation_csv: Path, unit_id: str = typer.Option("unknown"), out: Path = typer.Option(Path("unit_fingerprint.json"), "--out", "-o")):
    """Fingerprint a robot unit from a fixed excitation trajectory log (t, cmd_<j>, meas_<j> columns)."""
    from robotruth.fingerprint.unit import read_excitation_csv, unit_fingerprint
    t, cmd, meas = read_excitation_csv(excitation_csv)
    fp = unit_fingerprint(t, cmd, meas, unit_id)
    fp.to_json(out)
    for j, st in fp.joints.items():
        console.print(f"{j}: rmse={st.rmse:.4f} lag={1000*st.lag_s:.0f}ms backlash={st.backlash:.4f} offset={st.steady_error:+.4f} gain={st.gain:.3f}")
    console.print(f"[green]wrote[/] {out} fingerprint={fp.fingerprint[:16]}")


@fingerprint_app.command("cell")
def fingerprint_cell(image: Path, camera: str = typer.Option("cam"), aruco_dict: str = typer.Option("4x4_50"),
                     out: Path = typer.Option(Path("cell_fingerprint.json"), "--out", "-o")):
    """Fingerprint a workspace camera view: ArUco marker positions, exposure, sharpness, colour balance."""
    from robotruth.fingerprint import cell_fingerprint
    fp = cell_fingerprint(image, camera, aruco_dict)
    fp.to_json(out)
    console.print(f"{len(fp.markers)} markers, luma={fp.photometrics.mean_luma:.0f}, sharpness={fp.photometrics.sharpness:.0f}")
    console.print(f"[green]wrote[/] {out} fingerprint={fp.fingerprint[:16]}")


@fingerprint_app.command("diff")
def fingerprint_diff(reference: Path, current: Path, out: Optional[Path] = typer.Option(None, "--out", "-o")):
    """Compare two fingerprints (unit or cell) against drift tolerances. Exit 1 on FAIL."""
    from robotruth.fingerprint import CellFingerprint, UnitFingerprint, diff_cell, diff_unit
    ref_d = json.loads(reference.read_text(encoding="utf-8"))
    if "joints" in ref_d:
        rep = diff_unit(UnitFingerprint.from_json(reference), UnitFingerprint.from_json(current))
    else:
        rep = diff_cell(CellFingerprint.from_json(reference), CellFingerprint.from_json(current))
    table = Table(title=rep.summary())
    for col in ("severity", "metric", "reference", "current", "delta", "why"):
        table.add_column(col, overflow="fold")
    colours = {"fail": "red", "warn": "yellow", "ok": "green"}
    for f in sorted(rep.findings, key=lambda f: ["fail", "warn", "ok"].index(f.severity)):
        table.add_row(f"[{colours[f.severity]}]{f.severity}[/]", f.metric, _short(f.reference), _short(f.current), f"{f.delta:.4g}", f.message)
    console.print(table)
    if out:
        out.write_text(json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8")
    raise typer.Exit(code=0 if rep.passed else 1)


if __name__ == "__main__":
    app()
