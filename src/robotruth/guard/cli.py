"""`robotruth guard` commands. The main CLI mounts `guard_app`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

guard_app = typer.Typer(help="Runtime failure monitor: calibrate on successful rollouts, replay, account for false alarms.",
                        no_args_is_help=True)
console = Console()


def _episodes_in(directory: Path) -> list[Path]:
    files = sorted(p for p in Path(directory).glob("*.npz"))
    if not files:
        raise typer.BadParameter(f"no .npz episodes in {directory}")
    return files


@guard_app.command("calibrate")
def guard_calibrate(nominal_dir: Path = typer.Argument(..., help="Directory of nominal (successful) episode npz files."),
                    out: Path = typer.Option(Path("guard.json"), "--out", "-o"),
                    benign_dir: Optional[Path] = typer.Option(None, help="Directory of benign-shift episodes (SAFECAST contrast set)."),
                    alpha: float = typer.Option(0.05, help="Bound on P(any false alarm) per nominal episode."),
                    method: str = typer.Option("max", help="max (FAIL-Detect) | bonferroni (time-binned)."),
                    n_bins: int = typer.Option(4, help="Time bins for the bonferroni method."),
                    split: float = typer.Option(0.5, help="Fraction of nominal episodes used to fit scorers; the rest set thresholds."),
                    patience: int = typer.Option(3, help="Consecutive above-threshold steps per escalation."),
                    hysteresis: float = typer.Option(0.2, help="Release band as a fraction of the threshold."),
                    dt: float = typer.Option(1.0, help="Seconds per policy step when episodes carry no timestamps."),
                    stride: Optional[int] = typer.Option(None, help="Control steps between chunk emissions (for chunk consistency)."),
                    hard_limits: Optional[Path] = typer.Option(None, help="JSON with HardLimits fields."),
                    seed: int = 0):
    """Fit scorers and conformal thresholds from nominal episodes and write a guard json."""
    from robotruth.guard.adapters import load_episode_npz
    from robotruth.guard.monitor import HardLimits, calibrate_guard
    nominal = [load_episode_npz(p) for p in _episodes_in(nominal_dir)]
    benign = [load_episode_npz(p) for p in _episodes_in(benign_dir)] if benign_dir else None
    hl = HardLimits.from_dict(json.loads(hard_limits.read_text(encoding="utf-8"))) if hard_limits else None
    guard, report = calibrate_guard(nominal, benign, alpha=alpha, method=method, n_bins=n_bins, split=split,
                                    patience=patience, hysteresis=hysteresis, dt=dt, hard_limits=hl,
                                    stride=stride, seed=seed)
    guard.save(out)
    console.print(f"[green]wrote[/] {out}")
    console.print(guard.summary())
    console.print(f"fit on {report['n_fit']} episodes, thresholds from {report['n_calibration']} held out; "
                  f"inputs: {', '.join(report['inputs'])}")
    if report.get("warning"):
        console.print(f"[yellow]warning:[/] {report['warning']}")
    if "contrast" in report:
        c = report["contrast"]
        console.print(f"contrast set: {report['n_benign']} benign episodes moved the threshold by "
                      f"{c['threshold_shift']:+.3f} ({100 * c['threshold_shift_relative']:+.1f}%); "
                      f"{100 * c['benign_alarm_fraction_under_nominal']:.0f}% of them would have alarmed before.")


@guard_app.command("replay")
def guard_replay(guard_json: Path = typer.Argument(..., help="Guard json from `guard calibrate`."),
                 episode: Path = typer.Argument(..., help="Episode npz."),
                 out: Optional[Path] = typer.Option(None, "--out", "-o", help="Events jsonl (default: <episode>.events.jsonl)."),
                 truth: Optional[str] = typer.Option(None, help="success | failure (overrides the npz label).")):
    """Run the guard over one recorded episode. Prints a one-line verdict. Exit 1 if it alerted."""
    from robotruth.guard.adapters import OfflineReplay
    from robotruth.guard.monitor import Guard
    guard = Guard.load(guard_json)
    res = OfflineReplay(guard).run(episode, truth=truth)
    out = out or episode.with_suffix(".events.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for ev in res.events:
            f.write(json.dumps(ev.to_dict()) + "\n")
    console.print(res.verdict)
    console.print(f"[green]wrote[/] {out} ({len(res.events)} events)")
    raise typer.Exit(code=1 if res.episode.detected else 0)


@guard_app.command("metrics")
def guard_metrics(guard_json: Path = typer.Argument(..., help="Guard json from `guard calibrate`."),
                  episodes_dir: Path = typer.Argument(..., help="Directory of episode npz files with a 'truth' key."),
                  alpha: float = 0.05,
                  out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the Markdown here."),
                  json_out: Optional[Path] = typer.Option(None, help="Write the metrics as JSON here.")):
    """Detection rate, warning time and false alarms per hour over labelled episodes."""
    from robotruth.guard.adapters import OfflineReplay
    from robotruth.guard.metrics import GuardMetrics
    from robotruth.guard.monitor import Guard
    guard = Guard.load(guard_json)
    replay = OfflineReplay(guard)
    records = [replay.run(p).episode for p in _episodes_in(episodes_dir)]
    m = GuardMetrics.from_episodes(records, alpha=alpha)
    md = m.to_markdown()
    console.print(md)
    if out:
        out.write_text(md, encoding="utf-8")
        console.print(f"[green]wrote[/] {out}")
    if json_out:
        json_out.write_text(json.dumps(m.to_dict(), indent=2), encoding="utf-8")
