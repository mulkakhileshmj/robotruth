"""Command line for the outcome judge. Exposed as `judge_app`; the main CLI mounts it."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

judge_app = typer.Typer(help="Module 5: calibrated hybrid outcome judge with abstain.", no_args_is_help=True)
console = Console()


def _make_backend(backend: str, model: Optional[str], mock_success_prob: float, n_frames: int):
    from .vlm import DEFAULT_MODEL, AnthropicBackend, MockBackend

    b = backend.strip().lower()
    if b == "mock":
        return MockBackend(success_prob=mock_success_prob, jitter=0.2)
    if b == "anthropic":
        return AnthropicBackend(model=model or DEFAULT_MODEL, n_frames=n_frames)
    if b == "open":
        from .open_vlm import DEFAULT_OPEN_MODEL, OpenVLMBackend
        return OpenVLMBackend(model_id=model or DEFAULT_OPEN_MODEL, n_frames=n_frames)
    raise typer.BadParameter("backend must be mock, anthropic or open")


def _make_judge(backend: str, model: Optional[str], fusion: Optional[Path], calibrator: Optional[Path],
                mock_success_prob: float, n_frames: int):
    from .calibrate import FusionModel, load_any_calibrator
    from .judge import HybridJudge

    fm = FusionModel.load(fusion) if fusion else None
    cal = load_any_calibrator(calibrator) if calibrator else None
    return HybridJudge(_make_backend(backend, model, mock_success_prob, n_frames), fusion=fm, calibrator=cal)


@judge_app.command("features")
def judge_features(npz: Path = typer.Argument(..., help="npz with actions [T,D] or [T,C,D], timestamps [T], optional states, gripper."),
                   out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write features JSON here."),
                   instruction: str = typer.Option("", help="Instruction text (stored, not used by features)."),
                   window_s: Optional[float] = typer.Option(None, help="Also emit windowed features with this window length.")):
    """Compute action-stream features for one episode."""
    from .failbench import load_actions_npz
    from .features import EpisodeSignals, action_stream_features, windowed_features

    arrays = load_actions_npz(npz, None)
    sig = EpisodeSignals(actions=arrays["actions"], timestamps=arrays["timestamps"], states=arrays["states"],
                         gripper=arrays["gripper"], instruction=instruction, episode_id=npz.stem)
    payload: dict = {"episode_id": npz.stem, "features": action_stream_features(sig)}
    if window_s:
        payload["windows"] = windowed_features(sig, window_s)
    text = json.dumps(payload, indent=2)
    if out:
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]wrote[/] {out}")
    else:
        console.print(text)


@judge_app.command("calibrate")
def judge_calibrate(scores: Path = typer.Argument(..., help="CSV with columns score,label[,task]. label is success|failure or 1|0."),
                    out: Path = typer.Option(Path("calibrator.json"), "--out", "-o"),
                    target_error: float = typer.Option(0.05, help="Target error rate among decided episodes."),
                    per_task: bool = typer.Option(True, help="Fit a per-task calibrator when the CSV has a task column."),
                    min_per_task: int = 30):
    """Fit conformal abstain thresholds from held-out scores and labels."""
    from .calibrate import Calibrator, TaskCalibrator

    rows = list(csv.DictReader(scores.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise typer.BadParameter("empty CSV")
    s = [float(r["score"]) for r in rows]
    y = [r["label"] for r in rows]
    tasks = [r.get("task") or "unknown" for r in rows] if "task" in rows[0] else None
    if per_task and tasks is not None:
        cal = TaskCalibrator().fit(s, y, tasks, target_error=target_error, min_per_task=min_per_task)
        rep = cal.calibrators["__default__"]
        extra = f", per-task calibrators: {len(cal.calibrators) - 1}"
    else:
        cal = Calibrator().fit(s, y, target_error=target_error)
        rep = cal
        extra = ""
    cal.save(out)
    console.print(f"[green]wrote[/] {out}  t_fail={rep.t_fail:.3f} t_succ={rep.t_succ:.3f} "
                  f"cal abstain={rep.cal_report.get('abstain_rate', 0):.3f} cal error={rep.cal_report.get('selective_error', 0):.3f}{extra}")


_backend_opt = typer.Option("mock", help="mock | anthropic | open (open-weight VLM on your GPU, no API key)")
_model_opt = typer.Option(None, help="Model id: anthropic backend default claude-opus-5; open backend default Qwen/Qwen2.5-VL-7B-Instruct.")
_fusion_opt = typer.Option(None, help="FusionModel JSON (from robotruth.judge.fit_fusion).")
_cal_opt = typer.Option(None, help="Calibrator JSON (from `judge calibrate`).")
_mock_opt = typer.Option(0.5, help="Constant success probability for the mock backend.")
_frames_opt = typer.Option(8, help="Frames sent to the VLM per episode.")


@judge_app.command("run")
def judge_run(dataset: Path = typer.Argument(..., help="FailBench-style JSONL."),
              backend: str = _backend_opt, model: Optional[str] = _model_opt,
              fusion: Optional[Path] = _fusion_opt, calibrator: Optional[Path] = _cal_opt,
              out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write per-episode results JSONL here."),
              mock_success_prob: float = _mock_opt, n_frames: int = _frames_opt):
    """Judge every episode in the dataset and print a summary. Labels are ignored here."""
    from .failbench import load_failbench_jsonl

    judge = _make_judge(backend, model, fusion, calibrator, mock_success_prob, n_frames)
    episodes = load_failbench_jsonl(dataset)
    results = [judge.judge_episode(sig) for sig, _ in episodes]
    table = Table(title=f"{len(results)} episodes, judge={judge.judge_model}")
    for col in ("episode_id", "task", "p_success", "decision", "failure_class"):
        table.add_column(col)
    for r in results:
        colour = {"success": "green", "failure": "red"}.get(r.decision, "yellow")
        table.add_row(r.episode_id, r.task, f"{r.success_prob:.3f}", f"[{colour}]{r.decision}[/]", r.failure_class or "")
    console.print(table)
    counts = {d: sum(r.decision == d for r in results) for d in ("success", "failure", "abstain")}
    console.print(f"success={counts['success']} failure={counts['failure']} abstain={counts['abstain']}")
    if out:
        with out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), default=_json_default) + "\n")
        console.print(f"[green]wrote[/] {out}")


@judge_app.command("evaluate")
def judge_evaluate(dataset: Path = typer.Argument(..., help="FailBench-style JSONL with labels."),
                   backend: str = _backend_opt, model: Optional[str] = _model_opt,
                   fusion: Optional[Path] = _fusion_opt, calibrator: Optional[Path] = _cal_opt,
                   out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the metrics Markdown here."),
                   json_out: Optional[Path] = typer.Option(None, help="Also write metrics JSON here."),
                   alpha: float = 0.05, mock_success_prob: float = _mock_opt, n_frames: int = _frames_opt):
    """Evaluate the judge against labels: balanced accuracy, abstain rate, false alarms per hour."""
    from .failbench import run_failbench

    judge = _make_judge(backend, model, fusion, calibrator, mock_success_prob, n_frames)
    metrics = run_failbench(judge, dataset, alpha=alpha)
    md = metrics.to_markdown(title=f"Judge evaluation: {dataset.name} ({judge.judge_model})")
    if out:
        out.write_text(md, encoding="utf-8")
        console.print(f"[green]wrote[/] {out}")
    else:
        console.print(md)
    if json_out:
        json_out.write_text(json.dumps(metrics.to_dict(), indent=2, default=_json_default), encoding="utf-8")
        console.print(f"[green]wrote[/] {json_out}")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
