"""Calibrated drift detection on real logs: does the conformal threshold hold its rate?

The earlier experiment showed the fingerprint separates injected command offsets perfectly,
at a quarter of the standard deviation the runtime guard needs two of. It also showed the
ad-hoc three-sigma rule used to read that signal false-alarming on 5 to 30 percent of clean
episodes, which is not a shipping threshold.

This replaces the rule with `DriftDetector`, which takes a split-conformal quantile of the
worst standardised per-joint deviation, and measures both halves of the trade on real robot
logs: the false-alarm rate on held-out clean episodes against the target alpha, and detection
at each offset magnitude. The naive rule is measured alongside it on the same episodes.

Usage: python ops/fingerprint_calibrate.py <data_root> <out_dir> [alpha]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprint_offset_test import episode_arrays  # noqa: E402

from robotruth.fingerprint import DriftDetector, unit_fingerprint  # noqa: E402
from robotruth.stats.intervals import wilson  # noqa: E402

MAGNITUDES = (0.25, 0.5, 1.0, 2.0)
FIELDS = ("steady_error",)


def shifted(cmd: dict, mag: float) -> dict:
    return {j: v + mag * (float(np.std(v)) + 1e-9) for j, v in cmd.items()}


def naive_rule(fps_fit, fps_test, k: float = 3.0):
    """The three-sigma rule this replaces: mean and standard deviation, per joint."""
    pooled: dict[str, list[float]] = {}
    for fp in fps_fit:
        for j, js in fp.joints.items():
            pooled.setdefault(j, []).append(js.steady_error)
    mu = {j: float(np.mean(v)) for j, v in pooled.items()}
    sd = {j: float(np.std(v)) + 1e-12 for j, v in pooled.items()}
    return [any(abs(js.steady_error - mu[j]) > k * sd[j] for j, js in fp.joints.items() if j in mu)
            for fp in fps_test]


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    out.mkdir(parents=True, exist_ok=True)
    sources = episode_arrays(root)

    doc = ["# Calibrated drift detection on real robot logs", "",
           f"`DriftDetector` at alpha = {alpha:g}, watching per-joint steady-state error, against "
           "the three-sigma rule it replaces. Both are fitted on the same clean episodes and "
           "measured on the same held-out ones, then on the same episodes with a constant "
           "command offset injected.", "",
           "The threshold is a split-conformal quantile of the worst standardised per-joint "
           "deviation, so its false-alarm rate is bounded by alpha under exchangeability alone, "
           "with no assumption that per-joint estimates are Gaussian or independent.", ""]
    index = {}
    for name, eps in sorted(sources.items()):
        n_fit = int(0.6 * len(eps))
        fit_eps, test_eps = eps[:n_fit], eps[n_fit:]
        fit_fps = [unit_fingerprint(*e, f"{name}-fit") for e in fit_eps]
        test_fps = [unit_fingerprint(*e, f"{name}-test") for e in test_eps]

        det = DriftDetector(alpha=alpha, fields=FIELDS).fit(fit_fps)
        fa = sum(det.alarms(fp) for fp in test_fps)
        fa_naive = sum(naive_rule(fit_fps, test_fps))

        rows = {}
        for mag in MAGNITUDES:
            faulted = [unit_fingerprint(t, shifted(cmd, mag), meas, f"{name}-drift")
                       for t, cmd, meas in test_eps]
            rows[f"{mag:g} sd"] = {
                "conformal": sum(det.alarms(fp) for fp in faulted),
                "naive": sum(naive_rule(fit_fps, faulted)),
                "n": len(faulted)}

        doc += [f"## {name}", "",
                f"{len(eps)} episodes: {len(fit_eps)} to fit, {len(test_eps)} held out. "
                f"{det.summary()}", "",
                "| | false alarms (clean) | " + " | ".join(f"offset {k}" for k in rows) + " |",
                "|---|---|" + "---|" * len(rows),
                f"| conformal, alpha {alpha:g} | {fa}/{len(test_fps)} = {wilson(fa, len(test_fps))} | "
                + " | ".join(f"{r['conformal']}/{r['n']}" for r in rows.values()) + " |",
                f"| three-sigma rule | {fa_naive}/{len(test_fps)} = {wilson(fa_naive, len(test_fps))} | "
                + " | ".join(f"{r['naive']}/{r['n']}" for r in rows.values()) + " |", ""]
        index[name] = {"n_episodes": len(eps), "n_fit": len(fit_eps), "n_test": len(test_fps),
                       "alpha": alpha, "threshold": det.threshold_, "saturated": det.saturated_,
                       "false_alarms_conformal": fa, "false_alarms_naive": fa_naive,
                       "false_alarm_interval_conformal": str(wilson(fa, len(test_fps))),
                       "detection": rows}
        print(f"{name}: conformal fa {fa}/{len(test_fps)}, naive fa {fa_naive}/{len(test_fps)}, "
              + ", ".join(f"{k} {v['conformal']}/{v['n']}" for k, v in rows.items()), flush=True)

    doc += ["", "FINGERPRINT_CALIBRATE_COMPLETE"]
    (out / "fingerprint_calibrate.md").write_text("\n".join(doc), encoding="utf-8")
    (out / "fingerprint_calibrate.json").write_text(json.dumps(index, indent=1, default=str),
                                                    encoding="utf-8")
    print("FINGERPRINT_CALIBRATE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
