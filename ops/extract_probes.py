"""Build ExecSpec manifests from every probe directory. Run on the GPU box.

Usage: python ops/extract_probes.py examples/probe/<date> examples/manifests/<date>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from robotruth.contract import check_contract, extract_spec

OVERLAYS = {
    # What the extractor cannot read from disk and a lab would record at deployment time.
    "nvidia__GR00T-N1.5-3B": {"embodiment_tag": "gr1"},
}


def main() -> int:
    probe_root, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for d in sorted(p for p in probe_root.iterdir() if p.is_dir()):
        spec = extract_spec(d, overlay=OVERLAYS.get(d.name), role="evaluated")
        out = out_root / f"{d.name}.execspec.json"
        spec.to_json(out)
        self_check = check_contract(spec, spec)
        summary.append({
            "repo": d.name.replace("__", "/"), "family": spec.policy.family, "name": spec.policy.name,
            "weights_gb": round((spec.policy.weights_bytes or 0) / 1e9, 2), "weights_sha256": (spec.policy.weights_sha256 or "")[:16],
            "action_dim": spec.action.dim, "chunk": spec.action.chunk_size, "n_action_steps": spec.action.n_action_steps,
            "control_hz": spec.action.control_hz, "mode": spec.action.mode, "norm": spec.action.normalization.type,
            "stats": (spec.action.normalization.stats_sha256 or "")[:12] or None, "stats_source": spec.action.normalization.stats_source,
            "cameras": [c.name for c in spec.observation.cameras], "state_dim": spec.observation.state_dim,
            "robot": spec.embodiment.robot, "unverified": spec.unverified, "notes": spec.notes,
            "self_check": self_check.summary(), "fingerprint": spec.fingerprint()[:16],
        })
        print(f"{d.name:45s} {spec.policy.family:8s} unverified={len(spec.unverified):2d} self-check: {self_check.summary()}")
    (out_root / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Extractor validation on real public checkpoints", "",
             "| repo | family | weights GB | dim | chunk | steps | Hz | mode | norm | stats present | cameras | robot | unverified | self-check |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['repo']} | {s['family']} | {s['weights_gb']} | {s['action_dim']} | {s['chunk']} | {s['n_action_steps']} | {s['control_hz']} | {s['mode']} | {s['norm']} | {'yes' if s['stats'] else 'NO'} | {len(s['cameras'])} | {s['robot']} | {len(s['unverified'])} | {s['self_check']} |")
    lines += ["", "## Unverified fields per checkpoint (what a lab must record at deployment)", ""]
    for s in summary:
        lines.append(f"- **{s['repo']}**: {', '.join(s['unverified']) or 'none'}")
        for n in s["notes"]:
            lines.append(f"  - note: {n}")
    (out_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
