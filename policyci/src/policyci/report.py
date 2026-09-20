"""The killer screen, as Markdown. Every number carries its evidence: intervals on rates,
a noise floor next to every flip count, and the verdict from the sequential test."""

from __future__ import annotations

from pathlib import Path

from policyci.regression import Diff

VERDICT_TEXT = {
    "B_better": "BETTER",
    "A_better": "WORSE",
    "no_difference_within_margin": "NO MEANINGFUL DIFFERENCE",
    "undecided": "INCONCLUSIVE (more trials needed)",
}


def _iv(iv) -> str:
    return f"{100 * iv.estimate:.1f}% [{100 * iv.lower:.1f}, {100 * iv.upper:.1f}]"


def render_diff(d: Diff, a: dict, b: dict, out_path: str | Path) -> Path:
    lines = []
    lines.append(f"# Policy CI: {d.b_name} vs {d.a_name}")
    lines.append("")
    lines.append(f"Battery `{a['battery_hash'][:16]}` ({d.n} scenarios), task `{a['task']}`, "
                 f"backend `{a['backend_id']}`, evaluator `{a['evaluator_version']}`.")
    lines.append("")
    lines.append("| | " + d.a_name + " | " + d.b_name + " |")
    lines.append("|---|---|---|")
    lines.append(f"| Success | {_iv(d.a_rate)} | {_iv(d.b_rate)} |")
    lines.append("")
    lines.append(f"Paired difference (B - A): {_iv(d.paired)}")
    lines.append("")
    seq = d.sequential
    lines.append(f"**Verdict: {VERDICT_TEXT.get(seq.decision, seq.decision)}** "
                 f"(anytime-valid, n={seq.n}, diff {seq.estimate:+.3f} [{seq.lower:+.3f}, {seq.upper:+.3f}])")
    lines.append("")
    lines.append("## Scenario diff")
    lines.append("")
    if d.noise_flips is not None:
        sig = d.significant_regressions
        lines.append(f"- Newly broken: {len(d.newly_broken)} observed | "
                     f"noise floor {d.noise_flips} (from A-vs-A on the same battery) | "
                     f"beyond noise: {sig}")
    else:
        lines.append(f"- Newly broken: {len(d.newly_broken)} observed | "
                     f"noise floor NOT MEASURED: run the same policy twice first")
    lines.append(f"- Fixed: {len(d.fixed)}")
    lines.append("")
    if d.newly_broken:
        lines.append("### Newly broken scenarios")
        lines.append("")
        lines.append("| scenario | reset seed | A | B failure | replay |")
        lines.append("|---|---|---|---|---|")
        for h in d.newly_broken[:50]:
            rb = b["results"][h]
            idx = rb["index"]
            lines.append(f"| `{h[:12]}` (#{idx}) | see battery | pass | "
                         f"{rb.get('failure') or 'fail'} | videos/{h[:12]}_{d.b_name}.mp4 |")
        if len(d.newly_broken) > 50:
            lines.append(f"| ... and {len(d.newly_broken) - 50} more | | | | |")
        lines.append("")
    if d.fixed:
        lines.append(f"### Fixed scenarios ({len(d.fixed)})")
        lines.append("")
        lines.append(", ".join(f"`{h[:12]}`" for h in d.fixed[:30])
                     + (" ..." if len(d.fixed) > 30 else ""))
        lines.append("")
    if d.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in d.warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("---")
    lines.append(f"Pins hash A `{a['pins_hash'][:16]}` B `{b['pins_hash'][:16]}`. "
                 f"Run manifests `{a['manifest_hash'][:16]}` / `{b['manifest_hash'][:16]}`. "
                 "A number without an interval is not a result.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
