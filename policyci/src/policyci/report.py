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
    lines.append("## Run-to-run variation")
    lines.append("")
    if getattr(d, "is_self_comparison", False):
        lines.append("**These two runs execute the same policy.** This is a noise-floor "
                     "measurement, not a regression test: any difference below is "
                     "run-to-run variation by construction.")
        lines.append("")
    if d.floor is None:
        lines.append("**Not measured.** Run the same policy twice over this battery before "
                     "reading anything into the scenario counts below. Without it, a broken "
                     "scenario and a coin flip look identical.")
    elif d.floor.is_deterministic:
        lines.append(f"**This cell is deterministic.** {d.floor.statement}")
        lines.append("")
        lines.append("That is the strong case, not a missing measurement: with no run-to-run "
                     "variation, every scenario difference below is real.")
    else:
        lines.append(f"**This cell is stochastic.** {d.floor.statement}")
    lines.append("")
    lines.append("## Scenario diff")
    lines.append("")
    if d.noise_flips is not None:
        sig = d.significant_regressions
        lines.append(f"- Newly broken: {len(d.newly_broken)} observed | "
                     f"expected from run-to-run variation alone: {d.noise_flips} | "
                     f"beyond that: {sig}")
    else:
        lines.append(f"- Newly broken: {len(d.newly_broken)} observed | "
                     f"noise floor NOT MEASURED: run the same policy twice first")
    lines.append(f"- Fixed: {len(d.fixed)}")
    lines.append("")

    lines.append("## Where the failures concentrate")
    lines.append("")
    lines.append("_A scenario can only be newly broken if the baseline passed it, so these "
                 "regions are conditioned on baseline success. A region where the baseline "
                 "already fails cannot appear here, however weak the candidate is there. To "
                 "find the baseline's own weak regions, cluster its failures directly._")
    lines.append("")
    if not d.hotspots:
        lines.append("_No region of the scene space concentrates these failures beyond "
                     "chance, or the battery carries no named factors. A seed-only battery "
                     "cannot describe a region; sample one with `--factors`._")
    else:
        lines.append("| region | inside | elsewhere | lift | p |")
        lines.append("|---|---|---|---|---|")
        for h in d.hotspots:
            lines.append(
                f"| `{h.description}` | {h.n_broken_inside}/{h.n_inside} "
                f"({100 * h.inside_rate.estimate:.1f}% [{100 * h.inside_rate.lower:.0f}, "
                f"{100 * h.inside_rate.upper:.0f}]) | {h.n_broken_outside}/{h.n_outside} "
                f"({100 * h.outside_rate.estimate:.1f}%) | {h.lift:.1f}x | {h.p_value:.1e} |")
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
