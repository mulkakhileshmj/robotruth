"""Turn the RoboArena data dump metadata into robotruth inputs. Runs on the GPU box.

RoboArena (CoRL 2025, arXiv 2506.18123) publishes evaluation sessions where two policies run
the same task in the same scene and an evaluator records a progress score (0 to 100) for each
and a preference. Only the small per-session metadata.yaml files are needed here.

Usage: python ops/roboarena_pairs.py <dump_dir> <out_dir>

Writes:
  pairs.csv            one row per session: task, policy_a, policy_b, score_a, score_b, preference
  results.csv          robotruth results table (score in [0,1], pair_id = session) for `stats compare`
  bradley_terry.md     ranking with bootstrap standard errors and pairwise win probabilities
  audit_claims.csv     per-policy "successes" (progress >= 50) and trials, for `stats audit`
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from robotruth.stats.pairwise import bradley_terry


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_sessions(dump: Path) -> list[dict]:
    out = []
    for meta in sorted(dump.glob("evaluation_sessions/*/metadata.yaml")):
        try:
            d = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        d["_session"] = meta.parent.name
        out.append(d)
    return out


def _policy_fields(d: dict) -> list[tuple[str, float | None, float | None, float | None]]:
    """[(name, partial_success, binary_success, duration)] for slots A, B (and C when present).

    Real schema (DataDump_02-03-2026): d["policies"] = {"A": {"policy_name", "binary_success",
    "partial_success", "duration"}, "B": {...}}; d["preference"] in {"A", "B", ...};
    d["language_instruction"] is the task.
    """
    out = []
    pols = d.get("policies")
    if isinstance(pols, dict):
        for slot in sorted(pols.keys()):
            v = pols[slot] or {}
            if isinstance(v, dict):
                out.append((str(v.get("policy_name") or slot), _num(v.get("partial_success")),
                            _num(v.get("binary_success")), _num(v.get("duration"))))
    return out


def main() -> int:
    dump, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(dump)
    if not sessions:
        print("no sessions found"); return 1
    # Record the raw key layout of the first session so schema drift is visible in the results.
    (out / "first_session_keys.json").write_text(json.dumps(sorted(sessions[0].keys()), indent=1))
    pairs, results, comps = [], [], []
    trials: dict[str, Counter] = defaultdict(Counter)
    skipped = 0
    for d in sessions:
        pols = _policy_fields(d)
        task = str(d.get("language_instruction") or d.get("task") or "unknown")
        loc = str(d.get("evaluation_location") or "unknown")
        pref = d.get("preference")
        if len(pols) < 2:
            skipped += 1
            continue
        (na, sa, ba, da), (nb, sb, bb, db) = pols[0], pols[1]
        pairs.append({"session": d["_session"], "location": loc, "task": task, "policy_a": na, "policy_b": nb,
                      "partial_a": sa, "partial_b": sb, "binary_a": ba, "binary_b": bb, "preference": pref})
        for name, partial, binary, dur in pols[:2]:
            if partial is None and binary is None:
                continue
            s01 = max(0.0, min(1.0, partial if partial is not None else float(binary)))
            succ = int(binary) if binary is not None else int(s01 >= 0.5)
            results.append({"episode_id": f"{d['_session']}:{name}", "policy": name, "task": loc, "success": succ,
                            "time_to_success": "", "timeout": "", "pair_id": d["_session"], "unit_id": loc,
                            "session_id": d["_session"], "score": round(s01, 4)})
            trials[name]["trials"] += 1
            trials[name]["successes"] += succ
            trials[name]["partial_sum"] += s01
        # Outcome for Bradley-Terry: evaluator preference first, else partial success, else tie.
        if isinstance(pref, str) and pref.strip().upper() == "A":
            outcome = 1.0
        elif isinstance(pref, str) and pref.strip().upper() == "B":
            outcome = 0.0
        elif sa is not None and sb is not None and sa != sb:
            outcome = 1.0 if sa > sb else 0.0
        else:
            outcome = 0.5
        comps.append((na, nb, outcome, loc))
    with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys())); w.writeheader(); w.writerows(pairs)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    with (out / "audit_claims.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["policy", "task", "successes", "trials"])
        for name, c in sorted(trials.items()):
            w.writerow([name, "ALL", int(c["successes"]), int(c["trials"])])
    bt = bradley_terry(comps, n_boot=200)
    lines = [f"# RoboArena Bradley-Terry ranking", "", f"{len(sessions)} sessions, {len(comps)} usable pairwise outcomes, {skipped} skipped (unparsed layout).", "",
             "| rank | policy | log-strength | bootstrap se | trials | binary success | mean partial |", "|---|---|---|---|---|---|---|"]
    for i, (name, s, se) in enumerate(bt.ranking(), start=1):
        c = trials.get(name, Counter())
        rate = f"{int(c['successes'])}/{int(c['trials'])}" if c["trials"] else ""
        mp = f"{c['partial_sum']/c['trials']:.2f}" if c["trials"] else ""
        lines.append(f"| {i} | {name} | {s:+.3f} | {se:.3f} | {int(c['trials'])} | {rate} | {mp} |")
    lines += ["", "Pairwise win probability P(row beats column):", "", "| | " + " | ".join(bt.policies) + " |", "|---|" + "---|" * len(bt.policies)]
    for i, p in enumerate(bt.policies):
        lines.append(f"| {p} | " + " | ".join(f"{bt.win_prob[i, j]:.2f}" for j in range(len(bt.policies))) + " |")
    (out / "bradley_terry.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(sessions)} sessions -> {len(comps)} comparisons, {len(results)} result rows, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
