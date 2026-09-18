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


def _policy_fields(d: dict) -> list[tuple[str, float | None]]:
    """Return [(policy_name, progress_score)] for A, B (and C when present), tolerant to schema variants."""
    found = []
    # Common layouts: top-level policy_a / policy_b with nested score, or a 'policies' dict/list.
    for key in ("policy_a", "policy_b", "policy_c", "A", "B", "C"):
        v = d.get(key)
        if isinstance(v, dict):
            name = v.get("name") or v.get("policy") or v.get("id")
            score = _num(v.get("progress") or v.get("progress_score") or v.get("score"))
            if name:
                found.append((str(name), score))
        elif isinstance(v, str):
            score = _num(d.get(f"{key}_progress") or d.get(f"progress_{key}") or d.get(f"{key}_score"))
            found.append((v, score))
    if not found and isinstance(d.get("policies"), dict):
        for name, v in d["policies"].items():
            score = _num(v.get("progress") if isinstance(v, dict) else v)
            found.append((str(name), score))
    if not found and isinstance(d.get("policies"), list):
        for v in d["policies"]:
            if isinstance(v, dict):
                found.append((str(v.get("name") or v.get("policy")), _num(v.get("progress") or v.get("score"))))
    return found


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
        task = str(d.get("task") or d.get("language_instruction") or d.get("instruction") or "unknown")
        pref = d.get("preference") or d.get("preferred") or d.get("winner")
        if len(pols) < 2:
            skipped += 1
            continue
        (na, sa), (nb, sb) = pols[0], pols[1]
        pairs.append({"session": d["_session"], "task": task, "policy_a": na, "policy_b": nb, "score_a": sa, "score_b": sb, "preference": pref})
        for name, score in pols[:2]:
            if score is not None:
                s01 = max(0.0, min(1.0, score / 100.0 if score > 1.0 else score))
                results.append({"episode_id": f"{d['_session']}:{name}", "policy": name, "task": task, "success": int(s01 >= 0.5),
                                "time_to_success": "", "timeout": "", "pair_id": d["_session"], "unit_id": "", "session_id": d["_session"], "score": round(s01, 4)})
                trials[name]["trials"] += 1
                trials[name]["successes"] += int(s01 >= 0.5)
        # Outcome for Bradley-Terry: preference if given, else higher progress, ties 0.5.
        if isinstance(pref, str) and pref.lower() in ("a", "policy_a", na.lower()):
            outcome = 1.0
        elif isinstance(pref, str) and pref.lower() in ("b", "policy_b", nb.lower()):
            outcome = 0.0
        elif sa is not None and sb is not None:
            outcome = 1.0 if sa > sb else (0.0 if sb > sa else 0.5)
        else:
            continue
        comps.append((na, nb, outcome, task))
    with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys())); w.writeheader(); w.writerows(pairs)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    with (out / "audit_claims.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["policy", "task", "successes", "trials"])
        for name, c in sorted(trials.items()):
            w.writerow([name, "ALL", c["successes"], c["trials"]])
    bt = bradley_terry(comps, n_boot=200)
    lines = [f"# RoboArena Bradley-Terry ranking", "", f"{len(sessions)} sessions, {len(comps)} usable pairwise outcomes, {skipped} skipped (unparsed layout).", "",
             "| rank | policy | log-strength | bootstrap se | trials | progress>=50 rate |", "|---|---|---|---|---|---|"]
    for i, (name, s, se) in enumerate(bt.ranking(), start=1):
        c = trials.get(name, Counter())
        rate = f"{c['successes']}/{c['trials']}" if c["trials"] else ""
        lines.append(f"| {i} | {name} | {s:+.3f} | {se:.3f} | {c['trials']} | {rate} |")
    lines += ["", "Pairwise win probability P(row beats column):", "", "| | " + " | ".join(bt.policies) + " |", "|---|" + "---|" * len(bt.policies)]
    for i, p in enumerate(bt.policies):
        lines.append(f"| {p} | " + " | ".join(f"{bt.win_prob[i, j]:.2f}" for j in range(len(bt.policies))) + " |")
    (out / "bradley_terry.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(sessions)} sessions -> {len(comps)} comparisons, {len(results)} result rows, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
