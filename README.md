# robotruth

Robot CI for learned robot policies. It tells a lab whether a policy change is real.

The problem it attacks: physical AI cannot cheaply and trustworthily tell whether a change made a robot better or worse. Zero of 13 audited real-robot VLA papers report a confidence interval. A 50-trial success rate carries a 20 to 30 point interval. Moving a camera shifts results by 22 points. The same weights with different normalization metadata go from 28/28 to 2/28. A second identical arm drops a policy from 98 to 18 percent. See `../physical_ai_research/ROOT_PROBLEM_MEMO.md` for the evidence.

robotruth is a Python library and a command-line tool. It is not a website and not a model.

## Three truths

| Truth | Question | Module | Status |
|---|---|---|---|
| Configuration | Is the policy you evaluated the policy you deployed? | `robotruth contract` | built |
| Statistical | Is checkpoint B really better than A? | `robotruth stats` | built |
| Outcome | Did the episode succeed, when did it fail, and why? | `robotruth schema`, `robotruth judge` | next |
| Drift | Did the cell or the robot unit change under you? | `robotruth fingerprint` | planned |

## Install

```bash
uv venv .venv && uv pip install -e ".[dev]"
```

## Use

Extract a manifest from a checkpoint directory (LeRobot, openpi, GR00T or anything with weight files):

```bash
robotruth contract extract runs/ckpt_0400/pretrained_model -o eval.json --role evaluated
robotruth contract extract /robot/deployed_policy -o deploy.json --role deployed
robotruth contract check eval.json deploy.json
```

The check fails closed. If it cannot prove the deployed tuple matches the evaluated one, the evaluation does not certify the deployment. Fields the extractor cannot read (gripper convention, control rate on openpi) are listed as unverified; supply them with `--overlay fields.json` or a `robotruth.spec.json` next to the checkpoint.

Plan before you roll:

```bash
robotruth stats plan --p-a 0.80 --p-b 0.90
```

Write a blinded interleaved schedule, run it, record outcomes in a CSV, then compare:

```bash
robotruth stats schedule --policies base,cand --conditions pose1,pose2,pose3,pose4 --repeats 5
robotruth stats compare results.csv --a base --b cand
```

The report carries a Wilson interval on every rate, a paired difference when `pair_id` is present, an anytime-valid sequential verdict (you may stop the moment it decides), a censored time-to-success comparison, and a statement of how many trials you would have needed if the result is inconclusive. Exit code 1 if the candidate is worse.

Results CSV columns: `episode_id, policy, task, success, time_to_success, timeout, pair_id, unit_id, session_id, score`.

## Methods and their sources

- Wilson and Clopper-Pearson intervals; paired Wald and bootstrap differences.
- Empirical Bernstein confidence sequences (Waudby-Smith and Ramdas, JRSS-B 2023), the same idea behind TRI's STEP sequential policy comparison (RSS 2025, arXiv 2503.10966).
- Kaplan-Meier and log-rank for censored time to success, motivated by PhAIL (arXiv 2605.29710).
- Bradley-Terry with task buckets, motivated by RoboArena (CoRL 2025, arXiv 2506.18123).
- The contract field list follows "Same Weights, Different Robot" (arXiv 2606.03724), ROEP (Sensors 2026), SPACE (arXiv 2606.24049) and the camera-conditioning result (arXiv 2510.02268).

## Layout

```
src/robotruth/contract/   module 1: ExecSpec manifest, extractors, fail-closed checker
src/robotruth/stats/      module 2: intervals, sequential, planning, timing, pairwise, schedule
src/robotruth/results.py  results CSV reader
src/robotruth/compare.py  the comparison report
src/robotruth/report.py   Markdown and HTML rendering
src/robotruth/cli.py      typer CLI
tests/                    pytest, including simulation checks of interval coverage
```

## Roadmap

1. Contract checker (done, adapters need validation on real LeRobot, openpi and GR00T checkpoints).
2. Statistics engine (done).
3. Episode record schema, failure taxonomy, MCAP bridge, Foxglove and Rerun loaders.
4. Cell and unit fingerprint with drift monitor.
5. Calibrated hybrid outcome judge with abstain, validated on FailBench; then `policy-guard`, a runtime monitor for LeRobot and openpi policy servers.

Licence: Apache 2.0.
