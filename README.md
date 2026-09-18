# robotruth

Robot CI for learned robot policies. It tells a lab whether a policy change is real.

The problem it attacks: physical AI cannot cheaply and trustworthily tell whether a change made a robot better or worse. Zero of 13 audited real-robot VLA papers report a confidence interval. A 50-trial success rate carries a 20 to 30 point interval. Moving a camera shifts results by 22 points. The same weights with different normalization metadata go from 28/28 to 2/28. A second identical arm drops a policy from 98 to 18 percent. See `../physical_ai_research/ROOT_PROBLEM_MEMO.md` for the evidence.

robotruth is a Python library and a command-line tool. It is not a website and not a model.

## Four truths

| Truth | Question | Command | Status |
|---|---|---|---|
| Configuration | Is the policy you evaluated the policy you deployed? | `robotruth contract` | built, validated on 5 public checkpoints |
| Statistical | Is checkpoint B really better than A? | `robotruth stats` | built |
| Outcome | Did the episode succeed, when did it fail, and why? | `robotruth episodes`, `robotruth judge` | episodes built, judge in progress |
| Drift | Did the cell or the robot unit change under you? | `robotruth fingerprint` | built |
| Runtime | Is the policy failing right now? | `robotruth guard` | in progress |

## Install

```bash
uv venv .venv && uv pip install -e ".[dev]"
```

## Use

**Contract.** Extract a manifest from a checkpoint directory (LeRobot, openpi, GR00T or anything with weight files), or from a probe folder produced on a GPU box by `ops/remote_checkpoint_probe.sh` so weights never move:

```bash
robotruth contract extract runs/ckpt_0400/pretrained_model -o eval.json --role evaluated
robotruth contract extract /robot/deployed_policy -o deploy.json --role deployed --overlay deploy_overlay.json
robotruth contract check eval.json deploy.json
```

The check fails closed. If it cannot prove the deployed tuple matches the evaluated one, the evaluation does not certify the deployment. Fields the extractor cannot read (gripper convention always; control rate and robot on base VLAs; which of several shipped normalizer statistics is selected) are listed as unverified and supplied with an overlay JSON or a `robotruth.spec.json` next to the checkpoint. See `examples/manifests/2026-09-18_lambda_a10/SUMMARY.md` for what five real checkpoints do and do not carry.

**Statistics.** Plan, schedule blind, run, compare:

```bash
robotruth stats plan --p-a 0.80 --p-b 0.90
robotruth stats schedule --policies base,cand --conditions pose1,pose2,pose3,pose4 --repeats 5
robotruth stats compare results.csv --a base --b cand
robotruth stats audit claims.csv
```

The comparison report carries a Wilson interval on every rate, a paired difference when `pair_id` is present, an anytime-valid sequential verdict (stop the moment it decides), a censored time-to-success comparison, and how many trials you would need if the result is inconclusive. Exit code 1 if the candidate is worse. `audit` takes reported success counts (policy, task, successes, trials) and says which pairwise comparisons survive an interval.

Results CSV columns: `episode_id, policy, task, success, time_to_success, timeout, pair_id, unit_id, session_id, score`.

**Episodes.** One record per rollout, with outcome, interventions, failure class and provenance hashes:

```bash
robotruth episodes validate episodes.jsonl
robotruth episodes metrics episodes.jsonl        # success, autonomous fraction, interventions/hour, MTBI, failure Pareto
robotruth episodes to-results episodes.jsonl -o results.csv
robotruth episodes to-mcap episodes.jsonl -o episodes.mcap   # topic /robotruth/episode for Foxglove or Rerun
robotruth episodes taxonomy
```

**Fingerprints.** Before a session, fingerprint the cell and the unit; diff against the reference:

```bash
robotruth fingerprint cell top_cam.png --camera top -o cell_ref.json
robotruth fingerprint unit excitation.csv --unit-id fr3_a -o unit_ref.json
robotruth fingerprint diff cell_ref.json cell_today.json
```

Unit fingerprints come from a fixed excitation trajectory (`t, cmd_<joint>, meas_<joint>` columns): tracking error, lag, backlash, offset, gain per joint. Cell fingerprints come from one camera frame with printed ArUco markers: marker positions and poses, exposure, sharpness, colour balance.

## Methods and their sources

- Wilson and Clopper-Pearson intervals; paired Wald, bootstrap and Newcombe differences; Fisher exact test.
- Empirical Bernstein confidence sequences (Waudby-Smith and Ramdas, JRSS-B 2023), the idea behind TRI's STEP sequential policy comparison (RSS 2025, arXiv 2503.10966).
- Kaplan-Meier and log-rank for censored time to success, motivated by PhAIL (arXiv 2605.29710).
- Bradley-Terry with task buckets, motivated by RoboArena (CoRL 2025, arXiv 2506.18123).
- Contract fields follow "Same Weights, Different Robot" (arXiv 2606.03724), ROEP (Sensors 2026), SPACE (arXiv 2606.24049) and the camera-conditioning result (arXiv 2510.02268).
- Episode schema follows the gap named in "Data Standards for Humanoid Robotics" (arXiv 2606.19769); failure taxonomy draws on RoboFAC and "How VLAs (Really) Work".
- Judge design answers FailBench (arXiv 2609.03611), ActProbe (arXiv 2606.08508), VLAConf (arXiv 2605.29605). Guard design follows FAIL-Detect (RSS 2025), VLA-FAIL (arXiv 2606.21386), SAFECAST (arXiv 2608.04246).

## Layout

```
src/robotruth/contract/     module 1: ExecSpec manifest, probe or directory source, extractors, fail-closed checker
src/robotruth/stats/        module 2: intervals, sequential, planning, timing, pairwise, schedule
src/robotruth/schema/       module 3: episode record, taxonomy, JSONL log, MCAP bridge, fleet metrics
src/robotruth/fingerprint/  module 4: unit and cell fingerprints, drift report
src/robotruth/judge/        module 5: hybrid outcome judge with abstain
src/robotruth/guard/        runtime failure monitor
src/robotruth/audit.py      claims audit
src/robotruth/compare.py    the comparison report
src/robotruth/report.py     Markdown and HTML rendering
src/robotruth/cli.py        typer CLI
ops/                        GPU box scripts: checkpoint probe, sync and test, probe extraction
examples/                   demo results, real-checkpoint probes and manifests, reports
tests/                      pytest, including simulation checks of interval coverage
```

## Working rules

All compute runs on a rented GPU box, never on the local CPU: `bash ops/sync_and_test.sh rt_main` syncs the tree and runs pytest there. Model weights never come to the local machine. Results are pulled back before the box is terminated. See STATUS.md for the box log.

Licence: Apache 2.0.
