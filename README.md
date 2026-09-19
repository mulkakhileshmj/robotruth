<div align="center">

<img src="docs/assets/banner.svg" alt="robotruth: Robot CI for learned robot policies" width="100%"/>

<br/>

[![CI](https://github.com/mulkakhileshmj/robotruth/actions/workflows/ci.yml/badge.svg)](https://github.com/mulkakhileshmj/robotruth/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/mulkakhileshmj/robotruth?color=38bdf8)](https://github.com/mulkakhileshmj/robotruth/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-818cf8.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776ab.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-72%20passing-22c55e.svg)](tests)

**Robot CI for learned robot policies.**
It tells a lab whether a policy change is real - before the robot, the eval, or the launch demo tells them the hard way.

[Why](#why-this-exists) · [What's inside](#whats-inside) · [Install](#install) · [Quickstart](#quickstart) · [Command reference](#command-reference) · [Python API](#python-api) · [Validation](#validated-on-real-data) · [License](#license)

</div>

---

## Why this exists

Robot learning has a measurement problem. These are published findings, not opinions:

| The field today | Source |
|---|---|
| 0 of 13 audited real-robot VLA papers report a confidence interval | PhAIL, arXiv 2605.29710 |
| A 50-trial success rate carries a 20–30 point wide 95% interval | Toyota Research Institute, LBM study |
| Moving a camera or a tote shifts task completion by 22 points — more than the gap between models | PhAIL |
| The same weights with different action-normalization metadata: 28/28 → 2/28 | "Same Weights, Different Robot", arXiv 2606.03724 |
| The same policy on a second, identical robot arm: 98% → 18% | SPACE, arXiv 2606.24049 |
| Video-only VLM success judges cap at 0.77 balanced accuracy, 0.52 on contact-rich tasks | FailBench, arXiv 2609.03611 |

robotruth is the layer that makes robot numbers mean something: **a number without an interval is not a result, and an evaluation of one configuration does not certify another.**

It is a Python library and CLI. It is not a benchmark, not a leaderboard, not a simulator, and not a model. It runs next to your own stack — LeRobot, openpi, GR00T, or anything else — and reads files.

## What's inside

<div align="center">
<img src="docs/assets/pipeline.svg" alt="robotruth pipeline: checkpoints, logs, frames and live loops go in; fail-closed checks, calibrated judgments and honest statistics come out" width="100%"/>
</div>

| Module | Question it answers | Command |
|---|---|---|
| **Contract** | Is the policy you evaluated the policy you deployed? Weights, normalizer statistics, action semantics, control rate, cameras, embodiment — hashed into a manifest, compared fail-closed. | `robotruth contract` |
| **Statistics** | Is checkpoint B really better than A? Intervals on every rate, paired designs, anytime-valid sequential tests, censored time-to-success, Bradley-Terry rankings, claims audits. | `robotruth stats` |
| **Episodes** | What actually happened in each rollout? One record per episode with outcome, interventions, failure class and provenance; fleet metrics nobody publishes (interventions/hour, MTBI, autonomous fraction); MCAP bridge for Foxglove. | `robotruth episodes` |
| **Fingerprint** | Did the cell or the robot unit change under you? Camera/lighting fingerprints from one frame; per-joint lag, backlash, offset and gain from one excitation run. | `robotruth fingerprint` |
| **Judge** | Did the episode succeed? Action-stream features fused with a vision-language model, split-conformal calibration with abstention, false alarms per hour reported. Open-weight VLM backend — **no API key required**. | `robotruth judge` |
| **Guard** | Is the policy failing *right now*? Mahalanobis + chunk-consistency scores with time-uniform conformal thresholds calibrated on your own successful rollouts. Emits ok / slow / handover / stop. | `robotruth guard` |

Every report is written as **Markdown and styled HTML**, and every rate in it carries a confidence interval — the report layer refuses to print one without it.

## Install

```bash
pip install robotruth            # from a release wheel (see Releases) or PyPI when published
```

From source:

```bash
git clone https://github.com/mulkakhileshmj/robotruth.git
cd robotruth
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q    # 72 tests
```

Optional extras:

```bash
pip install "robotruth[open-vlm]"   # judge vision channel on your own GPU (Qwen-VL via transformers)
pip install "robotruth[vlm]"        # judge vision channel via the Claude API
pip install "robotruth[mcap]"       # MCAP bridge for Foxglove / Rerun
```

## Quickstart

**1. Certify that what you evaluated is what you deploy.**

```bash
robotruth contract extract runs/ckpt_0400/pretrained_model -o evaluated.json --role evaluated
robotruth contract extract /robot/current_policy -o deployed.json --role deployed
robotruth contract check evaluated.json deployed.json     # exit 1 on any fatal mismatch
```

The check fails closed: if it cannot *prove* the deployed tuple matches the evaluated one, the evaluation does not certify the deployment. Fields no checkpoint carries on disk (the gripper convention, which of several shipped normalizer statistics is live) are listed as `unverified` and supplied via a JSON overlay.

**2. Compare two policies like you mean it.**

```bash
robotruth stats plan --p-a 0.80 --p-b 0.90                # how many trials you need, before you start
robotruth stats schedule --policies base,cand --conditions pose1,pose2,pose3 --repeats 5
# ... run the schedule, record outcomes ...
robotruth stats compare results.csv --a base --b cand     # exit 1 if the candidate is worse
```

The report: Wilson intervals on every rate, a paired difference over matched trials, an **anytime-valid sequential verdict** (stop the moment it decides — peeking is allowed by construction), censored time-to-success, and the number of trials you would need if the comparison is still noise.

**3. Turn your logs into the fleet metrics nobody publishes.**

```bash
robotruth episodes from-lerobot /data/my_dataset -o episodes.jsonl
robotruth episodes metrics episodes.jsonl --out-dir reports/   # HTML + Markdown
```

**4. Catch drift before it eats your eval.**

```bash
robotruth fingerprint cell top_cam.png --camera top -o cell_ref.json
robotruth fingerprint unit excitation.csv --unit-id arm_a -o unit_ref.json
robotruth fingerprint diff cell_ref.json cell_today.json       # exit 1 on drift beyond tolerance
```

**5. Judge outcomes and monitor live runs.**

```bash
robotruth judge run episodes.jsonl --backend open              # open-weight VLM, your GPU, no key
robotruth guard calibrate nominal_episodes/ -o guard.json      # thresholds from YOUR successful rollouts
robotruth guard replay guard.json episode.npz                  # ok / slow / handover / stop, exit 1 on alert
```

## Command reference

| Command | What it does |
|---|---|
| `robotruth contract extract PATH` | Build an ExecSpec manifest from a checkpoint dir or a probe dir (`--family`, `--role`, `--overlay`) |
| `robotruth contract check EVAL DEPLOY` | Fail-closed comparison of two manifests (`--allow-unknown`, `--out`) |
| `robotruth stats plan` | Required trials for a target comparison (`--p-a`, `--p-b`, `--power`, paired vs independent) |
| `robotruth stats schedule` | Blinded, interleaved A/B/n schedule with pair ids and a separate blinding key |
| `robotruth stats compare CSV` | Full comparison report, Markdown + HTML (`--a`, `--b`, `--margin`, `--alpha`) |
| `robotruth stats audit CSV` | Which reported success counts survive an interval (Newcombe + Fisher per task) |
| `robotruth stats interval K N` | Wilson and Clopper-Pearson interval for k successes of n |
| `robotruth episodes from-lerobot DIR` | Ingest a LeRobot dataset (v2/v3, DROID aliases, success + intervention columns) |
| `robotruth episodes validate LOG` | Schema-check a JSONL episode log |
| `robotruth episodes metrics LOG` | Fleet metrics with intervals; `--out-dir` writes the HTML report |
| `robotruth episodes to-results LOG` | Export to the results CSV `stats compare` consumes |
| `robotruth episodes to-mcap / from-mcap` | Round-trip episodes through MCAP (`/robotruth/episode`) for Foxglove |
| `robotruth episodes taxonomy` | Print the failure taxonomy (12 classes, 60 subclasses) |
| `robotruth fingerprint unit CSV` | Per-joint lag, backlash, RMSE, offset, gain from an excitation log |
| `robotruth fingerprint cell IMAGE` | Marker positions/poses, exposure, sharpness, colour balance from one frame |
| `robotruth fingerprint diff REF CUR` | Drift verdict against evidence-based tolerances, exit 1 on FAIL |
| `robotruth judge features / calibrate / run / evaluate` | Action-stream features; conformal calibration with abstain; judge a dataset; score against labels (balanced accuracy, false alarms/hour) |
| `robotruth guard calibrate / replay / metrics` | Fit scorers + time-uniform thresholds on nominal episodes; replay any episode step-by-step; detection and false-alarm accounting |

Every command that produces a verdict uses its **exit code** (0 pass, 1 fail), so all of it drops straight into CI.

## Python API

```python
from robotruth.contract import extract_spec, check_contract
from robotruth.stats import wilson, sequential_paired_test
from robotruth.schema import EpisodeLog, fleet_metrics
from robotruth.judge import HybridJudge, fit_fusion, fit_calibrator
from robotruth.judge.open_vlm import OpenVLMBackend          # no API key
from robotruth.guard import calibrate_guard, OfflineReplay

spec = extract_spec("runs/ckpt_0400/pretrained_model", role="evaluated")
result = check_contract(spec, extract_spec("/robot/policy", role="deployed"))
print(result.summary())                                       # PASS / FAIL with findings

verdict = sequential_paired_test(a_outcomes, b_outcomes, alpha=0.05)
print(verdict)   # e.g. "B_better after n=212: diff(B-A)=+0.101 [+0.012, +0.190]"
```

## Validated on real data

Everything below was measured by robotruth itself on public data (2026-09-18); the raw bundles live in [`examples/validation/2026-09-18`](examples/validation/2026-09-18).

- **33 public datasets, 1,611 real robot episodes** ingested with zero errors.
- **A real DAgger deployment log**: 129.6 interventions/hour [120.9, 138.7], autonomous fraction 0.000 [0.000, 0.029] — what "assisted autonomy" looks like in numbers.
- **Five different physical SO-100/101 arms separated by fingerprint alone**: backlash 0.107–0.478, lag 101–135 ms.
- **RoboArena re-analysed**: 3,284 real pairwise sessions, 15 policies, Bradley-Terry ranking with bootstrap errors.
- **A 150-trial eval log audited**: 2 of 3 comparisons resolved at 95%; the third — a 10-point gap over 50 trials — is inside the noise.
- **A live policy loop** (ACT in gym-aloha, guard attached to every inference): judge balanced accuracy **0.929** [0.651, 0.987] with zero false alarms; the guard's conformal false-alarm bound held in every run. [Video](examples/validation/2026-09-18/live_guard/act_aloha_live_compat.mp4).
- **Open-weight vision judge, no API key** (Qwen2.5-VL-7B, one A10, 3.5 s/episode): balanced accuracy **0.555** [0.442, 0.663] zero-shot on 140 labeled real UR5 episodes, against the 0.77 frontier-API ceiling — stated plainly, because that gap is exactly what the fusion and calibration layers are for.
- **Found in the wild during our own live test**: lerobot 0.6.1 silently drops an older checkpoint's normalization buffers, taking an 83% policy to 0% with only a log warning. The contract checker fails closed on precisely this.

## Methods and their sources

Wilson / Clopper-Pearson / Newcombe intervals and Fisher tests · empirical-Bernstein confidence sequences (Waudby-Smith & Ramdas, JRSS-B 2023; the idea behind TRI's STEP, RSS 2025) · Kaplan-Meier and log-rank on censored time-to-success (motivated by PhAIL) · Bradley-Terry with task buckets (RoboArena, CoRL 2025) · contract fields per "Same Weights, Different Robot", ROEP, SPACE and camera-conditioning results · episode schema per the ISO/TC 299 WG 16 gap statement · judge per FailBench, ActProbe, VLAConf · guard per FAIL-Detect (RSS 2025), VLA-FAIL, SAFECAST.

## Project layout

```
src/robotruth/contract/      ExecSpec manifests, extractors (LeRobot, openpi, GR00T, custom), fail-closed checker
src/robotruth/stats/         intervals, sequential tests, planning, censored timing, Bradley-Terry, blinded schedules
src/robotruth/schema/        episode records, failure taxonomy, LeRobot ingest, MCAP bridge, fleet metrics
src/robotruth/fingerprint/   robot-unit and workspace-cell fingerprints, drift reports
src/robotruth/judge/         hybrid outcome judge: features, fusion, conformal calibration, open + API VLM backends
src/robotruth/guard/         runtime monitor: scorers, conformal thresholds, hard limits, replay, metrics
ops/                         GPU-box scripts used for the validation runs
examples/                    real-checkpoint probes, manifests, and the full validation bundles
tests/                       72 tests, including simulation checks of interval coverage and false-alarm rates
```

## Contributing & citation

See [CONTRIBUTING.md](CONTRIBUTING.md). The ground rules are short: a rate without an interval is not a result, checks fail closed, adapters read files and never import policy frameworks, and design choices cite their evidence. Cite via [CITATION.cff](CITATION.cff).

## License

[Apache License 2.0](LICENSE).
