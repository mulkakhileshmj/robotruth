# robotruth status

Updated 2026-09-20. Published to https://github.com/mulkakhileshmj/robotruth.

## Rules for this project

- This folder is the only place edits happen for this work. No existing codebase is touched.
- All compute runs on the rented GPU box, never on the local CPU. `bash ops/sync_and_test.sh <remote_dir> [pytest args]` syncs the tree and runs tests there. Results are pulled back into this folder before the box is terminated.
- Model weights never come to this machine. The box hashes them; only configs, headers and hashes come back (examples/probe).
- Public tools only. Methods, not code, are borrowed from prior work.
- A number without an interval is not a result.

## GPU box log

| when | box | cost | what ran | results pulled to |
|---|---|---|---|---|
| 2026-09-18 ~13:40 IST | Lambda 1x A10 24 GB, us-east-1, 132.145.160.192 (terminated) | $1.29/h | checkpoint probe (5 repos, 33 GB, hashed); full test suite; extractor validation; claims audit; judge and guard builds | examples/probe/2026-09-18_lambda_a10, examples/manifests/2026-09-18_lambda_a10, examples/reports |
| 2026-09-18 ~19:45 IST | Lambda 1x A10 24 GB, us-east-1, 129.80.77.108 (terminated) | $1.29/h | 16 public datasets fetched (parquet and metadata only, no video); full suite 72 passed; wheel and sdist built and clean-install verified; end-to-end validation over 13+ LeRobot datasets, BotFails nested sets, SO101 eval logs, RoboArena pairwise sessions | examples/validation/2026-09-18 (33 datasets, 1,611 episodes, RoboArena BT ranking, SO101 audit, box logs); dist/ wheel and sdist |
| 2026-09-18 ~20:30 IST | Lambda 1x A10 24 GB, us-east-1, 129.213.17.113 (terminated) | $1.29/h | openpi pi0_base probe (12 GB orbax hashed); live policy-loop test (ACT in gym-aloha, guard in the loop, judge vs sim truth, two runs); arm video | examples/validation/2026-09-18/final and live_guard |
| 2026-09-18 ~22:40 IST | Lambda 1x A10 24 GB, us-east-1, 129.80.241.17 (terminated) | $1.29/h | 72 tests, 0.1.1 wheel and sdist, open VLM judge benchmark on 140 real UR5 episodes (Qwen2.5-VL-7B) | examples/validation/2026-09-18/vlm_judge, dist/ |
| 2026-09-19 ~09:00 IST | Lambda 1x A10 24 GB, us-east-1, 150.136.95.69 (terminated) | $1.29/h | guard stability over 3 calibration draws by 2 methods (212 live episodes); fused judge on 323 BotFails and 103 ur5fail episodes; 72 tests; arm video | examples/validation/2026-09-19 |
| 2026-09-19 ~17:30 IST | Lambda 1x A10 24 GB, us-east-1, 129.213.48.169 | $1.29/h | guard detection with policy-breaking faults (250 live episodes, 3 fault types, 8 calibrate-replay combinations); 72 tests; 0.1.3 wheel | examples/validation/2026-09-19/guard_detection, dist/ |
| 2026-09-19 ~18:30 IST | Lambda 1x A10 24 GB, us-east-1, 129.153.172.211 | $1.29/h | stagnation scorer built and measured twice (500 live episodes across two protocols); 54-dataset cross-dataset benchmark and HTML report; pooled BotFails judge and guard; guard stall video; 74 tests; 0.1.4 wheel | examples/validation/2026-09-19/guard_detection_v2, /benchmark, dist/ |
| 2026-09-20 ~14:40 IST | Lambda 1x A10 24 GB, us-east-1, 129.213.87.96 | $1.29/h | tier 1: 400-episode false-alarm pool, gradual-ramp sweep, guard against real robot logs (own failures and injected faults), fingerprint offset sensitivity and conformal calibration, three second-combo attempts; 77 tests; 0.1.5 wheel | examples/validation/2026-09-20 |

## Done

- Module 1, contract checker: ExecSpec manifest, extractors for lerobot (old and new preprocessor formats), gr00t (per-embodiment metadata), openpi, custom; probe-directory source so weights never move; fail-closed checker; CLI `contract extract | check`.
  Validated on five real public checkpoints (see examples/manifests/2026-09-18_lambda_a10/SUMMARY.md):

  | repo | family | dim | chunk | Hz | norm | stats shipped | what must be recorded at deployment |
  |---|---|---|---|---|---|---|---|
  | lerobot/act_aloha_sim_transfer_cube_human | lerobot | 14 | 100 | 50 | mean_std | yes, inside weights | gripper convention |
  | lerobot/smolvla_base | lerobot | 6 | 50 | unknown | mean_std | yes, for 3 datasets in one file | which dataset stats, control Hz, robot, gripper |
  | lerobot/pi0 | lerobot | 6 | 50 | unknown | mean_std | NO (base model) | stats hash, control Hz, robot, gripper |
  | lerobot/pi05_base | lerobot | 32 | 50 | unknown | quantile | NO (base model) | stats hash, control Hz, robot, gripper |
  | nvidia/GR00T-N1.5-3B | gr00t | 44 (gr1 modalities; padded max 32) | 16 | 20 | quantile | yes, per embodiment tag | embodiment tag, executed steps, gripper |

  Every self-check fails closed until those fields are supplied, which is the point.
- Module 2, statistics: intervals, anytime-valid confidence sequence, sequential paired test, planner, censored time to success, Bradley-Terry, blinded schedule, comparison report, and `stats audit` (pairwise Newcombe intervals and Fisher tests over reported success counts). First audit on public numbers: 5 comparisons, 2 resolved, 3 inside the noise (examples/reports/claims_audit.md).
- Module 3, episode record: schema with failure taxonomy (12 classes, 60 subclasses), interventions, provenance hashes; JSONL log; results export; MCAP bridge (topic /robotruth/episode, jsonschema); fleet metrics with intervals (success, autonomous fraction, interventions per hour with Poisson interval, MTBI, failure Pareto, by policy and by unit). CLI `episodes validate | to-results | metrics | to-mcap | from-mcap | taxonomy`.
- Module 4, fingerprints: unit fingerprint from an excitation log (joint lag and backlash estimator by regression, tracking RMSE, offset, gain) and cell fingerprint from a camera frame (ArUco marker positions and poses, exposure, sharpness, colour balance, clipping) with drift tolerances tied to the evidence (22-point camera shift, SPACE unit variance). CLI `fingerprint unit | cell | diff`.
- 43 tests green on the box.

- Module 5, `judge` (hybrid outcome judge with abstain) and `guard` (runtime monitor): built, 71 tests total at merge; wired into the CLI.
- LeRobot ingest (v2 and v3, DROID aliases, success and intervention columns); validation runner over any folder of datasets.
- Real-data validation (2026-09-18, examples/validation/2026-09-18): 33 public datasets, 1,611 episodes, zero errors. Headlines: 129.6 interventions/hour [120.9, 138.7] on a real DAgger log; unit fingerprints separate five physical SO-100/101 arms (backlash 0.107 to 0.478, lag 101 to 135 ms); RoboArena 3,284 sessions re-ranked with Bradley-Terry; SO101 eval audit resolves 2 of 3 comparisons and flags the third as noise; action-only judge honestly abstains on DROID (semantic failures need the VLM channel).
- v0.1.0 wheel and sdist built on the box, clean-install verified (`robotruth --version` works in a fresh venv).

- Live policy-loop validation (2026-09-18, examples/validation/2026-09-18/final): a real ACT policy ran live in gym-aloha on the box with the guard attached to every step and the judge scoring outcomes against the simulator's truth.
  - The loop itself surfaced the contract failure class in the wild: lerobot 0.6.1 silently drops this checkpoint's normalization buffers on load, turning an 83 percent policy into 0 percent with only a log warning. Reattaching the statistics restored ~87 percent. This is "Same Weights, Different Robot" reproduced by accident.
  - Judge on 24 live episodes (action stream only): balanced accuracy 0.929 [0.651, 0.987], failure recall 6/7, zero false alarms. Contrast with DROID, where it abstains: timeouts here are kinematically visible.
  - Guard: zero false alarms on successes in both runs (the conformal alpha bound held). Detection varied with the calibration draw: run 1 caught 4/4 policy-breaking perturbed episodes at 2.7 to 3.0 s after onset plus 2 natural failures; run 2's draw set a higher threshold and caught 0/7. Honest conclusion: with ~20 calibration episodes the max-method threshold is noisy; more nominal episodes or the Bonferroni method are needed for stable detection.
  - openpi extractor validated on the real pi0_base orbax checkpoint (12 GB hashed; norm_stats for 9 robot assets found and flagged as part of the executable policy).
  - Video of the live arm: examples/validation/2026-09-18/live_guard/act_aloha_live_compat.mp4.

- Open VLM judge benchmarked on real labeled failures (2026-09-18, examples/validation/2026-09-18/vlm_judge): Qwen2.5-VL-7B zero-shot on 140 Guardian ur5fail episodes (69 success, 71 failure): balanced accuracy 0.555 [0.442, 0.663], failure recall 0.676, success recall 0.435, 0 parse errors, 3.5 s per episode on an A10. No API key involved. Next lift: fuse with action features and calibrate, and try larger open models.

## 2026-09-19 experiments (examples/validation/2026-09-19)

Two gaps from the 0.1.1 release were attacked on one A10. One closed, one is now properly characterised and still open.

**Fused judge: closed, and it works.** BotFails, 323 real episodes (178 success, 145 failure) across 20 tasks, stratified 40/30/30 splits, all three variants conformally calibrated at a 0.10 target selective error:

| judge | coverage | balanced accuracy on decided | false alarms per hour |
|---|---|---|---|
| action-stream only | 0.000 | n/a (abstains on everything) | 0.00 |
| VLM only, uncalibrated | 1.000 | 0.614 [0.488, 0.727] | 11.13 [6.03, 19.20] |
| fused, calibrated | 0.258 [0.181, 0.353] | 0.921 [0.617, 0.972] | 2.73 [0.94, 7.41] |

Fusing the action stream with the open-weight vision channel and calibrating turns a judge that answers everything at 0.61 into one that answers a quarter of episodes at 0.92 and abstains on the rest, with a quarter of the false alarms. On ur5fail (103 episodes, frames only, no action channel) the vision channel alone scores 0.550 [0.364, 0.723] and the calibrator abstains on all of it at the same target, which is the correct refusal.

**Guard stability: better understood, not fixed.** Shared evaluation pool of 20 nominal and 12 perturbed live episodes, three independent calibration pools of 60 live episodes each (44 to 49 successes), both methods at alpha 0.05:

| method | alarms on 17 held-out successes, per draw | verdict |
|---|---|---|
| max | 3, 0, 0 | closer to the bound, still not tight |
| bonferroni | 4, 4, 3 | roughly 20 percent, violates alpha 0.05 at this sample size |

Every bonferroni bin was saturated at 44 to 49 calibration episodes, so its guarantee did not hold. `max` stays the default and the caveat is now documented in `guard/conformal.py`. Detection could not be measured: the observation shift only broke the policy in 1 of 12 perturbed episodes, so there was almost nothing to detect. A perturbation that reliably breaks the policy, and more calibration episodes, are both needed before the guard's detection can be claimed.

## 2026-09-19 guard detection (examples/validation/2026-09-19/guard_detection)

The missing measurement. Faults injected into the executed action at t = 3.0 s, all of which broke the task (30/30): freeze (stall), offset (+0.15 rad, miscalibration), noise (sigma 0.2, erratic). Calibration pool 164 nominal live episodes; evaluation on 17 held-out successes and the 30 failures.

| pool | method | false alarms | freeze | offset | noise | median latency |
|---|---|---|---|---|---|---|
| 164 | max | 0/17 | 0/10 | 10/10 | 10/10 | 0.04 s |
| 164 | bonferroni | 0/17 | 0/10 | 10/10 | 9/10 | 0.04 s |
| 45 (3 draws) | max | 2, 0, 0 of 17 | 0/10 | 10/10 each | 10/10 each | 0.04 s |

Verdict: the guard detects distributional faults essentially instantly with a held false-alarm bound at proper calibration size, and is blind to stalls. Closed in 0.1.4, below.

## 2026-09-19 stall blind spot closed (examples/validation/2026-09-19/guard_detection_v2)

Two runs of the same experiment, same protocol, published side by side.

**Run 1, stagnation scorer inside the composite** (`composite_only/`, 163-episode pool): freeze 1/10, offset 10/10, noise 10/10, 0/16 false alarms. Averaging the stall signal with three scorers that see a frozen stream as perfectly nominal dilutes it below threshold. The scorer was right and the architecture was wrong.

**Run 2, stagnation scorer as its own conformal head** (`multi_head/`, 150-episode pool, alpha split 50/50 across the two heads, alarm on the union):

| pool | n | method | false alarms | freeze | offset | noise | median latency |
|---|---|---|---|---|---|---|---|
| full | 150 | max | 0/18 | 10/10 | 10/10 | 10/10 | 0.04 s |
| full | 150 | bonferroni | 0/18 | 10/10 | 10/10 | 10/10 | 0.04 s |
| 45 (3 draws) | 45 | max | 0, 0, 0 of 18 | 10/10 each | 10/10 each | 10/10 each | 0.04 s |
| 45 (3 draws) | 45 | bonferroni | 1, 2, 4 of 18 | 10/10 each | 10/10 each | 10/10 each | 0.04 s |

Pooled detection 30/30 = 1.000 [0.886, 1.000]; false alarms 0.000 [0.000, 0.176] against a 0.05 bound. `max` remains the default: `bonferroni` still leaks at 45 calibration episodes. All 250 rollouts are saved as npz next to the report, so future threshold work replays offline instead of costing GPU hours.

## 2026-09-19 cross-dataset benchmark (examples/validation/2026-09-19/benchmark)

54 public datasets ingested with zero errors: RoboMIND and AgiBotWorld task ports (BAAI-DataCube, LeRobot v3), OpenX conversions (IPEC-COMMUNITY), DROID, BotFails, and the DAgger and HIL-SERL intervention logs. One HTML report (`benchmark.html`) leads with the labelled sets, since that is where identification can be checked against truth, then covers every dataset, guard replay, fingerprints and a per-dataset failure breakdown. Most public sets are demo-only and are reported as unlabelled rather than scored.

## 2026-09-20 tier 1 (examples/validation/2026-09-20)

Aimed at the four gaps in the 0.1.4 envelope. Two closed, one corrected the product claim,
one is still open.

**The guard's claim was too broad.** Against real robot logs it caught 5/145 of BotFails'
own failures and 0/58 of DROID's. Per-head AUROC, which is threshold-free, came out at 0.50
to 0.51 and 0.33 to 0.44, so no threshold would help. A robot that fails a task keeps moving
normally. What the guard detects is an execution fault, and that does transfer to real
robots: stalls and erratic control in 153/153, 72/72 and 136/137 real episodes within 0.1 to
0.2 s, false alarms inside the bound. Constant offsets need about 2 sd, and on the HIL-SERL
corpus were missed at every magnitude. `DriftDetector` catches those at 0.25 sd, so the
division of labour is fingerprint for miscalibration, guard for stalls and erratic control,
vision judge for task failure. Nothing detects a normal-looking robot failing its task from
actions alone.

**False alarms, closed.** 0/338 held-out successes, [0.000, 0.011], against 0.1.4's 0/18
whose interval reached 0.176.

**Gradual faults, closed.** Detection holds at 100% for ramps from 0 to 4 s; the delay tracks
how long the fault takes to leave the envelope rather than being a fixed lag.

**Fingerprint threshold, closed.** Conformal at alpha 0.05 gives 3/270 false alarms against
23/270 for the three-sigma rule, detection of a 0.25 sd offset unchanged at 268/270, and the
three leaks are exactly the datasets where the detector declared itself saturated.

**Second policy and task, still open.** The public insertion checkpoint completes 0/10, a
community PushT checkpoint 1/8, and `lerobot/diffusion_pusht` raises an uncorrectable ECC
error on an A10 and poisons the GPU for every other process. Working public sim checkpoints
are scarce.


## Next

1. PyPI upload; publish LAUNCH_NOTE.md.
2. Anthropic API key, then validate the judge VLM channel on BotFails and the Guardian failure sets (frames are on the box paths in examples/validation).
3. A robot in the loop. Everything real so far is replay: the guard watches recordings and never intervenes, so nothing says what happens when it actually gates a robot.
4. A second working policy and task, which needs a checkpoint that performs its task well enough to calibrate on.
5. First external user. Everything here has been exercised by its author only.
