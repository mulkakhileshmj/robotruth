# robotruth status

Updated 2026-09-19. Published to https://github.com/mulkakhileshmj/robotruth.

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
| 2026-09-19 ~09:00 IST | Lambda 1x A10 24 GB, us-east-1, 150.136.95.69 | $1.29/h | guard stability over 3 calibration draws by 2 methods (212 live episodes); fused judge on 323 BotFails and 103 ur5fail episodes; 72 tests; arm video | examples/validation/2026-09-19 |

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

## Next

1. GitHub repo (user creates it, we push) and PyPI upload; publish LAUNCH_NOTE.md.
2. Anthropic API key, then validate the judge VLM channel on BotFails and the Guardian failure sets (frames are on the box paths in examples/validation).
3. Guard replay against the HIL-SERL failure episodes with reward signals as truth.
4. Keep the box only while iterating; results are already pulled.
