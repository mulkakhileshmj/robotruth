# robotruth status

Updated 2026-09-18 (evening).

## Rules for this project

- This folder is the only place edits happen for this work. No existing codebase is touched.
- All compute runs on the rented GPU box, never on the local CPU. `bash ops/sync_and_test.sh <remote_dir> [pytest args]` syncs the tree and runs tests there. Results are pulled back into this folder before the box is terminated.
- Model weights never come to this machine. The box hashes them; only configs, headers and hashes come back (examples/probe).
- Public tools only. Methods, not code, are borrowed from prior work.
- A number without an interval is not a result.

## GPU box log

| when | box | cost | what ran | results pulled to |
|---|---|---|---|---|
| 2026-09-18 ~13:40 IST start | Lambda 1x A10 24 GB, us-east-1, 132.145.160.192 | $1.29/h | checkpoint probe (5 repos, 33 GB downloaded, hashed); full test suite; extractor validation; claims audit; parallel builders for judge and guard | examples/probe/2026-09-18_lambda_a10, examples/manifests/2026-09-18_lambda_a10, examples/reports |

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

## In progress (parallel builders on the box)

- Module 5, `judge`: calibrated hybrid outcome judge with abstain (action-stream features plus VLM backend, conformal calibration, false alarms per hour, FailBench loader). Anthropic backend needs a key to be validated.
- `guard`: runtime failure monitor (Mahalanobis plus chunk consistency, time-uniform conformal thresholds, contrast-set calibration, hysteresis, hard limits, offline replay, metrics).

## Next

1. Wire `judge_app` and `guard_app` into the main CLI; full suite on the box; commit.
2. Anthropic API key from the user, then validate the judge on a FailBench-style set.
3. Two identical cheap arms or partner logs to validate the unit fingerprint on real hardware.
4. Terminate the Lambda box once results are local.
