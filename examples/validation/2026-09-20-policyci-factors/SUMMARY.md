# Policy CI with named scene factors

2026-09-20, second run. Lambda A10, 30 vCPUs, us-east-1, terminated after the pull. 1,280
episodes, 5 runs, one shared 256-scenario factor battery `5996cc2a571ad4dc`, 31 workers,
about 1 h 40 m. Replay videos (339) stayed local.

The first run could only say "74 scenarios broke". This one asks three things it could not,
because every scenario now carries named coordinates instead of a bare seed:

| factor | range | note |
|---|---|---|
| `cube_x` | 0.0 to 0.2 m | gym-aloha's own training range |
| `cube_y` | 0.4 to 0.6 m | gym-aloha's own training range |
| `cube_yaw_deg` | −30 to +30 deg | **out of distribution**: gym-aloha never rotates the cube |

## The finding: the public checkpoint is brittle to rotation

`lerobot/act_aloha_sim_transfer_cube_human`, unmodified weights, no candidate involved.
89 of 256 scenarios failed. Clustering its own failures, with no instruction to look at
rotation:

| region | inside | elsewhere | lift | p |
|---|---|---|---|---|
| `cube_y <= 0.42 and abs_cube_yaw_deg > 12.1` | 14/16 (87.5% [64, 97]) | 75/240 (31.2%) | 2.8x | 1.1e-05 |
| `cube_yaw_deg > 11.9` | 46/77 (59.7% [49, 70]) | 43/179 (24.0%) | 2.5x | 5.9e-08 |
| `cube_y <= 0.42` | 17/26 (65.4% [46, 81]) | 72/230 (31.3%) | 2.1x | 7.9e-04 |
| `abs_cube_yaw_deg > 12.1` | 66/153 (43.1% [36, 51]) | 23/103 (22.3%) | 1.9x | 4.2e-04 |

Past about 12 degrees of cube rotation the failure rate roughly doubles, and in the corner
where the cube sits near the front edge *and* is rotated, it fails 87.5% of the time. The
tool was told to vary three factors and to cluster failures; it was not told that rotation
mattered, and gym-aloha itself never rotates the cube in training or evaluation.

That is an operating limit, and it is actionable without retraining: fixture the part
orientation, or collect data in that region.

Note the baseline scores 65.2% [59.2, 70.8] here against 85.0% on the earlier seed-only
battery. Same policy, harder battery. That gap *is* the out-of-distribution rotation.

## The noise floor now has something to measure

Same weights, Gaussian action noise (sigma 0.01), run twice:

| | |
|---|---|
| one-directional flips (pass to fail) | 19 of 157 passing scenarios |
| measured floor | 12.1% [7.9, 18.1] |
| flips in either direction | 42 |

Two things this settles. The machinery works when variation exists, which the first run
could not show because the cell was bit-deterministic. And halving the 42 two-directional
flips would have given 21 where the true one-directional count is 19, which is the estimate
this release replaced.

## The regression is still caught, now against a real floor

| | act_v18 | act_v19_bias005 |
|---|---|---|
| success | 65.2% [59.2, 70.8] | 4.3% [2.4, 7.5] |

Verdict WORSE, exit 1. 160 newly broken against a measured floor of 19, so 141 beyond noise.

## A subtlety worth knowing

The regression's hotspots landed where rotation is *small* (`cube_yaw_deg <= 11.9`:
129/179 broken). That is correct, not a bug. A scenario can only be *newly broken* if the
baseline passed it, and the baseline already fails at high rotation, so those scenarios are
not eligible. Diff hotspots are conditioned on baseline success. The report now says so.

## A bug this run exposed, fixed

Diffing a policy against its own rerun is the natural way to ask "is this cell
deterministic", and it printed `no noise floor measured` on exactly the comparison that
measures it, because the floor was only computed when a separate `--noise` reference was
passed. `same_policy()` now compares the executable parts of the contract while ignoring the
run label, and a self-comparison is reported as a floor measurement rather than a regression
test.

## Files

- `battery_f1.jsonl` — the factor battery, `5996cc2a571ad4dc`
- `baseline_weakness.md` — the finding above
- `diff_determinism.md`, `diff_noise_pair.md`, `diff_bias005.md` — the three comparisons
- `browser_bias005.html` — the scenario browser
- `merged/<run>/` — per-run manifests and 256 episode records each

Reproduce with `policyci/ops/run_factors_box.sh <dir> 256 6 0.01`.
