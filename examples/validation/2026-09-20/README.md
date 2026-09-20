# Tier 1 production-readiness experiments, 2026-09-20

Five experiments aimed at the gaps the 0.1.4 envelope left open: one policy on one task,
abrupt faults only, a false-alarm bound resting on 18 episodes, and no real-robot evidence
for the runtime guard. One A10, about six hours.

The short version: the guard's headline claim was too broad and is now corrected. It detects
execution faults, meaning the robot has stopped behaving like itself, and that capability
does transfer to real robot logs. It does not detect task failures, where the robot moves
normally and still fails, and no threshold change would make it. That job belongs to the
vision judge, and constant miscalibration belongs to the fingerprint.

## 1. The guard cannot see real task failures

`guard_real_logs.py`, `guard_real_diagnose.py`

Replayed against recorded episodes from physical robots, using each dataset's own failure
labels:

| dataset | robot | detection on real failures | false alarms on held-out successes |
|---|---|---|---|
| BotFails (145 failures) | SO-100 | 5/145 = 0.034 [0.015, 0.078] | 0/54 = 0.000 [0.000, 0.066] |
| DROID (58 failures) | Franka | 0/58 = 0.000 [0.000, 0.062] | 4/103 = 0.039 [0.015, 0.096] |

Whether that is a threshold problem or a signal problem decides whether it is fixable, so
each head was scored by AUROC, which is threshold-free. 0.5 is a coin flip:

| dataset | chunk consistency | action stats | stagnation |
|---|---|---|---|
| BotFails | 0.500 | 0.512 | 0.512 |
| DROID | 0.500 | 0.330 | 0.443 |

There is no signal to recover. On DROID the action-stats head scores failures slightly lower
than successes. A robot that fails a task keeps moving normally, and the action stream does
not reveal it.

One flaw in this method, stated because it affects one row: replaying a recorded log forces
action chunks to be faked as sliding windows, which overlap by construction, so the
chunk-consistency head is structurally meaningless here and reads exactly 0.500 with a median
score of 0.000 on both classes. The other two heads are unaffected and still found nothing.

## 2. What the guard does detect transfers to real robots

`guard_real_faults.py`

The same three execution faults injected into the recorded action stream of real successful
episodes, at magnitudes measured in each action dimension's own standard deviation so the
severity means the same thing on a Franka and an SO-100:

| fault | HIL-SERL SO-100 | BotFails SO-100 | DROID Franka |
|---|---|---|---|
| stall (freeze) | 153/153 | 72/72 | 136/137 |
| erratic (noise, 0.25 sd) | 153/153 | 72/72 | 136/137 |
| offset, 1 sd | 0/153 | 2/72 | 62/137 |
| offset, 2 sd | 0/153 | 50/72 | 124/137 |
| offset, 4 sd | 2/153 | 72/72 | 136/137 |
| false alarms (clean copies) | 0/153 | 0/72 | 2/137 |

Stalls and erratic control are caught within 0.1 to 0.2 seconds on real data, with false
alarms inside the 0.05 bound. Constant offsets need roughly two standard deviations, and on
the HIL-SERL corpus are missed at every magnitude tried. This is replay, not closed-loop
control, and injected faults are not naturally occurring ones.

## 3. The fingerprint catches the offsets the guard misses

`fingerprint_offset_test.py`

The same constant offsets put through the unit fingerprint, which regresses commanded against
measured joint positions across a whole episode:

| dataset | 0.25 sd | 0.5 sd | 1 sd | 2 sd | false alarms |
|---|---|---|---|---|---|
| aloha_static_screw_driver | 20/20 | 20/20 | 20/20 | 20/20 | 4/20 |
| DROID | 48/48 | 48/48 | 48/48 | 48/48 | 3/48 |
| svla_so100_pickplace | 20/20 | 20/20 | 20/20 | 20/20 | 6/20 |
| svla_so101_pickplace | 20/20 | 20/20 | 20/20 | 20/20 | 1/20 |
| so100_pick_cube_in_box | 40/40 | 40/40 | 40/40 | 40/40 | 4/40 |

Perfect separation at a quarter of a standard deviation, eight times more sensitive than the
runtime guard. The caveat was the threshold, not the signal: the ad-hoc three-sigma rule used
here false-alarmed on 5 to 30 percent of clean episodes. That is fixed in section 7.

This gives a division of labour that is measured rather than asserted. The fingerprint is for
drift and miscalibration between sessions. The guard is for stalls and erratic control at
runtime. The vision judge is for task failure. Nothing detects a robot that moves normally and
still fails the task from the action stream alone.

## 4. Gradual faults are detected, and latency tracks the ramp

`guard_experiments.py ramp`

Every previously published detection number used a step change, which is the easiest possible
case. Here each fault ramps to full strength over the time shown:

| fault | 0.0 s | 0.5 s | 1.0 s | 2.0 s | 4.0 s |
|---|---|---|---|---|---|
| freeze | 10/10, 0.68 s | 10/10, 1.20 s | 10/10, 1.70 s | 10/10, 2.68 s | 10/10, 4.68 s |
| offset | 10/10, 0.04 s | 10/10, 0.41 s | 10/10, 1.04 s | 10/10, 1.99 s | 10/10, 3.04 s |
| noise | 10/10, 0.04 s | 9/9, 0.16 s | 9/9, 0.28 s | 10/10, 0.57 s | 10/10, 1.02 s |

Detection holds at 100 percent at every ramp rate, and the delay is not a fixed lag but the
time for the fault to grow large enough to leave the nominal envelope. Noise is caught well
before its ramp completes; a freeze takes about the ramp duration plus 0.7 seconds. Cells hold
10 episodes each, so each interval is [0.722, 1.000] and the claim is "no misses observed",
not "the rate is one".

## 5. A second policy and task: still open

Three candidates, none usable, for three different reasons:

- `lerobot/act_aloha_sim_insertion_human` loads correctly with finite statistics and reaches
  partial reward, but completed the task 0 of 10 times. Real policy, hard task, no nominal
  pool to calibrate on.
- `lerobot/diffusion_pusht` raises an uncorrectable ECC error on an A10 while copying
  `diffusion.unet.final_conv` and poisons the GPU for every other process on the box. Two
  separate cards went from 0 to exactly 64 uncorrectable DRAM errors within a minute of that
  load; a GPU reset cleared both. Identical counts on different cards mean the load triggers
  it, not defective memory.
- `aadarshram/act_pusht` reached 1 of 8 successes, too weak to calibrate on. It also needs
  `pymunk<7`, since pymunk 7 removed the collision-handler API gym-pusht calls.

So the guard's live numbers still come from one policy on one task. Working public simulation
checkpoints are scarcer than expected, which is itself worth knowing.

Chasing this did improve the harness: the loader now reads min/max scaling and the newer
lerobot layout that keeps statistics in separate preprocessor files, and it takes the scheme
from the policy's own config instead of guessing. Applying mean/std where a policy expects
min/max would run it on differently scaled inputs than it was trained on, which is the exact
failure class this project exists to catch.

## 6. False alarms on a large held-out pool

`guard_experiments.py fa`

Pending; see `guard_fa.md` in this directory.

## 7. The fingerprint threshold is now calibrated, not guessed

`robotruth.fingerprint.DriftDetector`, `fingerprint_calibrate.py`

The three-sigma rule is replaced by a split-conformal quantile of the worst standardised
per-joint deviation, so the false-alarm rate is bounded by alpha under exchangeability alone,
with no assumption that per-joint estimates are Gaussian or independent. They are neither,
which is why the old rule leaked. Both rules fitted and measured on the same episodes, eight
real datasets, 270 held-out clean episodes:

| rule | false alarms on clean | detection at 0.25 sd offset |
|---|---|---|
| three-sigma | 23/270 = 0.085 [0.057, 0.125] | 268/270 |
| conformal, alpha 0.05 | 3/270 = 0.011 [0.004, 0.032] | 268/270 |

Sensitivity is unchanged and false alarms drop by a factor of eight. Restricting to the four
datasets with enough episodes to certify alpha = 0.05 at all, the conformal rule produced
**0 false alarms in 190 clean episodes** against 12 for the three-sigma rule.

The three datasets that did leak are exactly the ones where the detector reported itself
saturated: with 15 calibration fingerprints the smallest certifiable rate is 0.062, and it
says so rather than claiming 0.05. That is the intended behaviour, and it is also the
practical guidance, which is to collect at least 20 fingerprints per cell before trusting the
bound.

## Method notes

- A rented A10 was physically faulty and burned an hour of rollouts before its ECC counter
  surfaced in the job logs. The runner now checks the card and runs an arithmetic self-check
  before spending anything.
- Every rollout from the 0.1.4 experiment is saved as npz, so the calibration pool here was
  reused instead of regenerated, and threshold work replays offline for free.
