# robotruth: is your robot policy actually better?

All numbers below were measured by robotruth itself on public data on 18 September 2026 (examples/validation/2026-09-18). Nothing is simulated unless marked.

## The problem

Robot learning has a measurement problem, and almost nobody is working on it.

- Zero of 13 recent real-robot VLA papers report a confidence interval (PhAIL audit, May 2026).
- A 50-trial success rate carries a 20 to 30 point interval (Toyota Research Institute, LBM study).
- Moving a camera or a tote shifts task completion by 22 points, more than the gap between competing models.
- The same weights with different action-normalization metadata go from 28/28 to 2/28 successes ("Same Weights, Different Robot", June 2026).
- Two identical Franka arms replaying the same commands differ 5x in tracking error, and a policy drops from 98 to 18 percent between them (SPACE, June 2026).
- Only 19.8 percent of LIBERO state-of-the-art claims are provably significant.

The people shipping robots say the same thing. Physical Intelligence's CEO: the data bottleneck is closing and evaluation does not scale. Genesis AI: evaluation is the bottleneck for iteration speed. Dyna Robotics hires for "the monitoring that should have existed already."

Capital in the first half of 2026: $18.8 billion into robotics, well under $150 million into evaluation and observability.

## What robotruth does

robotruth is a Python library and command-line tool. It makes four kinds of robot numbers trustworthy.

1. **Configuration truth.** `robotruth contract` extracts a manifest of everything that makes a checkpoint an executable policy (weights hash, normalizer statistics, action mode, chunking, control rate, camera set, embodiment) and fails closed if the deployed tuple does not match the evaluated one. Validated on five public checkpoints: base pi0 and pi05 ship no normalizer statistics at all; SmolVLA ships statistics for three different robots in one file; nothing on disk says which way the gripper closes.
2. **Statistical truth.** `robotruth stats` puts an interval on every rate, pairs interleaved trials, runs an anytime-valid sequential test so you can stop the moment the answer is clear, compares censored time to success, ranks policies with Bradley-Terry, and tells you how many trials you needed if the answer is inconclusive. `robotruth stats audit` takes reported success counts and says which comparisons survive.
3. **Outcome truth.** `robotruth episodes` records every rollout with interventions, failure class and provenance hashes, writes MCAP for Foxglove, and computes the fleet metrics nobody publishes: interventions per hour, mean time between interventions, autonomous fraction. `robotruth judge` fuses action-stream features with a vision-language model and abstains when uncertain, reporting false alarms per hour.
4. **Drift truth.** `robotruth fingerprint` fingerprints the cell (marker positions, exposure, sharpness) and the robot unit (lag, backlash, offset, gain per joint) so "the model got worse" can be separated from "the camera moved 4 mm". `robotruth guard` monitors a policy at runtime with conformal thresholds calibrated on successful rollouts.

## What we measured on public data

33 public datasets, 1,611 real robot episodes, zero ingest errors, all processed end to end on one $1.29/hour GPU instance.

- **The fleet metrics nobody publishes, computed from a real DAgger deployment log** (dual PiperX folding demo, 130 episodes, 6.4 robot-hours): 129.6 interventions per hour [120.9, 138.7], autonomous fraction 0.000 [0.000, 0.029], mean time between interventions 0.5 minutes, 16.7 percent of time under human control. This is what "assisted autonomy" actually looks like in numbers.
- **Unit-to-unit variance measured across five different physical SO-100 and SO-101 arms** owned by different people doing similar tasks: base-joint backlash spans 0.107 to 0.478 (a 4.5x spread) and command-to-response lag spans 101 to 135 ms. The SPACE paper showed a policy dropping from 98 to 18 percent between two identical arms; these fingerprints are how you catch that before the eval.
- **RoboArena, re-analysed**: 3,284 real pairwise evaluation sessions over 15 policies on DROID Frankas, parsed into a Bradley-Terry ranking with bootstrap errors and per-policy binary and partial success. pi0.5 ranks at the top among large-sample policies; the binning-decoder baseline finishes last at 2 successes in 629 trials.
- **A real eval log, audited** (150 SO-101 trials over three camera configurations): two of three pairwise comparisons are resolved at 95 percent confidence; the third, a 10-point gap over 50 trials per arm, is inside the noise. This is exactly the call labs currently make by eyeballing two percentages.
- **An honest negative result**: on 400 DROID episodes (85 percent success), a judge built from action-stream features alone abstains on 97 percent of episodes and adds nothing. DROID failures are semantic, not kinematic. The calibrated abstain is doing precisely its job: refusing to guess, and telling you the vision-language channel is required there.
- 72 tests pass, including simulation checks that the intervals actually cover and that the sequential test's false-alarm rate stays below alpha. The wheel installs clean in a fresh environment.

## What it is not

Not a benchmark, not a leaderboard, not a simulator, not a model. It runs next to your own stack (LeRobot, openpi, GR00T or anything else) and reads files.

## Install

```bash
pip install robotruth
```

Apache 2.0. Methods and their sources are listed in the README.
