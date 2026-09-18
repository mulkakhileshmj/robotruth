# robotruth: is your robot policy actually better?

Draft launch note. Numbers marked TBD are filled from examples/validation once the real-data run completes.

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

TBD from examples/validation/VALIDATION.md:

- Claims audit on published numbers: N comparisons, K resolved, the rest inside the noise.
- RoboArena pairwise sessions: Bradley-Terry ranking with bootstrap errors over N policies.
- Real DAgger and HIL-SERL datasets: interventions per hour, autonomous fraction, success with intervals.
- Unit fingerprints across N different SO-100 and Franka units performing the same task family: median lag, backlash and tracking error per unit.
- Judge from action-stream features alone on labeled real episodes: balanced accuracy with interval, abstain rate, false alarms per hour.
- Guard on real episodes: detection rate and false alarms per hour at alpha 0.05.

## What it is not

Not a benchmark, not a leaderboard, not a simulator, not a model. It runs next to your own stack (LeRobot, openpi, GR00T or anything else) and reads files.

## Install

```bash
pip install robotruth
```

Apache 2.0. Methods and their sources are listed in the README.
