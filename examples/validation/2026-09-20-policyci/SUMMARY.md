# Policy CI first light

2026-09-20. Lambda A10, 30 vCPUs, us-east-1, terminated after the pull. 1,000 episodes,
5 policies, one shared 200-scenario battery `515e96ff75282363`, 31 workers, about 90 minutes
wall clock. Replay videos (515 of them, 142 MB) stayed local; everything else is here.

The cell: `lerobot/act_aloha_sim_transfer_cube_human` in gym-aloha, weights unmodified. The
candidates are the same checkpoint with a constant bias added to proprioception before
normalization, which degrades the policy without touching a single weight.

## Results

| run | success | paired diff vs baseline | verdict | newly broken | fixed | deploy record |
|---|---|---|---|---|---|---|
| act_v18 (baseline) | 85.0% [79.4, 89.3] | — | — | — | — | — |
| act_v18_s1 (same policy, seed 1) | 85.0% [79.4, 89.3] | 0.0% [0.0, 0.0] | inconclusive | 0 | 0 | insufficient_evidence |
| bias 0.02 | 56.0% [49.1, 62.7] | −29.0% [−37.4, −20.6] | **worse** | 74 | 16 | **block** |
| bias 0.05 | 3.5% | −81.5% [−87.1, −75.9] | **worse** | 164 | 1 | **block** |
| bias 0.10 | 0.5% | −84.5% [−89.5, −79.5] | **worse** | 169 | 0 | **block** |

The baseline landed at 85.0%, inside the 83 to 87 percent this checkpoint is documented to
achieve. That is the end-to-end check on policy loading, the reattached normalization
statistics, the simulator and the evaluator: if any of them were wrong, this number would
not be here.

All three genuine regressions were caught and blocked. The identical-policy pair was not
called a regression. That is the behaviour the design required.

## The finding that matters more than the table

**The cell is bit-deterministic, so the noise floor is genuinely zero and the negative
control passed trivially.**

Comparing the two baseline runs scenario by scenario:

| | |
|---|---|
| outcome disagreements | 0 / 200 |
| identical `max_reward` | 200 / 200 |
| identical step count | 200 / 200 |

Not merely the same success rate: the same reward and the same number of steps in every
scenario. ACT is deterministic (it emits an action chunk, it does not sample), and MuJoCo
from a fixed reset seed is deterministic, so `--policy-seed` changes nothing here.

Two consequences, and the second is the uncomfortable one:

1. **Scenario identity works exactly as intended.** Two independent processes, six shards
   each, reconstructed identical scenes and identical trajectories from content hashes alone.
   That is a strong result for the reproducibility machinery.
2. **The noise-floor machinery is untested by this run.** It reported zero because there was
   no noise, not because it separated noise from signal. Every "beyond noise" figure in this
   bundle equals the raw count. A stochastic policy (diffusion, any sampling head) or domain
   randomisation in the scene is needed before that feature can be said to work.

## Failure mode shifts with dose, not just failure rate

| run | timeout | grasp |
|---|---|---|
| act_v18 | 23 | 7 |
| bias 0.02 | 46 | 42 |
| bias 0.05 | 46 | 147 |
| bias 0.10 | 19 | 180 |

The baseline mostly runs out of time. Under a small bias both modes rise together. Under
larger bias the policy stops being able to grasp at all, and timeouts fall only because
failure arrives sooner. A rate alone would not show this; the taxonomy does.

## Brittleness cuts both ways

At bias 0.02 the candidate **fixed 16 scenarios** while breaking 74. Sixteen scenes the
baseline could not complete were completed by a deliberately degraded policy. The baseline's
failures are therefore not simply "hard scenarios", and a headline success rate hides that
entirely. This is the argument for scenario-level diffing in one number.

## Known weakness in the reporting, visible here

The identical-policy comparison returns **inconclusive**, not "no difference". With a margin
of zero, an equivalence claim is unreachable by construction, so two provably identical runs
are reported as undecided. The verdict is not wrong, but it is unhelpful, and a default
equivalence margin should be set.

## Files

- `battery_b1.jsonl` — the battery, content hash `515e96ff75282363`
- `merged/<run>/run_manifest.json` — per-run manifest: simulator pins, policy contract, per-scenario results
- `merged/<run>/episodes.jsonl` — 200 robotruth episode records per run
- `diff_*.md` — the four comparisons
- `passport_*.json` — the deploy records, each carrying its own digest
- `box.txt`, `gpu.txt` — the machine

Reproduce with `policyci/ops/run_policyci_box.sh <dir> 200 6 "0.02 0.05 0.10"`.
