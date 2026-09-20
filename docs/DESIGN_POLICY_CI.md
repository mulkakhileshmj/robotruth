# Design note: Policy CI

Status: draft for build. Written 2026-09-20. This note freezes the brainstorm into a buildable v1 scope.

## The one question v1 answers

Can Policy CI reliably tell a robotics team whether a new policy is better, worse, or unsafe compared with the previous version?

Not prompt-to-policy. Not policy generation. Not compliance. Those come later and sit on top of this.

## Positioning

Continuous testing, regression detection and evidence for robot policies. The customer brings their own policy. We tell them whether it actually works, where it breaks, and whether v19 is really better than v18. Every number carries evidence: an interval, a noise floor, a replayable scenario.

The generator (Policy Factory) is a later expansion. The gate is the business; the generator is the demo.

## Relationship to robotruth today

robotruth already provides, in shipped and validated form:

- Contract: hashed manifests of the executable policy tuple, fail-closed comparison. This is the skeleton of the Policy Passport.
- Stats: Wilson intervals, paired sequential tests, planners, claim audits. This is the regression engine's math.
- Episodes: one record per rollout with a 12-class failure taxonomy, provenance hashes, fleet metrics. This is the evaluator's output format.
- Judge and Guard: outcome judgment and runtime monitoring, calibrated conformally.

Policy CI adds three new pieces and one product surface:

1. Scenario system (deterministic scene generation)
2. Policy runner (policy against simulator, at scale)
3. Simulator backend interface (MuJoCo/Genesis first, Isaac Sim second)
4. Scenario browser with replay (the developer experience)

Nothing existing is rewritten. Policy CI composes the existing modules.

## The frozen first cell

Amendment 2026-09-20: the build starts on ALOHA transfer-cube in gym-aloha (MuJoCo) with the
public `lerobot/act_aloha_sim_transfer_cube_human` ACT checkpoint, because robotruth already
validated that exact policy live on the box (~83-87% success, known normalization failure
mode). Franka/UR5 bin picking becomes cell two behind the same backend interface. The rest
of this section describes the target cell, unchanged.

- Robot: Franka or UR5 class arm (simulated), parallel gripper
- Task class: bin picking / pick-and-place, one task class only for at least a year
- Simulator backend 1: MuJoCo (or Genesis), because it is free, fast, headless-friendly and pip-installable on a rented GPU box
- Simulator backend 2 (later): Isaac Sim, behind the same interface
- Policy: customer-supplied or a public baseline (e.g. an ACT/diffusion policy from LeRobot), never our own generator

Compute rule: all simulation and policy inference runs on rented GPU boxes. The local machine does orchestration, code, and result analysis only. Every run's outputs are pulled back before the box is terminated, versioned under examples/validation as today.

## Foundation layer: four definitions that come before any feature

These decide whether the killer screen is true or noise.

### 1. Deterministic scenario identity

A scenario is a value, not a run. It is fully defined by:

- a parameter vector (object pose, object set, clutter layout, lighting, camera pose, physics randomization seed)
- a scenario schema version
- a content hash of the above

"37 new failures" only means something if scenario 4812 is bit-identical scene setup for v18 and v19. Physics engines are not bit-reproducible across GPU drivers or engine versions, so the Passport also pins: simulator name and version, physics backend, GPU driver, container image digest. Two runs are comparable only when those pins match; the regression engine refuses to diff across mismatched pins (fail closed, same philosophy as contract check).

### 2. Noise floor before regression

Stochastic policies and stochastic physics mean the same policy run twice will not agree on every scenario. Before any A/B claim:

- run v18 against v18 (independent seeds where the policy is stochastic) and measure the per-scenario flip rate
- report every regression count next to the measured noise floor
- flag only hotspots that survive a paired test (robotruth stats already has the machinery: paired designs, anytime-valid sequential tests)

Product rule: the UI never shows "37 regressions" bare. It shows observed / baseline-noise / statistically significant / inconclusive. Every number needs evidence behind it. This is the existing robotruth rule ("a number without an interval is not a result") applied to diffs.

### 3. Policy interface contract

One small, boring, stable contract. The platform must not care whether the policy is a VLA, RL, imitation, classical control, or generated code.

- Transport: a local gRPC service (or in-process Python adapter for v1) the runner calls
- Declared observation schema: named camera streams with resolution and encoding, proprioception vector layout, optional language instruction
- Declared action schema: dimension, semantics (joint / EE delta / EE absolute), gripper convention, chunk size
- Declared control frequency
- Episode lifecycle: reset(scenario) -> step(obs) -> action, termination, timeout
- The whole declaration is hashed into the Passport via the existing contract module. A policy whose declared schema does not match what the runner feeds it fails closed before a single episode runs. This is the "Same Weights, Different Robot" failure class, blocked at the door.

### 4. Versioned success definition

Success for pick-and-place v1, evaluated from simulator ground truth:

- target object inside the target volume
- at rest (velocity below threshold for N steps)
- gripper open, not in contact with the object
- no contact event above a force threshold during the episode (collision gate)
- within the episode time budget

This definition is code, versioned, and its version appears in the Passport. Changing the evaluator invalidates comparisons across the change, and the regression engine enforces that. Failure categories map into the existing episode taxonomy (grasp failure, perception/localization failure, collision, placement error, timeout, trajectory fault).

## The five components

### A. Scenario generator

Samples parameter vectors from a declared distribution: object position and orientation ranges, clutter density, occlusion, lighting, camera jitter. Each sample gets a seed, hash, and schema version. Supports two modes:

- battery mode: N scenarios from the base distribution (the standing test suite for the task class)
- targeted mode: N scenarios sampled densely around a named hotspot region (thin falsification loop, in scope for v1 as a follow-on once clustering works)

Scenario batteries are immutable once published: a battery is itself content-addressed, so "ran battery B3" is a reproducible claim.

### B. Policy runner

Runs one policy over one battery on the GPU box. Parallel headless environments, per-episode outputs written as robotruth episode records plus a replay bundle (scenario params, observation stream sample, full action trace, contact/termination events). Deterministic mapping scenario -> seed -> initial state.

### C. Evaluator

Applies the versioned success definition to each episode, assigns a failure category, writes the episode record. Sim ground truth makes this cheap; the judge module is not needed in sim v1 (it becomes essential at the physical-validation stage).

### D. Regression engine

Input: two run manifests over the same battery with matching pins. Output:

- overall rates with intervals (existing stats module)
- scenario-by-scenario diff: fixed / newly broken / unchanged
- noise-floor line from the self-comparison run
- significant regression hotspots: clusters of newly broken scenarios described by their shared parameter region ("rotation above 32 degrees, high occlusion")
- deploy signal: better / worse / inconclusive, with the trial count needed if inconclusive (stats plan already computes this)

Clustering v1 is deliberately simple: cluster in scenario-parameter space (the parameters are known and low-dimensional), describe clusters by the tightest parameter bounds. Sensor-signature clustering is a later phase.

### E. Passport

Extends the existing ExecSpec manifest into a run-level record:

- policy contract hash (weights, normalizer stats, action semantics, control rate)
- policy interface declaration hash
- scenario battery hash and schema version
- simulator pins (engine, version, physics backend, driver, image digest)
- evaluator version
- results with intervals, failure Pareto, regression diff against the named baseline
- signature over the whole record

Two Passports diff cleanly. A Passport is reproducible from its stored scenarios. Nothing in it is editable without breaking the hash.

## The product surface: scenario browser

The killer developer feature. For any scenario:

- seed, hash, pins, both policies' results side by side
- per-gate breakdown (grasp / collision / placement / time)
- replay of both runs (v1: rendered video per episode saved by the runner; interactive replay later)
- "find similar failures": nearest scenarios in parameter space with their pass rates, which surfaces the shared condition ("218 similar scenarios, common condition rotation > 32 degrees, v18 91% vs v19 64%")

The top-level screen:

```
POLICY v19 vs v18            battery B3 (10,000 scenarios)

Overall:  94.2% [93.7, 94.6]     was 91.7% [91.1, 92.2]

Newly broken:   37 observed | noise floor 11 | significant 21 | inconclusive 5
Fixed:          112

Critical gates:
  collision      0.1%
  failed grasp   4.2%
  bad placement  1.5%

Significant regression hotspots:
  object rotation > 35 deg
  high occlusion
  dense clutter

Verdict: BETTER overall, REGRESSED in 2 regions -> human review
```

v1 UI can be the existing robotruth HTML report style plus saved videos. A live web UI is not required to prove the loop.

## Physical validation (phase 2, design constraint now)

Sim results and physical validation are separate stages by design. CI works before calibration exists; calibration progressively tells you how much to trust sim conclusions.

First physical experiment, when hardware exists: 20 to 50 real trials, stratified across sim-predicted easy / borderline / hard scenarios (not only hard ones), scored for rank correlation between sim difficulty and real failure. That is the first calibration datum. The judge and guard modules run on the physical side. No hardware is needed to build and prove everything above; the GPU box covers all of v1.

## Measured: what actually sets the wall clock (2026-09-20, Lambda A10)

The plan assumed the GPU would be the constraint. It is not. Numbers from the box:

| what | measured |
|---|---|
| ALOHA sim, one worker, 400-step episode | 24.5 s |
| simulator step rate, single-threaded | 12 steps/s |
| aggregate throughput, 30 workers on 30 vCPUs | 11 episodes/min |
| implied parallel speedup from 30x workers | about 4x |

The reason is rendering. This image cannot create an NVIDIA EGL context, so MuJoCo falls
back to Mesa's software rasteriser: rendering is CPU and memory-bandwidth bound, and 30
workers contend for it. Policy inference does run on the GPU, and it is not the bottleneck
(ACT emits 100-step action chunks, so the model runs about four times per episode).

Consequences for the design, not just for one run:

- The cost model for a battery is CPU-hours, not GPU-hours. A cheap many-core box beats an
  expensive GPU box for this workload, and the simulator backend choice should be judged on
  headless render throughput before anything else.
- Getting hardware offscreen rendering working is worth more than any other optimisation
  available here, plausibly 5x or more. It belongs in the base image, not the run bundle.
- Battery size is a real cost decision. 200 scenarios gives roughly a five point interval on
  a success rate near 90 percent; 100 gives about seven. That trade should be stated
  whenever a battery size is chosen, because the interval is the product.

## Build order

1. Simulator interface + MuJoCo backend + one pick-and-place scene with a public baseline policy (proves the runner end to end)
2. Scenario generator + battery hashing + deterministic replay of a single scenario
3. Evaluator gates + episode records + per-run report with intervals
4. Self-comparison noise floor, then v18 vs v19 regression diff on two checkpoints of the same policy
5. Parameter-space failure clustering + hotspot description
6. Scenario browser v1 (HTML report + per-failure videos + similar-failure lookup)
7. Passport record extension + fail-closed pin matching
8. Thin targeted-scenario loop around a detected hotspot

Each step runs on a GPU box and pulls its evidence back into examples/validation, same discipline as the existing log.

## Explicit non-goals for v1

- No policy generation, planning, or VLM task understanding
- No Isaac Sim (backend 2, behind the interface)
- No multi-robot, no second task class
- No sim-to-real calibration model (only the design constraint that stages stay separate)
- No compliance claims in positioning; auditable evidence stands on its own
- No live web app; reports and saved replays first

## Later phases (recorded, not scoped)

Phase 2 sim-to-real calibration. Phase 3 more robots and task classes. Phase 4 Passport deployment gates. Phase 5 autonomous targeted testing (falsification-style search over scenario space). Phase 6 Policy Factory: generation, skill composition, demonstration ingestion. The pitch progression: "we tell you whether it works" -> "we help you fix it" -> "we generate it".

## Open questions

- First backend: MuJoCo vs Genesis. Decide by a one-day spike on the box: headless throughput, contact stability for grasping, asset pipeline.
- Baseline policy for development: a public LeRobot pick policy, or train a quick ACT baseline on the box.
- Battery size vs box cost: start with 1,000 scenarios per run and measure wall-clock before committing to 10,000.
- Whether the policy interface v1 is in-process Python (fastest to build) or gRPC from day one (cleaner isolation). Leaning in-process with the gRPC schema already written down.
