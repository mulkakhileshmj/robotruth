# Changelog

## 0.1.4 (2026-09-19)

- **The stall blind spot is closed.** A `StagnationScorer` watches windowed motion of the action stream, and `calibrate_guard` now gives it its own conformal threshold in a `MultiHeadScorer` instead of averaging it into the composite, alarming on the union of the two heads with alpha split between them. Measured on the same live policy and fault protocol as 0.1.3, on a 150-episode calibration pool: stall, offset and noise faults all detected 10/10, pooled 30/30 = 1.000 [0.886, 1.000], median latency 0.04 s, 0/18 false alarms on held-out successes.
- The dedicated head is the reason it works. With the stagnation score averaged into the composite, freeze detection was only 1/10: a frozen action stream is self-consistent and in-distribution, so three of the four component scores report nominal and dilute the one that sees the stall. Both runs are published side by side in `examples/validation/2026-09-19/guard_detection_v2/`.
- The stall head reports itself as saturated below roughly 40 calibration episodes rather than claiming an uncertifiable guarantee.
- **Cross-dataset benchmark**: `ops/benchmark_fetch.py`, `ops/benchmark_report.py` and `ops/botfails_bench.py` fetch public LeRobot datasets (metadata and parquet only), run every module over them and render one HTML report. First run covers 54 datasets from RoboMIND and AgiBotWorld task ports, OpenX conversions, DROID, BotFails and DAgger logs, with zero ingest failures.
- **HTML reports redesigned**: markdown in report sections now renders as HTML instead of being dumped as raw text, with a document header, table of contents, numbered sections, sentence-cased headings and column headers, and print styles.
- `ops/guard_detection.py` saves every rollout as npz, so threshold changes can be replayed offline instead of costing GPU hours.

## 0.1.3 (2026-09-19)

- Guard detection measured with policy-breaking faults injected into executed actions on a live policy (all 30 fault episodes failed the task). On a 164-episode calibration pool: offset (miscalibration) faults 10/10 detected, noise (erratic policy) faults 10/10 with max thresholds, median detection latency 0.04 s, 0/17 false alarms on held-out successes. Stall (freeze) faults 0/10: a frozen action stream is self-consistent and in-distribution, so the current scorers cannot see it; a stagnation scorer is the planned fix and the blind spot is documented in Known limits.
- Calibration-size guidance is now measured: 45-episode pools leaked up to 3/17 false alarms in the same experiment while the 164-episode pool leaked none. Calibrate on 150 or more nominal episodes.

## 0.1.2 (2026-09-19)

- Fused judge validated on real labeled data: on 323 BotFails episodes the fused and calibrated judge reaches 0.921 [0.617, 0.972] balanced accuracy at 25.8 percent coverage with 2.73 false alarms per hour, against 0.614 and 11.13 for the uncalibrated vision channel alone. The action channel alone abstains on everything, which is why fusion is the shipped default.
- Guard conformal methods measured across three independent calibration draws on a live policy. The bonferroni method alarmed on about 20 percent of held-out successes at 44 to 49 calibration episodes because every time bin was saturated, so its alpha did not hold; `max` remains the default and the caveat is documented in `guard/conformal.py`. Detection remains unmeasured because the perturbation broke the policy in only 1 of 12 episodes.
- Memory-safe frame sampling in the evaluation scripts: sampling frames from a decoded video kept the whole video alive through numpy views, which grew to 140 GB across a dataset. Frames are now streamed and copied.
- Em dashes removed from all documentation.

## 0.1.1 (2026-09-18)

- Open-weight VLM backend for the judge (`robotruth.judge.open_vlm.OpenVLMBackend`, Qwen-VL class via transformers): the vision channel runs on your own GPU with no API key. Benchmarked zero-shot on 140 labeled real UR5 episodes: balanced accuracy 0.555 [0.442, 0.663] at 3.5 s per episode on one A10 (frontier ceiling on FailBench: 0.77). The Claude backend remains available behind the `vlm` extra.
- Live policy-loop validation: a real ACT policy in gym-aloha with the guard attached to every step and the judge scored against simulator truth. Judge balanced accuracy 0.929 [0.651, 0.987] over 24 live episodes with zero false alarms; guard false-alarm bound held in both runs; run-to-run threshold variance at small calibration sizes documented.
- openpi extractor validated against the real pi0_base orbax checkpoint (12 GB hashed; norm_stats for nine robot assets detected and flagged as part of the executable policy).
- The live test reproduced the contract failure class in the wild: lerobot 0.6.1 silently drops this checkpoint's normalization buffers, taking an 83 percent policy to 0 percent with only a log warning.

## 0.1.0 (2026-09-18)

First public release.

- Contract checker: ExecSpec manifests for LeRobot (old and new preprocessor formats), GR00T, openpi and custom checkpoints; probe-directory source so weights never move; fail-closed comparison. Validated against five public checkpoints (ACT, SmolVLA, pi0, pi05, GR00T N1.5).
- Statistics: Wilson and Clopper-Pearson intervals, paired and bootstrap differences, anytime-valid empirical Bernstein confidence sequences, sequential paired test, sample-size planner, Kaplan-Meier and log-rank on censored time to success, Bradley-Terry with bootstrap errors, blinded interleaved schedules, claims audit (Newcombe intervals and Fisher tests over reported counts).
- Episode records: schema with 12 failure classes and 60 subclasses, interventions with source, provenance hashes; JSONL log; MCAP bridge; fleet metrics with intervals (success, autonomous fraction, interventions per hour, MTBI, failure Pareto); LeRobot dataset ingest (v2 and v3, success and intervention columns, DROID column aliases).
- Fingerprints: robot unit fingerprint from an excitation log (joint lag and backlash by regression, RMSE, offset, gain) and workspace cell fingerprint from a camera frame (ArUco markers, exposure, sharpness, colour balance) with drift tolerances.
- Judge: action-stream features, VLM backend protocol with a Claude backend, logistic fusion, split-conformal calibration with abstain, evaluation with balanced accuracy and false alarms per hour, FailBench-style loader.
- Guard: Mahalanobis (Ledoit-Wolf, optional CUDA), chunk-consistency and action-stat scorers, time-uniform conformal thresholds (max and Bonferroni), contrast-set calibration, hysteresis and hard limits, offline replay, metrics.
- Ops: GPU-box scripts for checkpoint probing, dataset fetching, sync-and-test, and end-to-end validation on public LeRobot datasets.
