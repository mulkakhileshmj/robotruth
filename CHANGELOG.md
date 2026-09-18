# Changelog

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
