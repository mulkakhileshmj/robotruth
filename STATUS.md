# robotruth status

Updated 2026-09-18.

## Done

- Project scaffold: pyproject (hatchling), Apache 2.0, uv venv at .venv, 23 tests green.
- Module 1, contract checker: ExecSpec manifest (pydantic, validate on assignment, stable fingerprint), extractors for lerobot, openpi, gr00t and custom directories (reads config and safetensors headers only, never imports the framework), fail-closed checker with fatal, warn and info fields, overlay mechanism for fields the extractor cannot read. CLI: `robotruth contract extract`, `robotruth contract check` (exit 1 on fail).
- Module 2, statistics: Wilson and Clopper-Pearson intervals, paired Wald and bootstrap differences, empirical Bernstein confidence sequence (anytime-valid, coverage verified by simulation in tests), sequential paired test with margin, sample-size planner (independent and paired), Kaplan-Meier and log-rank for censored time to success, KS variant, Bradley-Terry with bootstrap standard errors, blinded interleaved schedule with pair ids. CLI: `robotruth stats plan | schedule | compare | interval`.
- Report: Markdown and HTML, every rate carries an interval, verdict PASS / WARN / FAIL, exit code 1 when the candidate is worse. Demo in examples/.

## Next

1. Validate the three extractors on real public checkpoints (SmolVLA and ACT from the LeRobot hub, an openpi pi0 checkpoint, GR00T N1.5). Needs a Hugging Face token for gated repos. Record what each adapter still cannot read and document the overlay needed.
2. Module 3: episode record schema (policy manifest hash, cell fingerprint, intervention start and stop with operator source, failure class, recovery outcome) as MCAP channels alongside LeRobotDataset; Foxglove panel and Rerun loader.
3. Publish the first evidence artifact: re-analyse the 13 real-robot papers audited by PhAIL and the LIBERO leaderboard through `robotruth stats`, and report how many claims survive an interval.
4. Module 4: cell and unit fingerprint (fiducial camera pose, exposure, excitation trajectory, actuator parameter fit). Needs two cheap arms or partner logs.
5. Module 5: hybrid outcome judge with abstain, validated on FailBench. Needs an Anthropic API key.

## Rules

- This folder is the only place edits happen for this work. No existing codebase is touched.
- Public tools only. Methods, not code, are borrowed from prior work.
- A number without an interval is not a result.
