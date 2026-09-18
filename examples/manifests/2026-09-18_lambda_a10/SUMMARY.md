# Extractor validation on real public checkpoints

| repo | family | weights GB | dim | chunk | steps | Hz | mode | norm | stats present | cameras | robot | unverified | self-check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lerobot/act_aloha_sim_transfer_cube_human | lerobot | 0.21 | 14 | 100 | 100 | 50.0 | absolute | mean_std | yes | 1 | aloha | 1 | FAIL: 1 fatal, 2 warn, 28 checks |
| lerobot/pi0 | lerobot | 14.01 | 6 | 50 | 50 | None | absolute | mean_std | NO | 3 | unknown | 4 | FAIL: 5 fatal, 6 warn, 32 checks |
| lerobot/pi05_base | lerobot | 14.47 | 32 | 50 | 50 | None | absolute | quantile | NO | 3 | unknown | 4 | FAIL: 5 fatal, 6 warn, 32 checks |
| lerobot/smolvla_base | lerobot | 0.91 | 6 | 50 | 50 | None | absolute | mean_std | yes | 3 | unknown | 3 | FAIL: 3 fatal, 6 warn, 32 checks |
| nvidia/GR00T-N1.5-3B | gr00t | 5.45 | 44 | 16 | None | 20.0 | absolute | quantile | yes | 1 | gr1 | 2 | FAIL: 2 fatal, 2 warn, 28 checks |

## Unverified fields per checkpoint (what a lab must record at deployment)

- **lerobot/act_aloha_sim_transfer_cube_human**: action.gripper_convention
  - note: built from a probe directory (hashes and configs); weights were hashed on the machine that holds them.
- **lerobot/pi0**: action.control_hz, action.gripper_convention, action.normalization.stats_sha256, embodiment.robot
  - note: built from a probe directory (hashes and configs); weights were hashed on the machine that holds them.
  - note: no normalization statistics shipped with this checkpoint (base model); stats are set at fine-tune time and must be recorded then.
  - note: denoising steps: 10 (drives latency; VLA-Perf)
- **lerobot/pi05_base**: action.control_hz, action.gripper_convention, action.normalization.stats_sha256, embodiment.robot
  - note: built from a probe directory (hashes and configs); weights were hashed on the machine that holds them.
  - note: no normalization statistics shipped with this checkpoint (base model); stats are set at fine-tune time and must be recorded then.
  - note: denoising steps: 10 (drives latency; VLA-Perf)
- **lerobot/smolvla_base**: action.control_hz, action.gripper_convention, embodiment.robot
  - note: built from a probe directory (hashes and configs); weights were hashed on the machine that holds them.
  - note: normalizer file holds statistics for 3 datasets (so100, so100-blue, so100-red); the one selected at runtime is part of the executable policy. Record it in the overlay as action.normalization.stats_source.
  - note: denoising steps: 10 (drives latency; VLA-Perf)
- **nvidia/GR00T-N1.5-3B**: action.gripper_convention, action.n_action_steps
  - note: built from a probe directory (hashes and configs); weights were hashed on the machine that holds them.
  - note: denoising steps: 4