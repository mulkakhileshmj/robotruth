# Policy CI: act_v18_s1 vs act_v18

Battery `5996cc2a571ad4dc` (256 scenarios), task `aloha_transfer_cube`, backend `gym_aloha.transfer_cube.v0`, evaluator `0.1.0`.

| | act_v18 | act_v18_s1 |
|---|---|---|
| Success | 65.2% [59.2, 70.8] | 65.2% [59.2, 70.8] |

Paired difference (B - A): 0.0% [0.0, 0.0]

**Verdict: INCONCLUSIVE (more trials needed)** (anytime-valid, n=256, diff +0.000 [-0.058, +0.058])

## Run-to-run variation

**Not measured.** Run the same policy twice over this battery before reading anything into the scenario counts below. Without it, a broken scenario and a coin flip look identical.

## Scenario diff

- Newly broken: 0 observed | noise floor NOT MEASURED: run the same policy twice first
- Fixed: 0

## Where the failures concentrate

_No region of the scene space concentrates these failures beyond chance, or the battery carries no named factors. A seed-only battery cannot describe a region; sample one with `--factors`._

---
Pins hash A `59cdd82cfd86dc3a` B `59cdd82cfd86dc3a`. Run manifests `7353b63d580f4a32` / `a309ebbad78c42c0`. A number without an interval is not a result.
