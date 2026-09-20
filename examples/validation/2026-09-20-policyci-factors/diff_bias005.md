# Policy CI: act_v19_bias005 vs act_v18

Battery `5996cc2a571ad4dc` (256 scenarios), task `aloha_transfer_cube`, backend `gym_aloha.transfer_cube.v0`, evaluator `0.1.0`.

| | act_v18 | act_v19_bias005 |
|---|---|---|
| Success | 65.2% [59.2, 70.8] | 4.3% [2.4, 7.5] |

Paired difference (B - A): -60.9% [-67.3, -54.6]

**Verdict: WORSE** (anytime-valid, n=256, diff -0.611 [-0.724, -0.497])

## Run-to-run variation

**This cell is stochastic.** Measured floor: 19 of 157 passing scenarios flipped to failure when the same policy was run again (12.1% [7.9, 18.1]). 42 scenarios flipped in either direction.

## Scenario diff

- Newly broken: 160 observed | expected from run-to-run variation alone: 19 | beyond that: 141
- Fixed: 4

## Where the failures concentrate

| region | inside | elsewhere | lift | p |
|---|---|---|---|---|
| `cube_y > 0.42` | 151/230 (65.7% [59, 71]) | 9/26 (34.6%) | 1.9x | 2.3e-03 |
| `cube_yaw_deg <= 11.9` | 129/179 (72.1% [65, 78]) | 31/77 (40.3%) | 1.8x | 1.7e-06 |
| `abs_cube_yaw_deg <= 12.1` | 76/103 (73.8% [65, 81]) | 84/153 (54.9%) | 1.3x | 1.6e-03 |

### Newly broken scenarios

| scenario | reset seed | A | B failure | replay |
|---|---|---|---|---|
| `bb5cd1d76e42` (#1) | see battery | pass | timeout | videos/bb5cd1d76e42_act_v19_bias005.mp4 |
| `050e4251e9c7` (#3) | see battery | pass | grasp | videos/050e4251e9c7_act_v19_bias005.mp4 |
| `637dfa65d5aa` (#4) | see battery | pass | grasp | videos/637dfa65d5aa_act_v19_bias005.mp4 |
| `5324adf7593e` (#5) | see battery | pass | grasp | videos/5324adf7593e_act_v19_bias005.mp4 |
| `a6937a4fce38` (#6) | see battery | pass | timeout | videos/a6937a4fce38_act_v19_bias005.mp4 |
| `1960af17880d` (#7) | see battery | pass | grasp | videos/1960af17880d_act_v19_bias005.mp4 |
| `05c4fce2fe2e` (#8) | see battery | pass | grasp | videos/05c4fce2fe2e_act_v19_bias005.mp4 |
| `c82176a2e4eb` (#9) | see battery | pass | timeout | videos/c82176a2e4eb_act_v19_bias005.mp4 |
| `2a0dc40dc998` (#10) | see battery | pass | grasp | videos/2a0dc40dc998_act_v19_bias005.mp4 |
| `2883c1826316` (#11) | see battery | pass | grasp | videos/2883c1826316_act_v19_bias005.mp4 |
| `599f04bf1d0a` (#13) | see battery | pass | grasp | videos/599f04bf1d0a_act_v19_bias005.mp4 |
| `969194617883` (#14) | see battery | pass | grasp | videos/969194617883_act_v19_bias005.mp4 |
| `e56330daa8bd` (#16) | see battery | pass | grasp | videos/e56330daa8bd_act_v19_bias005.mp4 |
| `53544b05e1a8` (#17) | see battery | pass | grasp | videos/53544b05e1a8_act_v19_bias005.mp4 |
| `d1703201f0eb` (#19) | see battery | pass | grasp | videos/d1703201f0eb_act_v19_bias005.mp4 |
| `405fd66c6b45` (#22) | see battery | pass | grasp | videos/405fd66c6b45_act_v19_bias005.mp4 |
| `16863c530e5b` (#23) | see battery | pass | timeout | videos/16863c530e5b_act_v19_bias005.mp4 |
| `eb9a7c752e2d` (#24) | see battery | pass | timeout | videos/eb9a7c752e2d_act_v19_bias005.mp4 |
| `68f2c0448c16` (#25) | see battery | pass | grasp | videos/68f2c0448c16_act_v19_bias005.mp4 |
| `815cc72533db` (#26) | see battery | pass | timeout | videos/815cc72533db_act_v19_bias005.mp4 |
| `776bdaed0627` (#27) | see battery | pass | timeout | videos/776bdaed0627_act_v19_bias005.mp4 |
| `1aeb0d0fda82` (#29) | see battery | pass | grasp | videos/1aeb0d0fda82_act_v19_bias005.mp4 |
| `5ed134ff8b23` (#30) | see battery | pass | timeout | videos/5ed134ff8b23_act_v19_bias005.mp4 |
| `30554aea570a` (#31) | see battery | pass | grasp | videos/30554aea570a_act_v19_bias005.mp4 |
| `0f6ec2a84560` (#32) | see battery | pass | timeout | videos/0f6ec2a84560_act_v19_bias005.mp4 |
| `04ec4fcaedee` (#33) | see battery | pass | grasp | videos/04ec4fcaedee_act_v19_bias005.mp4 |
| `84b3dabe86c7` (#36) | see battery | pass | grasp | videos/84b3dabe86c7_act_v19_bias005.mp4 |
| `598c45d18130` (#37) | see battery | pass | timeout | videos/598c45d18130_act_v19_bias005.mp4 |
| `b2df1cfa675c` (#38) | see battery | pass | grasp | videos/b2df1cfa675c_act_v19_bias005.mp4 |
| `4da09ac021f5` (#40) | see battery | pass | timeout | videos/4da09ac021f5_act_v19_bias005.mp4 |
| `42be238f98b8` (#42) | see battery | pass | grasp | videos/42be238f98b8_act_v19_bias005.mp4 |
| `a176710816e4` (#43) | see battery | pass | grasp | videos/a176710816e4_act_v19_bias005.mp4 |
| `702298d045a1` (#44) | see battery | pass | grasp | videos/702298d045a1_act_v19_bias005.mp4 |
| `2cd8a080fad7` (#45) | see battery | pass | grasp | videos/2cd8a080fad7_act_v19_bias005.mp4 |
| `9048a45ec843` (#47) | see battery | pass | grasp | videos/9048a45ec843_act_v19_bias005.mp4 |
| `2c08160452e7` (#49) | see battery | pass | grasp | videos/2c08160452e7_act_v19_bias005.mp4 |
| `b555d7da62e1` (#50) | see battery | pass | grasp | videos/b555d7da62e1_act_v19_bias005.mp4 |
| `b2e8982be6fe` (#51) | see battery | pass | timeout | videos/b2e8982be6fe_act_v19_bias005.mp4 |
| `89b94cd67a3b` (#53) | see battery | pass | grasp | videos/89b94cd67a3b_act_v19_bias005.mp4 |
| `33b70005fc48` (#54) | see battery | pass | grasp | videos/33b70005fc48_act_v19_bias005.mp4 |
| `33bbf6f89a14` (#55) | see battery | pass | grasp | videos/33bbf6f89a14_act_v19_bias005.mp4 |
| `a9109e0143d1` (#58) | see battery | pass | timeout | videos/a9109e0143d1_act_v19_bias005.mp4 |
| `8c2724112758` (#59) | see battery | pass | grasp | videos/8c2724112758_act_v19_bias005.mp4 |
| `56dbeb51e459` (#60) | see battery | pass | grasp | videos/56dbeb51e459_act_v19_bias005.mp4 |
| `0e8da8b839f8` (#62) | see battery | pass | grasp | videos/0e8da8b839f8_act_v19_bias005.mp4 |
| `8eca978de350` (#65) | see battery | pass | grasp | videos/8eca978de350_act_v19_bias005.mp4 |
| `423e16ef04b4` (#66) | see battery | pass | grasp | videos/423e16ef04b4_act_v19_bias005.mp4 |
| `3340fa7da83c` (#67) | see battery | pass | grasp | videos/3340fa7da83c_act_v19_bias005.mp4 |
| `15d3ce997c3e` (#69) | see battery | pass | grasp | videos/15d3ce997c3e_act_v19_bias005.mp4 |
| `7b29a390c558` (#71) | see battery | pass | grasp | videos/7b29a390c558_act_v19_bias005.mp4 |
| ... and 110 more | | | | |

### Fixed scenarios (4)

`68a2441e5a52`, `7ba82b57c7dd`, `75d1bdbcd9bb`, `ea2e1e9a5d5d`

---
Pins hash A `59cdd82cfd86dc3a` B `59cdd82cfd86dc3a`. Run manifests `7353b63d580f4a32` / `30a7d1180e0519f4`. A number without an interval is not a result.
