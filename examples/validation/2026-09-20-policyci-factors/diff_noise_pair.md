# Policy CI: act_noise_b vs act_noise_a

Battery `5996cc2a571ad4dc` (256 scenarios), task `aloha_transfer_cube`, backend `gym_aloha.transfer_cube.v0`, evaluator `0.1.0`.

| | act_noise_a | act_noise_b |
|---|---|---|
| Success | 61.3% [55.2, 67.1] | 62.9% [56.8, 68.6] |

Paired difference (B - A): 1.6% [-3.4, 6.5]

**Verdict: INCONCLUSIVE (more trials needed)** (anytime-valid, n=256, diff +0.015 [-0.076, +0.106])

## Run-to-run variation

**Not measured.** Run the same policy twice over this battery before reading anything into the scenario counts below. Without it, a broken scenario and a coin flip look identical.

## Scenario diff

- Newly broken: 19 observed | noise floor NOT MEASURED: run the same policy twice first
- Fixed: 23

## Where the failures concentrate

| region | inside | elsewhere | lift | p |
|---|---|---|---|---|
| `cube_yaw_deg > -5.87` | 17/153 (11.1% [7, 17]) | 2/103 (1.9%) | 5.7x | 4.0e-03 |
| `cube_x > 0.12 and cube_y > 0.56` | 5/21 (23.8% [11, 45]) | 14/235 (6.0%) | 4.0x | 1.2e-02 |
| `cube_x > 0.12 and abs_cube_yaw_deg <= 5.87` | 4/19 (21.1% [9, 43]) | 15/237 (6.3%) | 3.3x | 4.1e-02 |
| `cube_x > 0.12` | 12/102 (11.8% [7, 19]) | 7/154 (4.5%) | 2.6x | 2.9e-02 |

### Newly broken scenarios

| scenario | reset seed | A | B failure | replay |
|---|---|---|---|---|
| `5324adf7593e` (#5) | see battery | pass | timeout | videos/5324adf7593e_act_noise_b.mp4 |
| `1960af17880d` (#7) | see battery | pass | timeout | videos/1960af17880d_act_noise_b.mp4 |
| `090228219f51` (#12) | see battery | pass | timeout | videos/090228219f51_act_noise_b.mp4 |
| `97734931bf56` (#28) | see battery | pass | timeout | videos/97734931bf56_act_noise_b.mp4 |
| `2c08160452e7` (#49) | see battery | pass | timeout | videos/2c08160452e7_act_noise_b.mp4 |
| `7b29a390c558` (#71) | see battery | pass | timeout | videos/7b29a390c558_act_noise_b.mp4 |
| `de6981824941` (#86) | see battery | pass | timeout | videos/de6981824941_act_noise_b.mp4 |
| `97acf034b438` (#96) | see battery | pass | timeout | videos/97acf034b438_act_noise_b.mp4 |
| `75928ac709f2` (#132) | see battery | pass | timeout | videos/75928ac709f2_act_noise_b.mp4 |
| `ad87a260171e` (#146) | see battery | pass | timeout | videos/ad87a260171e_act_noise_b.mp4 |
| `5fd20167775b` (#150) | see battery | pass | timeout | videos/5fd20167775b_act_noise_b.mp4 |
| `334fd216fedf` (#161) | see battery | pass | timeout | videos/334fd216fedf_act_noise_b.mp4 |
| `9472cc99bf73` (#181) | see battery | pass | grasp | videos/9472cc99bf73_act_noise_b.mp4 |
| `43e6144790c3` (#193) | see battery | pass | timeout | videos/43e6144790c3_act_noise_b.mp4 |
| `0ef5b2ec076a` (#195) | see battery | pass | timeout | videos/0ef5b2ec076a_act_noise_b.mp4 |
| `2b7f167fd674` (#199) | see battery | pass | timeout | videos/2b7f167fd674_act_noise_b.mp4 |
| `70902cd598b9` (#238) | see battery | pass | timeout | videos/70902cd598b9_act_noise_b.mp4 |
| `3e274f8937d2` (#240) | see battery | pass | grasp | videos/3e274f8937d2_act_noise_b.mp4 |
| `78235b4616af` (#254) | see battery | pass | grasp | videos/78235b4616af_act_noise_b.mp4 |

### Fixed scenarios (23)

`599f04bf1d0a`, `eb9a7c752e2d`, `73ab471f7407`, `d25c69b77e2b`, `9ae5cbf06e87`, `9486e78a201c`, `e4b367d403ed`, `68a2441e5a52`, `7f2d3c9ec304`, `319606e4d0ae`, `364c2cfd7170`, `1bb9f80dc8e4`, `e076ce8e5ee0`, `b68539f2b702`, `d2b637519e07`, `4a54bd33284d`, `0347be293e78`, `839a9849bb95`, `44c71b2389f8`, `595bf15f7eb5`, `18481eab8627`, `7c207f907498`, `58d0537eada7`

---
Pins hash A `59cdd82cfd86dc3a` B `59cdd82cfd86dc3a`. Run manifests `cc2fb182ed25eb95` / `a27968961eb2b1c3`. A number without an interval is not a result.
