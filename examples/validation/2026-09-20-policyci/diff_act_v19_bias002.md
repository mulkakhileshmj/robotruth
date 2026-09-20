# Policy CI: act_v19_bias002 vs act_v18

Battery `515e96ff75282363` (200 scenarios), task `aloha_transfer_cube`, backend `gym_aloha.transfer_cube.v0`, evaluator `0.1.0`.

| | act_v18 | act_v19_bias002 |
|---|---|---|
| Success | 85.0% [79.4, 89.3] | 56.0% [49.1, 62.7] |

Paired difference (B - A): -29.0% [-37.4, -20.6]

**Verdict: WORSE** (anytime-valid, n=200, diff -0.287 [-0.437, -0.137])

## Scenario diff

- Newly broken: 74 observed | noise floor 0 (from A-vs-A on the same battery) | beyond noise: 74
- Fixed: 16

### Newly broken scenarios

| scenario | reset seed | A | B failure | replay |
|---|---|---|---|---|
| `886f35a0fc1b` (#1) | see battery | pass | grasp | videos/886f35a0fc1b_act_v19_bias002.mp4 |
| `9f64b0d1a58e` (#3) | see battery | pass | timeout | videos/9f64b0d1a58e_act_v19_bias002.mp4 |
| `49a7b9fe8a04` (#4) | see battery | pass | timeout | videos/49a7b9fe8a04_act_v19_bias002.mp4 |
| `470c97aa9df7` (#6) | see battery | pass | timeout | videos/470c97aa9df7_act_v19_bias002.mp4 |
| `443acdb24cfb` (#9) | see battery | pass | timeout | videos/443acdb24cfb_act_v19_bias002.mp4 |
| `abc58f7f862b` (#14) | see battery | pass | grasp | videos/abc58f7f862b_act_v19_bias002.mp4 |
| `6b7e11f16f4c` (#16) | see battery | pass | grasp | videos/6b7e11f16f4c_act_v19_bias002.mp4 |
| `f3624f37b7f6` (#24) | see battery | pass | grasp | videos/f3624f37b7f6_act_v19_bias002.mp4 |
| `86044038eda6` (#28) | see battery | pass | grasp | videos/86044038eda6_act_v19_bias002.mp4 |
| `3a8030a47343` (#29) | see battery | pass | timeout | videos/3a8030a47343_act_v19_bias002.mp4 |
| `d042c0d25d6a` (#34) | see battery | pass | timeout | videos/d042c0d25d6a_act_v19_bias002.mp4 |
| `3bc78b834a35` (#37) | see battery | pass | timeout | videos/3bc78b834a35_act_v19_bias002.mp4 |
| `18fd153309e2` (#38) | see battery | pass | timeout | videos/18fd153309e2_act_v19_bias002.mp4 |
| `e6fe22ba4972` (#39) | see battery | pass | grasp | videos/e6fe22ba4972_act_v19_bias002.mp4 |
| `86910c665591` (#46) | see battery | pass | timeout | videos/86910c665591_act_v19_bias002.mp4 |
| `67645f845b4a` (#47) | see battery | pass | timeout | videos/67645f845b4a_act_v19_bias002.mp4 |
| `071961b48b7e` (#50) | see battery | pass | timeout | videos/071961b48b7e_act_v19_bias002.mp4 |
| `e99c76becb48` (#53) | see battery | pass | timeout | videos/e99c76becb48_act_v19_bias002.mp4 |
| `29f9b71c1f9a` (#55) | see battery | pass | timeout | videos/29f9b71c1f9a_act_v19_bias002.mp4 |
| `5a88538e4642` (#57) | see battery | pass | grasp | videos/5a88538e4642_act_v19_bias002.mp4 |
| `517fc501a9f9` (#58) | see battery | pass | grasp | videos/517fc501a9f9_act_v19_bias002.mp4 |
| `12cfbc2aa00a` (#59) | see battery | pass | grasp | videos/12cfbc2aa00a_act_v19_bias002.mp4 |
| `7844958f5ded` (#64) | see battery | pass | grasp | videos/7844958f5ded_act_v19_bias002.mp4 |
| `d678ddd6a6b1` (#66) | see battery | pass | grasp | videos/d678ddd6a6b1_act_v19_bias002.mp4 |
| `d9e4be3fc9e8` (#69) | see battery | pass | timeout | videos/d9e4be3fc9e8_act_v19_bias002.mp4 |
| `f09a37c6238a` (#71) | see battery | pass | timeout | videos/f09a37c6238a_act_v19_bias002.mp4 |
| `80fab9d9fa54` (#73) | see battery | pass | timeout | videos/80fab9d9fa54_act_v19_bias002.mp4 |
| `de4af1fb42c0` (#74) | see battery | pass | grasp | videos/de4af1fb42c0_act_v19_bias002.mp4 |
| `8ef4442efbc8` (#76) | see battery | pass | timeout | videos/8ef4442efbc8_act_v19_bias002.mp4 |
| `2a8d7a972753` (#86) | see battery | pass | grasp | videos/2a8d7a972753_act_v19_bias002.mp4 |
| `7fed2cb0c35e` (#87) | see battery | pass | grasp | videos/7fed2cb0c35e_act_v19_bias002.mp4 |
| `c8200f8a4875` (#89) | see battery | pass | timeout | videos/c8200f8a4875_act_v19_bias002.mp4 |
| `adcb3922d6d7` (#95) | see battery | pass | grasp | videos/adcb3922d6d7_act_v19_bias002.mp4 |
| `dcfe2f91903a` (#96) | see battery | pass | timeout | videos/dcfe2f91903a_act_v19_bias002.mp4 |
| `10a0f36590d6` (#100) | see battery | pass | grasp | videos/10a0f36590d6_act_v19_bias002.mp4 |
| `d06f46ec20f8` (#101) | see battery | pass | grasp | videos/d06f46ec20f8_act_v19_bias002.mp4 |
| `1e96d7ca71e4` (#102) | see battery | pass | timeout | videos/1e96d7ca71e4_act_v19_bias002.mp4 |
| `59dadc400ba8` (#104) | see battery | pass | timeout | videos/59dadc400ba8_act_v19_bias002.mp4 |
| `88f3fec966fd` (#105) | see battery | pass | grasp | videos/88f3fec966fd_act_v19_bias002.mp4 |
| `d41a7256d379` (#107) | see battery | pass | timeout | videos/d41a7256d379_act_v19_bias002.mp4 |
| `817d1c4dfe51` (#110) | see battery | pass | timeout | videos/817d1c4dfe51_act_v19_bias002.mp4 |
| `b1f4a5b4ea48` (#111) | see battery | pass | grasp | videos/b1f4a5b4ea48_act_v19_bias002.mp4 |
| `1ad8dc4f1c98` (#117) | see battery | pass | timeout | videos/1ad8dc4f1c98_act_v19_bias002.mp4 |
| `220371257759` (#118) | see battery | pass | grasp | videos/220371257759_act_v19_bias002.mp4 |
| `348a2a35db98` (#119) | see battery | pass | timeout | videos/348a2a35db98_act_v19_bias002.mp4 |
| `6444d19089ec` (#124) | see battery | pass | timeout | videos/6444d19089ec_act_v19_bias002.mp4 |
| `02b0a9b33edf` (#127) | see battery | pass | grasp | videos/02b0a9b33edf_act_v19_bias002.mp4 |
| `5939e7d929d1` (#131) | see battery | pass | timeout | videos/5939e7d929d1_act_v19_bias002.mp4 |
| `960c17a3e84d` (#134) | see battery | pass | timeout | videos/960c17a3e84d_act_v19_bias002.mp4 |
| `ec61f90a06ba` (#137) | see battery | pass | grasp | videos/ec61f90a06ba_act_v19_bias002.mp4 |
| ... and 24 more | | | | |

### Fixed scenarios (16)

`932bcd9c6a00`, `f46d4c63f80e`, `9e78d41a524e`, `a3d09d1409b2`, `1047205a5c8e`, `bceb9c4f1ee4`, `e08059ead281`, `955d77ea2329`, `b0039eee77db`, `016915012e98`, `bfc8b9caf773`, `d106c682d2f3`, `94575c4c8499`, `3d9575d3f796`, `b85811a2c872`, `cdba11f3df8a`

---
Pins hash A `59cdd82cfd86dc3a` B `59cdd82cfd86dc3a`. Run manifests `9a3294cbb211339e` / `1099bdc4245cea27`. A number without an interval is not a result.
