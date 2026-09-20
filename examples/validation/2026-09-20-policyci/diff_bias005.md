# Policy CI: act_v19_bias005 vs act_v18

Battery `515e96ff75282363` (200 scenarios), task `aloha_transfer_cube`, backend `gym_aloha.transfer_cube.v0`, evaluator `0.1.0`.

| | act_v18 | act_v19_bias005 |
|---|---|---|
| Success | 85.0% [79.4, 89.3] | 3.5% [1.7, 7.0] |

Paired difference (B - A): -81.5% [-87.1, -75.9]

**Verdict: WORSE** (anytime-valid, n=200, diff -0.812 [-0.921, -0.703])

## Scenario diff

- Newly broken: 164 observed | noise floor 0 (from A-vs-A on the same battery) | beyond noise: 164
- Fixed: 1

### Newly broken scenarios

| scenario | reset seed | A | B failure | replay |
|---|---|---|---|---|
| `218b360f3b8d` (#0) | see battery | pass | timeout | videos/218b360f3b8d_act_v19_bias005.mp4 |
| `886f35a0fc1b` (#1) | see battery | pass | grasp | videos/886f35a0fc1b_act_v19_bias005.mp4 |
| `5e416104fde1` (#2) | see battery | pass | timeout | videos/5e416104fde1_act_v19_bias005.mp4 |
| `9f64b0d1a58e` (#3) | see battery | pass | grasp | videos/9f64b0d1a58e_act_v19_bias005.mp4 |
| `49a7b9fe8a04` (#4) | see battery | pass | grasp | videos/49a7b9fe8a04_act_v19_bias005.mp4 |
| `25c68bd7144d` (#5) | see battery | pass | grasp | videos/25c68bd7144d_act_v19_bias005.mp4 |
| `470c97aa9df7` (#6) | see battery | pass | grasp | videos/470c97aa9df7_act_v19_bias005.mp4 |
| `f060b34bb765` (#7) | see battery | pass | timeout | videos/f060b34bb765_act_v19_bias005.mp4 |
| `55d69d57855f` (#8) | see battery | pass | timeout | videos/55d69d57855f_act_v19_bias005.mp4 |
| `443acdb24cfb` (#9) | see battery | pass | timeout | videos/443acdb24cfb_act_v19_bias005.mp4 |
| `bffef68ead34` (#13) | see battery | pass | grasp | videos/bffef68ead34_act_v19_bias005.mp4 |
| `abc58f7f862b` (#14) | see battery | pass | grasp | videos/abc58f7f862b_act_v19_bias005.mp4 |
| `6b7e11f16f4c` (#16) | see battery | pass | grasp | videos/6b7e11f16f4c_act_v19_bias005.mp4 |
| `937ac657aae9` (#17) | see battery | pass | grasp | videos/937ac657aae9_act_v19_bias005.mp4 |
| `4feb2585b50f` (#19) | see battery | pass | timeout | videos/4feb2585b50f_act_v19_bias005.mp4 |
| `6739b7a3f392` (#22) | see battery | pass | timeout | videos/6739b7a3f392_act_v19_bias005.mp4 |
| `90ec5f0e2531` (#23) | see battery | pass | grasp | videos/90ec5f0e2531_act_v19_bias005.mp4 |
| `f3624f37b7f6` (#24) | see battery | pass | grasp | videos/f3624f37b7f6_act_v19_bias005.mp4 |
| `c75a4a402b9f` (#25) | see battery | pass | timeout | videos/c75a4a402b9f_act_v19_bias005.mp4 |
| `a7ae20fd0e5f` (#26) | see battery | pass | grasp | videos/a7ae20fd0e5f_act_v19_bias005.mp4 |
| `86044038eda6` (#28) | see battery | pass | grasp | videos/86044038eda6_act_v19_bias005.mp4 |
| `3a8030a47343` (#29) | see battery | pass | grasp | videos/3a8030a47343_act_v19_bias005.mp4 |
| `543a3a0869ce` (#30) | see battery | pass | grasp | videos/543a3a0869ce_act_v19_bias005.mp4 |
| `6b9104248e8f` (#31) | see battery | pass | grasp | videos/6b9104248e8f_act_v19_bias005.mp4 |
| `f88b2a62c6de` (#33) | see battery | pass | grasp | videos/f88b2a62c6de_act_v19_bias005.mp4 |
| `d042c0d25d6a` (#34) | see battery | pass | timeout | videos/d042c0d25d6a_act_v19_bias005.mp4 |
| `3bc78b834a35` (#37) | see battery | pass | grasp | videos/3bc78b834a35_act_v19_bias005.mp4 |
| `18fd153309e2` (#38) | see battery | pass | grasp | videos/18fd153309e2_act_v19_bias005.mp4 |
| `e6fe22ba4972` (#39) | see battery | pass | grasp | videos/e6fe22ba4972_act_v19_bias005.mp4 |
| `c00dcf987a9f` (#40) | see battery | pass | timeout | videos/c00dcf987a9f_act_v19_bias005.mp4 |
| `926370404c2d` (#44) | see battery | pass | timeout | videos/926370404c2d_act_v19_bias005.mp4 |
| `2a5e12332134` (#45) | see battery | pass | timeout | videos/2a5e12332134_act_v19_bias005.mp4 |
| `86910c665591` (#46) | see battery | pass | grasp | videos/86910c665591_act_v19_bias005.mp4 |
| `67645f845b4a` (#47) | see battery | pass | grasp | videos/67645f845b4a_act_v19_bias005.mp4 |
| `5c838d1857ad` (#48) | see battery | pass | timeout | videos/5c838d1857ad_act_v19_bias005.mp4 |
| `03c4a950a856` (#49) | see battery | pass | grasp | videos/03c4a950a856_act_v19_bias005.mp4 |
| `071961b48b7e` (#50) | see battery | pass | grasp | videos/071961b48b7e_act_v19_bias005.mp4 |
| `c77e609ae4ad` (#51) | see battery | pass | timeout | videos/c77e609ae4ad_act_v19_bias005.mp4 |
| `7dd427e49726` (#52) | see battery | pass | grasp | videos/7dd427e49726_act_v19_bias005.mp4 |
| `e99c76becb48` (#53) | see battery | pass | grasp | videos/e99c76becb48_act_v19_bias005.mp4 |
| `49f6084251fc` (#54) | see battery | pass | grasp | videos/49f6084251fc_act_v19_bias005.mp4 |
| `29f9b71c1f9a` (#55) | see battery | pass | grasp | videos/29f9b71c1f9a_act_v19_bias005.mp4 |
| `5a88538e4642` (#57) | see battery | pass | grasp | videos/5a88538e4642_act_v19_bias005.mp4 |
| `517fc501a9f9` (#58) | see battery | pass | grasp | videos/517fc501a9f9_act_v19_bias005.mp4 |
| `12cfbc2aa00a` (#59) | see battery | pass | grasp | videos/12cfbc2aa00a_act_v19_bias005.mp4 |
| `e0a095230764` (#60) | see battery | pass | grasp | videos/e0a095230764_act_v19_bias005.mp4 |
| `c081f96f5011` (#61) | see battery | pass | grasp | videos/c081f96f5011_act_v19_bias005.mp4 |
| `1acafd98a753` (#62) | see battery | pass | timeout | videos/1acafd98a753_act_v19_bias005.mp4 |
| `7844958f5ded` (#64) | see battery | pass | grasp | videos/7844958f5ded_act_v19_bias005.mp4 |
| `d2c522a8d9b4` (#65) | see battery | pass | grasp | videos/d2c522a8d9b4_act_v19_bias005.mp4 |
| ... and 114 more | | | | |

### Fixed scenarios (1)

`e08059ead281`

---
Pins hash A `59cdd82cfd86dc3a` B `59cdd82cfd86dc3a`. Run manifests `9a3294cbb211339e` / `4429801ba7a97309`. A number without an interval is not a result.
