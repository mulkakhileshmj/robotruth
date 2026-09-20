# Where the baseline itself is weak

89 of 256 scenarios failed under act_v18 (no candidate involved: this is the policy's own envelope).

| region | inside | elsewhere | lift | p |
|---|---|---|---|---|
| `cube_y <= 0.42 and abs_cube_yaw_deg > 12.1` | 14/16 (87.5% [64, 97]) | 75/240 (31.2%) | 2.8x | 1.1e-05 |
| `cube_yaw_deg > 11.9` | 46/77 (59.7% [49, 70]) | 43/179 (24.0%) | 2.5x | 5.9e-08 |
| `cube_y <= 0.42` | 17/26 (65.4% [46, 81]) | 72/230 (31.3%) | 2.1x | 7.9e-04 |
| `abs_cube_yaw_deg > 12.1` | 66/153 (43.1% [36, 51]) | 23/103 (22.3%) | 1.9x | 4.2e-04 |
