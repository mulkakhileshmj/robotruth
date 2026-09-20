# Guard detection when the fault arrives gradually

Policy lerobot/act_aloha_sim_transfer_cube_human live in gym_aloha/AlohaTransferCube-v0, calibration pool 150 saved nominal episodes, alpha 0.05, max method. Each fault begins at t = 3.0 s and reaches full strength over the ramp time shown. A ramp of 0.0 s is the step change every previously published number used. Detection counts only episodes the fault actually broke.

| fault | ramp | broke the task | detected | detection rate | median latency after onset |
|---|---|---|---|---|---|
| freeze | 0.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 0.68 s |
| freeze | 0.5 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 1.20 s |
| freeze | 1.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 1.70 s |
| freeze | 2.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 2.68 s |
| freeze | 4.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 4.68 s |
| offset | 0.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 0.04 s |
| offset | 0.5 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 0.41 s |
| offset | 1.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 1.04 s |
| offset | 2.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 1.99 s |
| offset | 4.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 3.04 s |
| noise | 0.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 0.04 s |
| noise | 0.5 s | 9/10 | 9/9 | 1.000 [0.701, 1.000] (95% Wilson, n=9) | 0.16 s |
| noise | 1.0 s | 9/10 | 9/9 | 1.000 [0.701, 1.000] (95% Wilson, n=9) | 0.28 s |
| noise | 2.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 0.57 s |
| noise | 4.0 s | 10/10 | 10/10 | 1.000 [0.722, 1.000] (95% Wilson, n=10) | 1.02 s |

Total wall time 5470 s.

GUARD_RAMP_COMPLETE