# Guard detection with policy-breaking faults

Policy lerobot/act_aloha_sim_transfer_cube_human live in gym-aloha. Calibration pool 163 nominal episodes (from 200 runs). Evaluation: 20 held-out nominal plus 10 episodes per fault, faults injected into the executed action at t = 3.0 s. Alpha 0.05. An episode counts for detection only if the task actually failed.

| pool | n | method | false alarms on successes | freeze det | offset det | noise det | median latency s |
|---|---|---|---|---|---|---|---|
| full | 163 | max | 0/16 | 1/10 | 10/10 | 10/10 | 0.04 |
| full | 163 | bonferroni | 0/16 | 1/10 | 10/10 | 10/10 | 0.04 |
| sub45-0 | 45 | max | 0/16 | 1/10 | 10/10 | 10/10 | 0.04 |
| sub45-0 | 45 | bonferroni | 1/16 | 6/10 | 10/10 | 10/10 | 0.04 |
| sub45-1 | 45 | max | 0/16 | 1/10 | 10/10 | 10/10 | 0.04 |
| sub45-1 | 45 | bonferroni | 0/16 | 2/10 | 9/10 | 10/10 | 0.04 |
| sub45-2 | 45 | max | 0/16 | 0/10 | 1/10 | 10/10 | 0.04 |
| sub45-2 | 45 | bonferroni | 0/16 | 9/10 | 9/10 | 9/10 | 0.04 |

Full-pool max method, all faults pooled: detection 21/30 = 0.700 [0.521, 0.833] (95% Wilson, n=30).
Full-pool max method false-alarm rate on held-out successes: 0.000 [0.000, 0.194] (95% Wilson, n=16) (bound: 0.05).

Total wall time 7176 s.

GUARD_DETECTION_COMPLETE