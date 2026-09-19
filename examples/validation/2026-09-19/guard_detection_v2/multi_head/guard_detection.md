# Guard detection with policy-breaking faults

Policy lerobot/act_aloha_sim_transfer_cube_human live in gym-aloha. Calibration pool 150 nominal episodes (from 200 runs). Evaluation: 20 held-out nominal plus 10 episodes per fault, faults injected into the executed action at t = 3.0 s. Alpha 0.05. An episode counts for detection only if the task actually failed.

| pool | n | method | false alarms on successes | freeze det | offset det | noise det | median latency s |
|---|---|---|---|---|---|---|---|
| full | 150 | max | 0/18 | 10/10 | 10/10 | 10/10 | 0.04 |
| full | 150 | bonferroni | 0/18 | 10/10 | 10/10 | 10/10 | 0.04 |
| sub45-0 | 45 | max | 0/18 | 10/10 | 10/10 | 10/10 | 0.04 |
| sub45-0 | 45 | bonferroni | 1/18 | 10/10 | 10/10 | 10/10 | 0.04 |
| sub45-1 | 45 | max | 0/18 | 10/10 | 10/10 | 10/10 | 0.04 |
| sub45-1 | 45 | bonferroni | 2/18 | 10/10 | 10/10 | 10/10 | 0.04 |
| sub45-2 | 45 | max | 0/18 | 10/10 | 10/10 | 10/10 | 0.04 |
| sub45-2 | 45 | bonferroni | 4/18 | 10/10 | 10/10 | 10/10 | 0.04 |

Full-pool max method, all faults pooled: detection 30/30 = 1.000 [0.886, 1.000] (95% Wilson, n=30).
Full-pool max method false-alarm rate on held-out successes: 0.000 [0.000, 0.176] (95% Wilson, n=18) (bound: 0.05).

Total wall time 6827 s.

GUARD_DETECTION_COMPLETE