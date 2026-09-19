# Guard stability across calibration draws

Policy lerobot/act_aloha_sim_transfer_cube_human live in gym-aloha. Shared evaluation pool: 20 nominal + 12 perturbed episodes (observation shift at 3.0 s). 3 independent calibration pools of 60 live episodes; both conformal methods at alpha 0.05.

| seed | method | nominal cal eps | perturbed alerted after onset | early alerts | alarms on successes | false alarms/h |
|---|---|---|---|---|---|---|
| 0 | max | 44 | 2/12 | 0 | 3/17 | 106.6 |
| 0 | bonferroni | 44 | 2/12 | 0 | 4/17 | 142.2 |
| 1 | max | 46 | 1/12 | 0 | 0/17 | 0.0 |
| 1 | bonferroni | 46 | 1/12 | 0 | 4/17 | 142.2 |
| 2 | max | 49 | 1/12 | 0 | 0/17 | 0.0 |
| 2 | bonferroni | 49 | 1/12 | 0 | 3/17 | 106.6 |

Spread of perturbed-episode detection across seeds:
- max: min 0.08, max 0.17 over 3 draws
- bonferroni: min 0.08, max 0.17 over 3 draws

Total wall time 5885 s.

GUARD_STABILITY_COMPLETE