# Why the guard misses real failures

Per-head AUROC of the per-episode maximum score, real successes against real failures. AUROC is threshold-free: 0.5 means the head cannot see these failures at all and no threshold would help, while a high value means the signal is there and the threshold is what is wrong.

## kantine__BotFails

178 successes, 145 failures, robot `so100`.

| head | component | AUROC | median score on failures | median score on held-out successes |
|---|---|---|---|---|
| main | chunk_consistency | 0.500 | 0.000 | 0.000 |
| main | action_stats | 0.512 | 2.778 | 2.666 |
| main | stagnation | 0.512 | 2.755 | 2.755 |
| stall | stall | 0.512 | 2.755 | 2.755 |

## lerobot__droid_1.0.1

341 successes, 58 failures, robot `Franka`.

| head | component | AUROC | median score on failures | median score on held-out successes |
|---|---|---|---|---|
| main | chunk_consistency | 0.500 | 0.000 | 0.000 |
| main | action_stats | 0.330 | 1.648 | 2.054 |
| main | stagnation | 0.443 | 4.219 | 4.624 |
| stall | stall | 0.443 | 4.219 | 4.624 |


DIAGNOSE_COMPLETE