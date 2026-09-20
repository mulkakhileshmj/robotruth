# Why several detectors land below chance on real failures

An AUROC below 0.5 means the score runs the wrong way: real failures score lower, so a monitor tuned on this data would alarm on successes first. The table gives the median of each raw quantity for successful and failed episodes, and the AUROC of that quantity on its own, so the direction of the effect is visible without any detector in the way.

## kantine__BotFails

Robot `so100`, 178 successes, 145 failures.

| quantity | median, successes | median, failures | AUROC of this quantity alone |
|---|---|---|---|
| steps | 1194 | 1194 | 0.460 |
| motion per step | 1.566 | 1.316 | 0.457 |
| total motion | 1968 | 1600 | 0.452 |
| action spread | 7.168 | 5.35 | 0.461 |

## lerobot__droid_1.0.1

Robot `Franka`, 341 successes, 58 failures.

| quantity | median, successes | median, failures | AUROC of this quantity alone |
|---|---|---|---|
| steps | 239 | 188.5 | 0.363 |
| motion per step | 0.03309 | 0.02851 | 0.380 |
| total motion | 7.642 | 5.217 | 0.302 |
| action spread | 0.00193 | 0.001595 | 0.391 |


WHY_BELOW_CHANCE_COMPLETE