# Guard on real robot logs

The guard replayed against recorded episodes from physical robots, where the dataset itself says which episodes failed. Action chunks are sliding windows over the recorded action stream, standing in for a policy's emitted chunks.

Two honest caveats. This is replay, not closed-loop control: the guard watches a recording and never intervenes, so nothing here says what would have happened if it had. And in the BotFails corpus the actor is a human teleoperator, so what the scorers separate is good from bad robot motion, not good from bad policy behaviour.

## Elvinky__pi05-piperx-7h-demo-dagger-full-episodes-20260914

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

## Elvinky__so101-fold-clothes-dagger-20260909

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

## TheMuz__rollout_kirby_dagger_v2_iter1_20260507_143451

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

## jccj__so100_block_in_cup

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

## jpizarrom__hilserl_so100_grocery_so100_2025112223_20

Skipped: needs at least 30 successful and 10 failed episodes, has 381 and 0.

## k-chan-l__rollout_dice_color_sort_dagger

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

## kantine__BotFails

Robot `so100`, 30 Hz. 178 successful and 145 failed real episodes. Calibrated on 124 successes, replayed 54 held-out successes and 145 failures.

- Detection on real failures: 5/145 = 0.034 [0.015, 0.078] (95% Wilson, n=145)
- False alarms on held-out successes: 0/54 = 0.000 [0.000, 0.066] (95% Wilson, n=54) (bound 0.05)
- Median warning before the episode ended: 16.3 s
- False alarms per hour of robot time: 0.0
- Stall head: alpha 0.025, saturated False

## lerobot__aloha_static_screw_driver

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

## lerobot__droid_1.0.1

Robot `Franka`, 15 Hz. 341 successful and 58 failed real episodes. Calibrated on 238 successes, replayed 103 held-out successes and 58 failures.

- Detection on real failures: 0/58 = 0.000 [0.000, 0.062] (95% Wilson, n=58)
- False alarms on held-out successes: 4/103 = 0.039 [0.015, 0.096] (95% Wilson, n=103) (bound 0.05)
- Median warning before the episode ended: n/a
- False alarms per hour of robot time: 6.438152033352334
- Stall head: alpha 0.025, saturated False

## lerobot__example_hil_serl_dataset

Skipped: needs at least 30 successful and 10 failed episodes, has 1 and 14.

## lerobot__svla_so100_pickplace

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

## lerobot__svla_so101_pickplace

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

## recast-robotics__rollout_dagger_r1_s2_b0_20260915_121327

Skipped: needs at least 30 successful and 10 failed episodes, has 10 and 0.

## sixpigs1__so100_pick_cube_in_box

Skipped: needs at least 30 successful and 10 failed episodes, has 0 and 0.

Total wall time 98 s.

REAL_LOGS_COMPLETE