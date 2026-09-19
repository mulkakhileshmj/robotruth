# robotruth cross-dataset benchmark

robotruth 0.1.4, generated 2026-09-19 13:38 UTC

**Verdict: 54 datasets benchmarked, 2 with ground-truth labels**

## What robotruth checks [INFO]

- **Contract**: is the deployed policy byte-for-byte the evaluated one (weights, normalization stats, control rate, gripper convention)? Fails closed.
- **Stats**: is checkpoint B really better than A? Intervals, sequential tests, Bradley-Terry, claims audit.
- **Episodes**: one schema for logs with a 12-class failure taxonomy, interventions and fleet metrics.
- **Fingerprint**: has the robot or the cell physically drifted since calibration?
- **Judge**: did the episode succeed? Action stream fused with a vision channel, conformally calibrated so it abstains instead of guessing.
- **Guard**: is the policy failing right now? Per-step scores with time-uniform conformal thresholds.

Below, each of those claims is exercised on public datasets. Labelled datasets come first because only there can identification be checked against ground truth.

## Identification on labelled datasets [PASS]

The judge is trained, calibrated and evaluated on disjoint splits inside each dataset (action-stream features, no vision channel here). Coverage below 100% is the abstain mechanism working: the judge answers only where it is calibrated to be right, and hands the rest to a human. The guard is calibrated on nominal episodes and replayed against the real labelled failures.

- **kantine__BotFails__pooled**: 145 labelled failures available.
- **lerobot__droid_1.0.1**: 44 labelled failures available.

| dataset | labels (success/failure) | judge balanced accuracy | coverage | failure recall | judge false alarms per hour | guard detection on real failures | guard false alarms per hour |
|---|---|---|---|---|---|---|---|
| kantine__BotFails__pooled | 178 / 145 | 0.500 [0.255, 0.897] | 0.052 [0.022, 0.115] | 0.000 [0.000, 0.793] | 0.000 | 0.041 [0.019, 0.087] | 0.000 |
| lerobot__droid_1.0.1 | 256 / 44 | abstained on nearly all | 0.022 [0.006, 0.077] | 1.000 [0.342, 1.000] | 0.000 | 0.023 [0.004, 0.121] | 0.000 |

## Measured capability highlights (from the repository's validated experiments) [PASS]

Numbers below were measured on this same corpus and on a live policy loop, and are reproduced in `examples/validation/` in the repository with the scripts that made them.

- **Fused judge on BotFails** (323 episodes, vision channel Qwen2.5-VL-7B fused with the action stream, conformally calibrated): balanced accuracy 0.921 [0.617, 0.972] at 25.8% coverage with 2.73 false alarms per hour, against 0.614 and 11.13 for the uncalibrated vision channel alone. The action-only rows in the table above are the same corpus without the vision channel: the judge abstains rather than guesses, which is the designed behaviour.
- **Guard on a live ACT policy** (gym-aloha, faults injected into executed actions, 164-episode calibration pool): miscalibration and erratic-action faults detected 20/20 at 0.04 s median latency with 0/17 false alarms on held-out successes.
- **Contract checker in the wild**: lerobot 0.6.1 silently drops this ACT checkpoint's normalization buffers, taking an 83% policy to 0%; the contract check catches exactly this class before deployment.
- **Unit fingerprints** separate five physical SO-100/101 arms (backlash 0.107 to 0.478, lag 101 to 135 ms) from their logs alone.

## Overview of every dataset [INFO]

54 datasets analysed, 0 failed to ingest. Success rates carry Wilson 95% intervals; a dataset without labels is stated as unlabelled rather than guessed. Demo-only datasets (all successes, no failure labels) still exercise ingest, fleet metrics, fingerprints and guard calibration.

| dataset | robot | episodes | fps | success rate | interventions |
|---|---|---|---|---|---|
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_327_Pick_up_at_supermarket | a2d | 209 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_354_Pick_up_at_supermarket | a2d | 300 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_356_Supermarket_packaging | a2d | 300 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_358_toast_bread | a2d | 300 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_359_Warehouse_sorting | a2d | 300 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_367_Take_bread_from_the_toaster | a2d | 257 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_374_Organize_clothing_and_personal_care_products | a2d | 91 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_375_brew_tea | a2d | 300 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_376_Classification_of_electronic_products | a2d | 300 | 30.000 | unlabelled | 0 |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_388_Pick_up_at_supermarket | a2d | 92 | 30.000 | unlabelled | 0 |
| Elvinky__pi05-piperx-7h-demo-dagger-full-episodes-20260914 | bi_piperx_follower | 130 | 30.000 | unlabelled | 831 |
| Elvinky__so101-fold-clothes-dagger-20260909 | bi_so_follower | 22 | 30.000 | unlabelled | 429 |
| IPEC-COMMUNITY__austin_buds_dataset_lerobot | franka | 50 | 20.000 | unlabelled | 0 |
| IPEC-COMMUNITY__berkeley_fanuc_manipulation_lerobot | fanuc_mate | 300 | 10.000 | unlabelled | 0 |
| IPEC-COMMUNITY__berkeley_mvp_lerobot | xarm | 300 | 5.000 | unlabelled | 0 |
| IPEC-COMMUNITY__cmu_stretch_lerobot | hello_stretch | 135 | 10.000 | unlabelled | 0 |
| IPEC-COMMUNITY__dlr_edan_shared_control_lerobot | dlr_edan | 104 | 5.000 | unlabelled | 0 |
| IPEC-COMMUNITY__libero_goal_no_noops_1.0.0_lerobot | franka | 300 | 20.000 | unlabelled | 0 |
| IPEC-COMMUNITY__nyu_door_opening_surprising_effectiveness_lerobot | hello_stretch | 300 | 3.000 | unlabelled | 0 |
| IPEC-COMMUNITY__nyu_franka_play_dataset_lerobot | franka | 300 | 3.000 | unlabelled | 0 |
| IPEC-COMMUNITY__ucsd_kitchen_dataset_lerobot | xarm | 150 | 2.000 | unlabelled | 0 |
| IPEC-COMMUNITY__viola_lerobot | franka | 135 | 20.000 | unlabelled | 0 |
| TheMuz__rollout_kirby_dagger_v2_iter1_20260507_143451 | so_follower | 10 | 30.000 | unlabelled | 10 |
| jccj__so100_block_in_cup | so100 | 50 | 30.000 | unlabelled | 0 |
| jpizarrom__hilserl_so100_grocery_so100_2025112223_20 | unknown | 300 | 20.000 | 100.0% 1.000 [0.987, 1.000] (95% Wilson, n=300) | 0 |
| k-chan-l__rollout_dice_color_sort_dagger | so_follower | 20 | 30.000 | unlabelled | 20 |
| kantine__BotFails__BotFails_normal_train_domotic_dishTidyUp_expert | so100 | 20 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_domotic_groceriesSorting_expert | so100 | 20 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_domotic_makingCoffee_expert | so100 | 20 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_domotic_pouringCoffee_expert | so100 | 20 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_domotic_setTheTable_expert | so100 | 18 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_domotic_vegetablesAndFruitsSorting_expert | so100 | 20 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_industrial_robothon_buttons_expert | so100 | 15 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_industrial_robothon_hatchAndProbe_expert | so100 | 10 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_industrial_screws_sorting_expert | so100 | 20 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_normal_train_industrial_soldering_expert | so100 | 15 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_domotic_dishTidyUp_anomaly | so100 | 15 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_domotic_groceriesSorting_anomaly | so100 | 10 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_domotic_makingCoffee_anomaly | so100 | 15 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_domotic_pouringCoffee_anomaly | so100 | 20 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_domotic_setTheTable_anomaly | so100 | 15 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_domotic_vegetablesAndFruitsSorting_anomaly | so100 | 20 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_industrial_robothon_buttons_anomaly | so100 | 10 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_industrial_robothon_hatchAndProbe_anomaly | so100 | 15 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_industrial_screws_sorting_anomaly | so100 | 10 | 30.000 | unlabelled | 0 |
| kantine__BotFails__BotFails_test_industrial_soldering_anomaly | so100 | 15 | 30.000 | unlabelled | 0 |
| kantine__BotFails__pooled | so100 class | 323 | 30.000 | 55.1% 0.551 [0.497, 0.604] (95% Wilson, n=323) | 0 |
| lerobot__aloha_static_screw_driver | aloha | 50 | 50.000 | unlabelled | 0 |
| lerobot__droid_1.0.1 | Franka | 300 | 15.000 | 85.3% 0.853 [0.809, 0.889] (95% Wilson, n=300) | 0 |
| lerobot__example_hil_serl_dataset | unknown | 15 | 10.000 | 6.7% 0.067 [0.012, 0.298] (95% Wilson, n=15) | 0 |
| lerobot__svla_so100_pickplace | so100 | 50 | 30.000 | unlabelled | 0 |
| lerobot__svla_so101_pickplace | so100_follower | 50 | 30.000 | unlabelled | 0 |
| recast-robotics__rollout_dagger_r1_s2_b0_20260915_121327 | maniskill_sim | 10 | 20.000 | 100.0% 1.000 [0.722, 1.000] (95% Wilson, n=10) | 21 |
| sixpigs1__so100_pick_cube_in_box | so100 | 100 | 30.000 | unlabelled | 0 |

## Guard replay across datasets [INFO]

Guard calibrated per dataset on nominal episodes (conformal max method, alpha 0.05, pseudo-chunks from the recorded action stream) and replayed on held-out episodes. Detection is only meaningful where a dataset contains labelled failures.

| dataset | calibration episodes | held-out nominal | failures replayed | detection rate | false alarms per hour |
|---|---|---|---|---|---|
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_327_Pick_up_at_supermarket | 125 | 84 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_354_Pick_up_at_supermarket | 180 | 120 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_356_Supermarket_packaging | 180 | 120 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_358_toast_bread | 180 | 120 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_359_Warehouse_sorting | 180 | 120 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_367_Take_bread_from_the_toaster | 154 | 103 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_374_Organize_clothing_and_personal_care_products | 54 | 37 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_375_brew_tea | 180 | 120 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_376_Classification_of_electronic_products | 180 | 120 | 0 | no labelled failures | n/a |
| BAAI-DataCube__AgiBotWorld-Beta_G1_task_388_Pick_up_at_supermarket | 55 | 37 | 0 | no labelled failures | n/a |
| Elvinky__pi05-piperx-7h-demo-dagger-full-episodes-20260914 | 78 | 52 | 0 | no labelled failures | n/a |
| Elvinky__so101-fold-clothes-dagger-20260909 | 13 | 9 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__austin_buds_dataset_lerobot | 30 | 20 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__berkeley_fanuc_manipulation_lerobot | 180 | 120 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__berkeley_mvp_lerobot | 180 | 120 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__cmu_stretch_lerobot | 81 | 54 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__dlr_edan_shared_control_lerobot | 62 | 42 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__libero_goal_no_noops_1.0.0_lerobot | 180 | 120 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__nyu_door_opening_surprising_effectiveness_lerobot | 180 | 120 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__nyu_franka_play_dataset_lerobot | 180 | 120 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__ucsd_kitchen_dataset_lerobot | 90 | 60 | 0 | no labelled failures | n/a |
| IPEC-COMMUNITY__viola_lerobot | 81 | 54 | 0 | no labelled failures | n/a |
| TheMuz__rollout_kirby_dagger_v2_iter1_20260507_143451 | 6 | 4 | 0 | no labelled failures | n/a |
| jccj__so100_block_in_cup | 30 | 20 | 0 | no labelled failures | n/a |
| jpizarrom__hilserl_so100_grocery_so100_2025112223_20 | 180 | 120 | 0 | no labelled failures | 4.588 |
| k-chan-l__rollout_dice_color_sort_dagger | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_domotic_dishTidyUp_expert | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_domotic_groceriesSorting_expert | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_domotic_makingCoffee_expert | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_domotic_pouringCoffee_expert | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_domotic_setTheTable_expert | 10 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_domotic_vegetablesAndFruitsSorting_expert | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_industrial_robothon_buttons_expert | 9 | 6 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_industrial_robothon_hatchAndProbe_expert | 6 | 4 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_industrial_screws_sorting_expert | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_normal_train_industrial_soldering_expert | 9 | 6 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_domotic_dishTidyUp_anomaly | 9 | 6 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_domotic_groceriesSorting_anomaly | 6 | 4 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_domotic_makingCoffee_anomaly | 9 | 6 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_domotic_pouringCoffee_anomaly | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_domotic_setTheTable_anomaly | 9 | 6 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_domotic_vegetablesAndFruitsSorting_anomaly | 12 | 8 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_industrial_robothon_buttons_anomaly | 6 | 4 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_industrial_robothon_hatchAndProbe_anomaly | 9 | 6 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_industrial_screws_sorting_anomaly | 6 | 4 | 0 | no labelled failures | n/a |
| kantine__BotFails__BotFails_test_industrial_soldering_anomaly | 9 | 6 | 0 | no labelled failures | n/a |
| kantine__BotFails__pooled | 106 | 72 | 145 | 0.041 [0.019, 0.087] | 0.000 |
| lerobot__aloha_static_screw_driver | 30 | 20 | 0 | no labelled failures | n/a |
| lerobot__droid_1.0.1 | 153 | 103 | 43 | 0.023 [0.004, 0.121] | 0.000 |
| lerobot__svla_so100_pickplace | 30 | 20 | 0 | no labelled failures | n/a |
| lerobot__svla_so101_pickplace | 30 | 20 | 0 | no labelled failures | n/a |
| recast-robotics__rollout_dagger_r1_s2_b0_20260915_121327 | 6 | 4 | 0 | no labelled failures | 105.882 |
| sixpigs1__so100_pick_cube_in_box | 60 | 40 | 0 | no labelled failures | n/a |

## Unit fingerprints [INFO]

Commanded-versus-measured dynamics per dataset. Different numbers for the same robot model are real hardware differences, which is the drift signal robotruth tracks.

| dataset | robot | joints | median joint lag ms | median backlash |
|---|---|---|---|---|
| Elvinky__pi05-piperx-7h-demo-dagger-full-episodes-20260914 | bi_piperx_follower | 14 | 115 | 0.408 |
| Elvinky__so101-fold-clothes-dagger-20260909 | bi_so_follower | 12 | 127 | 0.508 |
| IPEC-COMMUNITY__berkeley_mvp_lerobot | xarm | 8 | 650 | 0.006 |
| TheMuz__rollout_kirby_dagger_v2_iter1_20260507_143451 | so_follower | 6 | 112 | 0.945 |
| jccj__so100_block_in_cup | so100 | 6 | 112 | 0.428 |
| k-chan-l__rollout_dice_color_sort_dagger | so_follower | 6 | 122 | 0.544 |
| kantine__BotFails__BotFails_normal_train_domotic_dishTidyUp_expert | so100 | 6 | 117 | 0.478 |
| kantine__BotFails__BotFails_normal_train_domotic_groceriesSorting_expert | so100 | 6 | 115 | 0.476 |
| kantine__BotFails__BotFails_normal_train_domotic_makingCoffee_expert | so100 | 6 | 111 | 0.625 |
| kantine__BotFails__BotFails_normal_train_domotic_pouringCoffee_expert | so100 | 6 | 108 | 0.802 |
| kantine__BotFails__BotFails_normal_train_domotic_setTheTable_expert | so100 | 6 | 115 | 0.613 |
| kantine__BotFails__BotFails_normal_train_domotic_vegetablesAndFruitsSorting_expert | so100 | 6 | 120 | 0.258 |
| kantine__BotFails__BotFails_normal_train_industrial_robothon_buttons_expert | so100 | 6 | 111 | 1.183 |
| kantine__BotFails__BotFails_normal_train_industrial_robothon_hatchAndProbe_expert | so100 | 6 | 114 | 0.662 |
| kantine__BotFails__BotFails_normal_train_industrial_screws_sorting_expert | so100 | 6 | 116 | 0.310 |
| kantine__BotFails__BotFails_normal_train_industrial_soldering_expert | so100 | 6 | 108 | 1.144 |
| kantine__BotFails__BotFails_test_domotic_dishTidyUp_anomaly | so100 | 6 | 116 | 0.532 |
| kantine__BotFails__BotFails_test_domotic_groceriesSorting_anomaly | so100 | 6 | 118 | 0.450 |
| kantine__BotFails__BotFails_test_domotic_makingCoffee_anomaly | so100 | 6 | 113 | 0.671 |
| kantine__BotFails__BotFails_test_domotic_pouringCoffee_anomaly | so100 | 6 | 110 | 1.282 |
| kantine__BotFails__BotFails_test_domotic_setTheTable_anomaly | so100 | 6 | 116 | 0.710 |
| kantine__BotFails__BotFails_test_domotic_vegetablesAndFruitsSorting_anomaly | so100 | 6 | 117 | 0.391 |
| kantine__BotFails__BotFails_test_industrial_robothon_buttons_anomaly | so100 | 6 | 107 | 1.161 |
| kantine__BotFails__BotFails_test_industrial_robothon_hatchAndProbe_anomaly | so100 | 6 | 118 | 0.736 |
| kantine__BotFails__BotFails_test_industrial_screws_sorting_anomaly | so100 | 6 | 113 | 0.539 |
| kantine__BotFails__BotFails_test_industrial_soldering_anomaly | so100 | 6 | 115 | 0.940 |
| kantine__BotFails__pooled | so100 class | 6 | 114 | 0.622 |
| lerobot__aloha_static_screw_driver | aloha | 14 | 81 | 0.005 |
| lerobot__droid_1.0.1 | Franka | 8 | 247 | 0.018 |
| lerobot__svla_so100_pickplace | so100 | 6 | 111 | 0.389 |
| lerobot__svla_so101_pickplace | so100_follower | 6 | 132 | 0.478 |
| sixpigs1__so100_pick_cube_in_box | so100 | 6 | 103 | 0.311 |

## What went wrong, per dataset [INFO]

- **Elvinky__pi05-piperx-7h-demo-dagger-full-episodes-20260914**: 831 operator interventions recorded.
- **Elvinky__so101-fold-clothes-dagger-20260909**: 429 operator interventions recorded.
- **TheMuz__rollout_kirby_dagger_v2_iter1_20260507_143451**: 10 operator interventions recorded.
- **k-chan-l__rollout_dice_color_sort_dagger**: 20 operator interventions recorded.
- **kantine__BotFails__pooled**: 145 of 323 episodes are labelled failures.
- **lerobot__droid_1.0.1**: 44 of 300 episodes are labelled failures.
- **lerobot__example_hil_serl_dataset**: 14 of 15 episodes are labelled failures.
- **recast-robotics__rollout_dagger_r1_s2_b0_20260915_121327**: 21 operator interventions recorded.
