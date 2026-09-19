# BotFails: action-only vs VLM-only vs fused, all conformal-calibrated with abstain

323 real episodes across 20 tasks (expert = success, anomaly = failure). Splits 129/97/97 stratified. Backend open:Qwen/Qwen2.5-VL-7B-Instruct.

# action-only judge

Episodes: 97 (0 decided, 97 abstained), robot time 1.10 h. Intervals are 95% Wilson.

| metric | value |
|---|---|
| abstain rate | 1.000 [0.962, 1.000] (n=97) |
| coverage | 0.000 [0.000, 0.038] (n=97) |
| accuracy on decided | n/a |
| balanced accuracy on decided | n/a |
| success recall (TPR) | n/a |
| failure recall (TNR) | n/a |
| failure precision | n/a |
| success bias (predicted minus true success rate, decided) | n/a |
| false alarms (failure called on a true success) | 0 |
| false alarm rate per true success | 0.000 [0.000, 0.066] (n=54) |
| false alarms per hour of robot time | 0.00 [0.00, 3.26] |

## Per task

| task | n | abstain | balanced accuracy | false alarms |
|---|---|---|---|---|
| domotic_dishTidyUp_anomaly | 8 | 8 | n/a | 0 |
| domotic_dishTidyUp_expert | 3 | 3 | n/a | 0 |
| domotic_groceriesSorting_anomaly | 1 | 1 | n/a | 0 |
| domotic_groceriesSorting_expert | 4 | 4 | n/a | 0 |
| domotic_makingCoffee_anomaly | 3 | 3 | n/a | 0 |
| domotic_makingCoffee_expert | 11 | 11 | n/a | 0 |
| domotic_pouringCoffee_anomaly | 8 | 8 | n/a | 0 |
| domotic_pouringCoffee_expert | 10 | 10 | n/a | 0 |
| domotic_setTheTable_anomaly | 3 | 3 | n/a | 0 |
| domotic_setTheTable_expert | 2 | 2 | n/a | 0 |
| domotic_vegetablesAndFruitsSorting_anomaly | 7 | 7 | n/a | 0 |
| domotic_vegetablesAndFruitsSorting_expert | 5 | 5 | n/a | 0 |
| industrial_robothon_buttons_anomaly | 3 | 3 | n/a | 0 |
| industrial_robothon_buttons_expert | 3 | 3 | n/a | 0 |
| industrial_robothon_hatchAndProbe_anomaly | 3 | 3 | n/a | 0 |
| industrial_robothon_hatchAndProbe_expert | 5 | 5 | n/a | 0 |
| industrial_screws_sorting_anomaly | 1 | 1 | n/a | 0 |
| industrial_screws_sorting_expert | 4 | 4 | n/a | 0 |
| industrial_soldering_anomaly | 6 | 6 | n/a | 0 |
| industrial_soldering_expert | 7 | 7 | n/a | 0 |

Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.
Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks.


# fused judge

Episodes: 97 (25 decided, 72 abstained), robot time 1.10 h. Intervals are 95% Wilson.

| metric | value |
|---|---|
| abstain rate | 0.742 [0.647, 0.819] (n=97) |
| coverage | 0.258 [0.181, 0.353] (n=97) |
| accuracy on decided | 0.880 [0.700, 0.958] (n=25) |
| balanced accuracy on decided | 0.921 [0.617, 0.972] |
| success recall (TPR) | 0.842 [0.624, 0.945] (n=19) |
| failure recall (TNR) | 1.000 [0.610, 1.000] (n=6) |
| failure precision | 0.667 [0.354, 0.879] (n=9) |
| success bias (predicted minus true success rate, decided) | -0.120 |
| false alarms (failure called on a true success) | 3 |
| false alarm rate per true success | 0.056 [0.019, 0.151] (n=54) |
| false alarms per hour of robot time | 2.73 [0.94, 7.41] |

## Per task

| task | n | abstain | balanced accuracy | false alarms |
|---|---|---|---|---|
| domotic_dishTidyUp_anomaly | 8 | 8 | n/a | 0 |
| domotic_dishTidyUp_expert | 3 | 1 | n/a | 0 |
| domotic_groceriesSorting_anomaly | 1 | 1 | n/a | 0 |
| domotic_groceriesSorting_expert | 4 | 4 | n/a | 0 |
| domotic_makingCoffee_anomaly | 3 | 3 | n/a | 0 |
| domotic_makingCoffee_expert | 11 | 11 | n/a | 0 |
| domotic_pouringCoffee_anomaly | 8 | 7 | n/a | 0 |
| domotic_pouringCoffee_expert | 10 | 0 | n/a | 0 |
| domotic_setTheTable_anomaly | 3 | 1 | n/a | 0 |
| domotic_setTheTable_expert | 2 | 0 | n/a | 0 |
| domotic_vegetablesAndFruitsSorting_anomaly | 7 | 7 | n/a | 0 |
| domotic_vegetablesAndFruitsSorting_expert | 5 | 5 | n/a | 0 |
| industrial_robothon_buttons_anomaly | 3 | 3 | n/a | 0 |
| industrial_robothon_buttons_expert | 3 | 1 | n/a | 2 |
| industrial_robothon_hatchAndProbe_anomaly | 3 | 3 | n/a | 0 |
| industrial_robothon_hatchAndProbe_expert | 5 | 4 | n/a | 1 |
| industrial_screws_sorting_anomaly | 1 | 0 | n/a | 0 |
| industrial_screws_sorting_expert | 4 | 2 | n/a | 0 |
| industrial_soldering_anomaly | 6 | 4 | n/a | 0 |
| industrial_soldering_expert | 7 | 7 | n/a | 0 |

Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.
Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks.


# VLM-only judge (uncalibrated)

Episodes: 97 (97 decided, 0 abstained), robot time 0.81 h. Intervals are 95% Wilson.

| metric | value |
|---|---|
| abstain rate | 0.000 [0.000, 0.038] (n=97) |
| coverage | 1.000 [0.962, 1.000] (n=97) |
| accuracy on decided | 0.639 [0.540, 0.728] (n=97) |
| balanced accuracy on decided | 0.614 [0.488, 0.727] |
| success recall (TPR) | 0.833 [0.713, 0.910] (n=54) |
| failure recall (TNR) | 0.395 [0.264, 0.544] (n=43) |
| failure precision | 0.654 [0.462, 0.806] (n=26) |
| success bias (predicted minus true success rate, decided) | +0.175 |
| false alarms (failure called on a true success) | 9 |
| false alarm rate per true success | 0.167 [0.090, 0.287] (n=54) |
| false alarms per hour of robot time | 11.13 [6.03, 19.20] |

Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.
Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks.

