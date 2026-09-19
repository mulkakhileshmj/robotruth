# BotFails pooled

## Fleet metrics

- Episodes: 323 (323 with an outcome)
- Success rate: 0.551 [0.497, 0.604] (95% Wilson, n=323)
- Autonomous fraction (no teleop, reset or stop): 1.000 [0.988, 1.000] (95% Wilson, n=323)
- Robot hours logged: 3.83
- Interventions per hour: 0.00 [0.00, 0.96] (0 interventions)
- Share of time under intervention: 0.0%

| policy | n | success | autonomous | interventions/h |
|---|---|---|---|---|
| dataset:domotic_dishTidyUp_anomaly | 15 | 0.00 [0.00, 0.20] | 1.00 | 0.00 |
| dataset:domotic_dishTidyUp_expert | 20 | 1.00 [0.84, 1.00] | 1.00 | 0.00 |
| dataset:domotic_groceriesSorting_anomaly | 10 | 0.00 [0.00, 0.28] | 1.00 | 0.00 |
| dataset:domotic_groceriesSorting_expert | 20 | 1.00 [0.84, 1.00] | 1.00 | 0.00 |
| dataset:domotic_makingCoffee_anomaly | 15 | 0.00 [0.00, 0.20] | 1.00 | 0.00 |
| dataset:domotic_makingCoffee_expert | 20 | 1.00 [0.84, 1.00] | 1.00 | 0.00 |
| dataset:domotic_pouringCoffee_anomaly | 20 | 0.00 [0.00, 0.16] | 1.00 | 0.00 |
| dataset:domotic_pouringCoffee_expert | 20 | 1.00 [0.84, 1.00] | 1.00 | 0.00 |
| dataset:domotic_setTheTable_anomaly | 15 | 0.00 [0.00, 0.20] | 1.00 | 0.00 |
| dataset:domotic_setTheTable_expert | 18 | 1.00 [0.82, 1.00] | 1.00 | 0.00 |
| dataset:domotic_vegetablesAndFruitsSorting_anomaly | 20 | 0.00 [0.00, 0.16] | 1.00 | 0.00 |
| dataset:domotic_vegetablesAndFruitsSorting_expert | 20 | 1.00 [0.84, 1.00] | 1.00 | 0.00 |
| dataset:industrial_robothon_buttons_anomaly | 10 | 0.00 [0.00, 0.28] | 1.00 | 0.00 |
| dataset:industrial_robothon_buttons_expert | 15 | 1.00 [0.80, 1.00] | 1.00 | 0.00 |
| dataset:industrial_robothon_hatchAndProbe_anomaly | 15 | 0.00 [0.00, 0.20] | 1.00 | 0.00 |
| dataset:industrial_robothon_hatchAndProbe_expert | 10 | 1.00 [0.72, 1.00] | 1.00 | 0.00 |
| dataset:industrial_screws_sorting_anomaly | 10 | 0.00 [0.00, 0.28] | 1.00 | 0.00 |
| dataset:industrial_screws_sorting_expert | 20 | 1.00 [0.84, 1.00] | 1.00 | 0.00 |
| dataset:industrial_soldering_anomaly | 15 | 0.00 [0.00, 0.20] | 1.00 | 0.00 |
| dataset:industrial_soldering_expert | 15 | 1.00 [0.80, 1.00] | 1.00 | 0.00 |

## Unit fingerprint

Per-episode unit fingerprints from 323 episodes (action = commanded, observation.state = measured). Median [IQR].

| joint | lag ms | backlash | rmse | offset | gain |
|---|---|---|---|---|---|
| j0 | 110.2 [105, 115] | 0.4764 [0.314, 0.659] | 0.6818 [0.637, 0.745] | 0.3279 [0.144, 0.455] | 0.969 [0.94, 0.977] |
| j1 | 114.4 [109, 118] | 0.7589 [0.524, 1.14] | 1.939 [1.63, 2.12] | -1.348 [-1.58, -0.977] | 0.9769 [0.969, 0.984] |
| j2 | 113.5 [111, 117] | 0.8008 [0.475, 1.29] | 1.013 [0.911, 1.21] | 0.3597 [0.202, 0.515] | 0.9426 [0.931, 0.951] |
| j3 | 112.5 [108, 117] | 0.3878 [0.241, 0.571] | 0.6677 [0.6, 0.747] | -0.3792 [-0.494, -0.155] | 0.9403 [0.925, 0.952] |
| j4 | 123.9 [118, 130] | 0.3287 [0.161, 0.718] | 0.7611 [0.673, 0.936] | -0.3545 [-0.549, -0.0622] | 0.9341 [0.886, 0.953] |
| j5 | 117 [106, 127] | 0.6224 [0.259, 1.26] | 1.36 [0.946, 2.22] | 0.7129 [0.302, 1.37] | 0.7968 [0.749, 0.834] |

## Judge

# Judge on real episodes (action-stream features only, no VLM)

Episodes: 97 (5 decided, 92 abstained), robot time 1.13 h. Intervals are 95% Wilson.

| metric | value |
|---|---|
| abstain rate | 0.948 [0.885, 0.978] (n=97) |
| coverage | 0.052 [0.022, 0.115] (n=97) |
| accuracy on decided | 0.800 [0.376, 0.964] (n=5) |
| balanced accuracy on decided | 0.500 [0.255, 0.897] |
| success recall (TPR) | 1.000 [0.510, 1.000] (n=4) |
| failure recall (TNR) | 0.000 [0.000, 0.793] (n=1) |
| failure precision | n/a |
| success bias (predicted minus true success rate, decided) | +0.200 |
| false alarms (failure called on a true success) | 0 |
| false alarm rate per true success | 0.000 [0.000, 0.066] (n=54) |
| false alarms per hour of robot time | 0.00 [0.00, 3.18] |

## Per task

| task | n | abstain | balanced accuracy | false alarms |
|---|---|---|---|---|
| 0:4 Normal test samples, 5:9 place the soldering iron on wood of on its base, 10:14 user places his hand between the soldering iron and the electronic card | 6 | 5 | n/a | 0 |
| 0:4 expert test sample, 5:14 items in the wrog box | 6 | 6 | n/a | 0 |
| 0:4 expert test sample, 5:9 bad disposition, 10:14 dropping items | 4 | 4 | n/a | 0 |
| 0:4 expert test samples, 5:9 storing the dishes the wrong way, 10:14 dropping dishes | 3 | 3 | n/a | 0 |
| 0:4 normal test episodes,5:9 pour the coffee on the table, 10:14 there is no container in the machine  | 6 | 6 | n/a | 0 |
| 2 kinds of anomalies while sorting screws: 1.bad sorting = screws in the wrong plate, 2.dropping some screws | 4 | 4 | n/a | 0 |
| 3 anomalies : forget to close the hatch, measure tension in the wrong hole, put the probe on the red hole instead of the black | 2 | 2 | n/a | 0 |
| Set the table: one glass, one plate, one fork and one knife. Fork on the left of the plate and knife on the right | 7 | 7 | n/a | 0 |
| anomalies while pouring coffe in a cup: 0-4:pour coffee incorrectly and put a lot on the side, 5-9:pour coffee next to the cup, 10-14:spill the cup while full, 15-19:normal samples for test | 8 | 8 | n/a | 0 |
| making coffee: take some coffee powder with the spoon and pour it into the filter of the coffee machine | 6 | 6 | n/a | 0 |
| open the hatch, grab the probe, measure tension, place the probe in the black plug, close the hatch | 5 | 5 | n/a | 0 |
| pouring coffe in a cup | 8 | 6 | n/a | 0 |
| push the red button then move the slider to the top and push the blue button | 4 | 4 | n/a | 0 |
| several anomalies: slider not completely pushed / inverse red and blue | 4 | 4 | n/a | 0 |
| soldering a part on an electronic board | 2 | 2 | n/a | 0 |
| sorting 2 types of screws  | 3 | 3 | n/a | 0 |
| sorting fruits and vegetables, put vegetables in the box with a yellow mark and fruits in the box with a pink mark | 5 | 4 | n/a | 0 |
| sorting groceries, vegetables in the box with a pink sign and sweets in the box with the yellow sign | 6 | 6 | n/a | 0 |
| store dishes in a dish rack | 8 | 7 | n/a | 0 |

Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.
Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks.


Conformal calibration with abstain fitted on a held-out split (target selective error 0.10).

## Guard

## Guard on real episodes (chunk consistency and action stats from pseudo-chunks, conformal max threshold)

- Episodes: 217 (145 failures, 72 successes, 0 unlabelled)

### Detection on true failures

- Detection rate: 0.041 [0.019, 0.087] (95% Wilson, n=145)
- Warning before episode end: mean 27.1 s, median 26.7 s (over 6 detected)
- Hard-limit stops on failures: 0

### False alarms on nominal operation

- False alarms per hour: 0.00 [0.00, 4.41] (0 alarms in 0.84 h, 95% Poisson)
- Nominal episodes with any alarm: 0.000 [0.000, 0.051] (95% Wilson, n=72) (this is the rate the conformal alpha bounds)
- Hard-limit stops on nominal episodes: 0

Calibrated on 106 nominal episodes, replayed 72 held-out nominal and 145 failures. Nominal pool = successes.
