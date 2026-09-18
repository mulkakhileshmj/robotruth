# Judge on live ACT rollouts (action stream only, env success as truth)

Episodes: 24 (24 decided, 0 abstained), robot time 0.04 h. Intervals are 95% Wilson.

| metric | value |
|---|---|
| abstain rate | 0.000 [0.000, 0.138] (n=24) |
| coverage | 1.000 [0.862, 1.000] (n=24) |
| accuracy on decided | 0.958 [0.798, 0.993] (n=24) |
| balanced accuracy on decided | 0.929 [0.651, 0.987] |
| success recall (TPR) | 1.000 [0.816, 1.000] (n=17) |
| failure recall (TNR) | 0.857 [0.487, 0.974] (n=7) |
| failure precision | 1.000 [0.610, 1.000] (n=6) |
| success bias (predicted minus true success rate, decided) | +0.042 |
| false alarms (failure called on a true success) | 0 |
| false alarm rate per true success | 0.000 [0.000, 0.184] (n=17) |
| false alarms per hour of robot time | 0.00 [0.00, 73.78] |

## Per task

| task | n | abstain | balanced accuracy | false alarms |
|---|---|---|---|---|
| aloha_transfer_cube | 24 | 0 | 0.929 | 0 |

Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.
Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks.


Too few live failures for conformal calibration; judge evaluated uncalibrated (threshold 0.5, no abstain). This is the honest small-sample mode: the policy succeeds too often to collect 5 failures per split.