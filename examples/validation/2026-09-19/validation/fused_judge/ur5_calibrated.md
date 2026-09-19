# ur5fail: VLM channel, uncalibrated vs conformal-calibrated with abstain

Backend open:Qwen/Qwen2.5-VL-7B-Instruct. 51 calibration / 51 test episodes (stratified).

# Uncalibrated (threshold 0.5, no abstain)

Episodes: 51 (51 decided, 0 abstained), robot time 0.42 h. Intervals are 95% Wilson.

| metric | value |
|---|---|
| abstain rate | 0.000 [0.000, 0.070] (n=51) |
| coverage | 1.000 [0.930, 1.000] (n=51) |
| accuracy on decided | 0.549 [0.414, 0.677] (n=51) |
| balanced accuracy on decided | 0.550 [0.364, 0.723] |
| success recall (TPR) | 0.500 [0.321, 0.679] (n=26) |
| failure recall (TNR) | 0.600 [0.407, 0.766] (n=25) |
| failure precision | 0.536 [0.358, 0.705] (n=28) |
| success bias (predicted minus true success rate, decided) | -0.059 |
| false alarms (failure called on a true success) | 13 |
| false alarm rate per true success | 0.500 [0.321, 0.679] (n=26) |
| false alarms per hour of robot time | 30.59 [19.61, 41.56] |

Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.
Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks.


# Calibrated with abstain (target selective error 0.10)

Episodes: 51 (0 decided, 51 abstained), robot time 0.42 h. Intervals are 95% Wilson.

| metric | value |
|---|---|
| abstain rate | 1.000 [0.930, 1.000] (n=51) |
| coverage | 0.000 [0.000, 0.070] (n=51) |
| accuracy on decided | n/a |
| balanced accuracy on decided | n/a |
| success recall (TPR) | n/a |
| failure recall (TNR) | n/a |
| failure precision | n/a |
| success bias (predicted minus true success rate, decided) | n/a |
| false alarms (failure called on a true success) | 0 |
| false alarm rate per true success | 0.000 [0.000, 0.129] (n=26) |
| false alarms per hour of robot time | 0.00 [0.00, 7.88] |

Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.
Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks.
