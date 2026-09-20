# Do published failure signals see real task failures in the action stream?

AUROC of each detector's per-episode peak score, real failures against held-out successes, and the same detectors against execution faults injected into those same held-out successes. Every detector is fitted on successful episodes only. 0.5 is a coin flip, and AUROC needs no threshold, so a low number here cannot be explained by bad calibration.

## kantine__BotFails

Robot `so100`, 30 Hz. Fitted on 106 successes, evaluated on 72 held-out successes and 145 real failures.

| detector | real task failures | freeze | offset 2 sd | noise 0.5 sd |
|---|---|---|---|---|
| chunk_consistency | 0.505 [0.44, 0.59] | 0.397 [0.30, 0.49] | 0.986 [0.96, 1.00] | 0.984 [0.96, 1.00] |
| mahalanobis | 0.528 [0.45, 0.61] | 0.988 [0.97, 1.00] | 0.991 [0.98, 1.00] | 0.979 [0.95, 1.00] |
| action_variance | 0.501 [0.43, 0.59] | 0.389 [0.30, 0.48] | 0.500 [0.40, 0.60] | 0.982 [0.95, 1.00] |
| density | 0.510 [0.43, 0.59] | 0.505 [0.40, 0.61] | 0.945 [0.90, 0.99] | 0.858 [0.79, 0.92] |
| pca_recon | 0.526 [0.45, 0.61] | 0.988 [0.96, 1.00] | 0.995 [0.98, 1.00] | 0.981 [0.95, 1.00] |
| iforest | 0.489 [0.41, 0.57] | 0.426 [0.34, 0.52] | 0.716 [0.63, 0.80] | 0.598 [0.51, 0.69] |
| ocsvm | 0.523 [0.45, 0.60] | 0.900 [0.84, 0.95] | 0.987 [0.97, 1.00] | 0.858 [0.79, 0.92] |
| rnd | 0.537 [0.46, 0.61] | 0.985 [0.96, 1.00] | 0.987 [0.96, 1.00] | 0.933 [0.88, 0.97] |
| lstm_pred | 0.555 [0.47, 0.64] | 0.916 [0.86, 0.96] | 0.957 [0.93, 0.98] | 0.750 [0.66, 0.84] |
| ours | 0.524 [0.49, 0.56] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |

## lerobot__droid_1.0.1

Robot `Franka`, 15 Hz. Fitted on 204 successes, evaluated on 137 held-out successes and 58 real failures.

| detector | real task failures | freeze | offset 2 sd | noise 0.5 sd |
|---|---|---|---|---|
| chunk_consistency | 0.313 [0.23, 0.39] | 0.261 [0.20, 0.32] | 0.998 [1.00, 1.00] | 0.977 [0.96, 0.99] |
| mahalanobis | 0.400 [0.31, 0.48] | 0.988 [0.98, 1.00] | 1.000 [1.00, 1.00] | 0.993 [0.99, 1.00] |
| action_variance | 0.313 [0.23, 0.40] | 0.264 [0.21, 0.32] | 0.501 [0.43, 0.57] | 0.976 [0.96, 0.99] |
| density | 0.394 [0.31, 0.48] | 0.475 [0.40, 0.54] | 0.767 [0.71, 0.82] | 0.627 [0.56, 0.69] |
| pca_recon | 0.356 [0.28, 0.44] | 0.988 [0.97, 1.00] | 1.000 [1.00, 1.00] | 0.987 [0.97, 1.00] |
| iforest | 0.471 [0.37, 0.56] | 0.447 [0.37, 0.52] | 0.653 [0.58, 0.72] | 0.546 [0.48, 0.62] |
| ocsvm | 0.505 [0.42, 0.59] | 0.539 [0.47, 0.60] | 0.808 [0.75, 0.86] | 0.613 [0.54, 0.68] |
| rnd | 0.432 [0.34, 0.51] | 0.887 [0.85, 0.92] | 0.978 [0.96, 0.99] | 0.813 [0.75, 0.86] |
| lstm_pred | 0.567 [0.48, 0.65] | 0.786 [0.73, 0.84] | 0.956 [0.93, 0.98] | 0.701 [0.63, 0.76] |
| ours | 0.461 [0.36, 0.55] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |

Total wall time 110 s.

BASELINE_COMPARISON_COMPLETE