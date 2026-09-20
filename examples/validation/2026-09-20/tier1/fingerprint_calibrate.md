# Calibrated drift detection on real robot logs

`DriftDetector` at alpha = 0.05, watching per-joint steady-state error, against the three-sigma rule it replaces. Both are fitted on the same clean episodes and measured on the same held-out ones, then on the same episodes with a constant command offset injected.

The threshold is a split-conformal quantile of the worst standardised per-joint deviation, so its false-alarm rate is bounded by alpha under exchangeability alone, with no assumption that per-joint estimates are Gaussian or independent.

## Elvinky__pi05-piperx-7h-demo-dagger-full-episodes-20260914

120 episodes: 72 to fit, 48 held out. DriftDetector: alpha=0.05, threshold=5.775 standardised units, fitted on 36 and calibrated on 36 fingerprints

| | false alarms (clean) | offset 0.25 sd | offset 0.5 sd | offset 1 sd | offset 2 sd |
|---|---|---|---|---|---|
| conformal, alpha 0.05 | 0/48 = 0.000 [0.000, 0.074] (95% Wilson, n=48) | 48/48 | 48/48 | 48/48 | 48/48 |
| three-sigma rule | 3/48 = 0.062 [0.021, 0.168] (95% Wilson, n=48) | 48/48 | 48/48 | 48/48 | 48/48 |

## jccj__so100_block_in_cup

50 episodes: 30 to fit, 20 held out. DriftDetector: alpha=0.05, threshold=15.556 standardised units, fitted on 15 and calibrated on 15 fingerprints; SATURATED, too few calibration fingerprints for alpha=0.05, smallest certifiable is 0.062

| | false alarms (clean) | offset 0.25 sd | offset 0.5 sd | offset 1 sd | offset 2 sd |
|---|---|---|---|---|---|
| conformal, alpha 0.05 | 0/20 = 0.000 [0.000, 0.161] (95% Wilson, n=20) | 20/20 | 20/20 | 20/20 | 20/20 |
| three-sigma rule | 0/20 = 0.000 [0.000, 0.161] (95% Wilson, n=20) | 20/20 | 20/20 | 20/20 | 20/20 |

## kantine__BotFails

133 episodes: 79 to fit, 54 held out. DriftDetector: alpha=0.05, threshold=9.868 standardised units, fitted on 40 and calibrated on 39 fingerprints

| | false alarms (clean) | offset 0.25 sd | offset 0.5 sd | offset 1 sd | offset 2 sd |
|---|---|---|---|---|---|
| conformal, alpha 0.05 | 0/54 = 0.000 [0.000, 0.066] (95% Wilson, n=54) | 53/54 | 53/54 | 53/54 | 53/54 |
| three-sigma rule | 2/54 = 0.037 [0.010, 0.125] (95% Wilson, n=54) | 53/54 | 53/54 | 53/54 | 53/54 |

## lerobot__aloha_static_screw_driver

50 episodes: 30 to fit, 20 held out. DriftDetector: alpha=0.05, threshold=4.696 standardised units, fitted on 15 and calibrated on 15 fingerprints; SATURATED, too few calibration fingerprints for alpha=0.05, smallest certifiable is 0.062

| | false alarms (clean) | offset 0.25 sd | offset 0.5 sd | offset 1 sd | offset 2 sd |
|---|---|---|---|---|---|
| conformal, alpha 0.05 | 0/20 = 0.000 [0.000, 0.161] (95% Wilson, n=20) | 20/20 | 20/20 | 20/20 | 20/20 |
| three-sigma rule | 4/20 = 0.200 [0.081, 0.416] (95% Wilson, n=20) | 20/20 | 20/20 | 20/20 | 20/20 |

## lerobot__droid_1.0.1

119 episodes: 71 to fit, 48 held out. DriftDetector: alpha=0.05, threshold=6.056 standardised units, fitted on 36 and calibrated on 35 fingerprints

| | false alarms (clean) | offset 0.25 sd | offset 0.5 sd | offset 1 sd | offset 2 sd |
|---|---|---|---|---|---|
| conformal, alpha 0.05 | 0/48 = 0.000 [0.000, 0.074] (95% Wilson, n=48) | 47/48 | 48/48 | 48/48 | 48/48 |
| three-sigma rule | 3/48 = 0.062 [0.021, 0.168] (95% Wilson, n=48) | 48/48 | 48/48 | 48/48 | 48/48 |

## lerobot__svla_so100_pickplace

50 episodes: 30 to fit, 20 held out. DriftDetector: alpha=0.05, threshold=5.242 standardised units, fitted on 15 and calibrated on 15 fingerprints; SATURATED, too few calibration fingerprints for alpha=0.05, smallest certifiable is 0.062

| | false alarms (clean) | offset 0.25 sd | offset 0.5 sd | offset 1 sd | offset 2 sd |
|---|---|---|---|---|---|
| conformal, alpha 0.05 | 2/20 = 0.100 [0.028, 0.301] (95% Wilson, n=20) | 20/20 | 20/20 | 20/20 | 20/20 |
| three-sigma rule | 6/20 = 0.300 [0.145, 0.519] (95% Wilson, n=20) | 20/20 | 20/20 | 20/20 | 20/20 |

## lerobot__svla_so101_pickplace

50 episodes: 30 to fit, 20 held out. DriftDetector: alpha=0.05, threshold=3.306 standardised units, fitted on 15 and calibrated on 15 fingerprints; SATURATED, too few calibration fingerprints for alpha=0.05, smallest certifiable is 0.062

| | false alarms (clean) | offset 0.25 sd | offset 0.5 sd | offset 1 sd | offset 2 sd |
|---|---|---|---|---|---|
| conformal, alpha 0.05 | 1/20 = 0.050 [0.009, 0.236] (95% Wilson, n=20) | 20/20 | 20/20 | 20/20 | 20/20 |
| three-sigma rule | 1/20 = 0.050 [0.009, 0.236] (95% Wilson, n=20) | 20/20 | 20/20 | 20/20 | 20/20 |

## sixpigs1__so100_pick_cube_in_box

100 episodes: 60 to fit, 40 held out. DriftDetector: alpha=0.05, threshold=32.450 standardised units, fitted on 30 and calibrated on 30 fingerprints

| | false alarms (clean) | offset 0.25 sd | offset 0.5 sd | offset 1 sd | offset 2 sd |
|---|---|---|---|---|---|
| conformal, alpha 0.05 | 0/40 = 0.000 [0.000, 0.088] (95% Wilson, n=40) | 40/40 | 40/40 | 40/40 | 40/40 |
| three-sigma rule | 4/40 = 0.100 [0.040, 0.231] (95% Wilson, n=40) | 40/40 | 40/40 | 40/40 | 40/40 |


FINGERPRINT_CALIBRATE_COMPLETE