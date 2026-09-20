# Guard on real robot logs, with execution faults injected

Real successful episodes from physical robots, with a freeze, a constant offset or added noise applied to the recorded action stream from the midpoint onward. The unfaulted copies of the same episodes give the false-alarm rate on identical data, so detection and false alarms are measured on one population.

This tests whether the scorers transfer to real robot data. It does not claim the faults are naturally occurring, and it is replay rather than closed-loop control.

## jpizarrom__hilserl_so100_grocery_so100_2025112223_20

Robot `unknown`, 20 Hz. Calibrated on 228 real successful episodes, evaluated on 153 held out. Inputs: state features and action chunks.

False alarms on the clean held-out episodes: 0/153 = 0.000 [0.000, 0.024] (95% Wilson, n=153) (bound 0.05).

| injected fault | detected | rate | median latency after onset |
|---|---|---|---|
| freeze | 153/153 | 1.000 [0.976, 1.000] (95% Wilson, n=153) | 0.15 s |
| offset 0.25 sd | 0/153 | 0.000 [0.000, 0.024] (95% Wilson, n=153) | n/a |
| offset 0.5 sd | 0/153 | 0.000 [0.000, 0.024] (95% Wilson, n=153) | n/a |
| offset 1 sd | 0/153 | 0.000 [0.000, 0.024] (95% Wilson, n=153) | n/a |
| offset 2 sd | 0/153 | 0.000 [0.000, 0.024] (95% Wilson, n=153) | n/a |
| offset 4 sd | 2/153 | 0.013 [0.004, 0.046] (95% Wilson, n=153) | 5.82 s |
| noise 0.25 sd | 153/153 | 1.000 [0.976, 1.000] (95% Wilson, n=153) | 0.10 s |
| noise 0.5 sd | 153/153 | 1.000 [0.976, 1.000] (95% Wilson, n=153) | 0.10 s |
| noise 1 sd | 153/153 | 1.000 [0.976, 1.000] (95% Wilson, n=153) | 0.10 s |
| noise 2 sd | 153/153 | 1.000 [0.976, 1.000] (95% Wilson, n=153) | 0.10 s |
| noise 4 sd | 153/153 | 1.000 [0.976, 1.000] (95% Wilson, n=153) | 0.10 s |

## kantine__BotFails

Robot `so100`, 30 Hz. Calibrated on 106 real successful episodes, evaluated on 72 held out. Inputs: state features and action chunks.

False alarms on the clean held-out episodes: 0/72 = 0.000 [0.000, 0.051] (95% Wilson, n=72) (bound 0.05).

| injected fault | detected | rate | median latency after onset |
|---|---|---|---|
| freeze | 72/72 | 1.000 [0.949, 1.000] (95% Wilson, n=72) | 0.10 s |
| offset 0.25 sd | 0/72 | 0.000 [0.000, 0.051] (95% Wilson, n=72) | n/a |
| offset 0.5 sd | 0/72 | 0.000 [0.000, 0.051] (95% Wilson, n=72) | n/a |
| offset 1 sd | 2/72 | 0.028 [0.008, 0.096] (95% Wilson, n=72) | 3.03 s |
| offset 2 sd | 50/72 | 0.694 [0.580, 0.789] (95% Wilson, n=72) | 0.07 s |
| offset 4 sd | 72/72 | 1.000 [0.949, 1.000] (95% Wilson, n=72) | 0.07 s |
| noise 0.25 sd | 72/72 | 1.000 [0.949, 1.000] (95% Wilson, n=72) | 0.07 s |
| noise 0.5 sd | 72/72 | 1.000 [0.949, 1.000] (95% Wilson, n=72) | 0.07 s |
| noise 1 sd | 72/72 | 1.000 [0.949, 1.000] (95% Wilson, n=72) | 0.07 s |
| noise 2 sd | 72/72 | 1.000 [0.949, 1.000] (95% Wilson, n=72) | 0.07 s |
| noise 4 sd | 72/72 | 1.000 [0.949, 1.000] (95% Wilson, n=72) | 0.07 s |

## lerobot__droid_1.0.1

Robot `Franka`, 15 Hz. Calibrated on 204 real successful episodes, evaluated on 137 held out. Inputs: state features and action chunks.

False alarms on the clean held-out episodes: 2/137 = 0.015 [0.004, 0.052] (95% Wilson, n=137) (bound 0.05).

| injected fault | detected | rate | median latency after onset |
|---|---|---|---|
| freeze | 136/137 | 0.993 [0.960, 0.999] (95% Wilson, n=137) | 0.20 s |
| offset 0.25 sd | 4/137 | 0.029 [0.011, 0.073] (95% Wilson, n=137) | 2.93 s |
| offset 0.5 sd | 10/137 | 0.073 [0.040, 0.129] (95% Wilson, n=137) | 2.63 s |
| offset 1 sd | 62/137 | 0.453 [0.372, 0.536] (95% Wilson, n=137) | 0.13 s |
| offset 2 sd | 124/137 | 0.905 [0.844, 0.944] (95% Wilson, n=137) | 0.13 s |
| offset 4 sd | 136/137 | 0.993 [0.960, 0.999] (95% Wilson, n=137) | 0.13 s |
| noise 0.25 sd | 136/137 | 0.993 [0.960, 0.999] (95% Wilson, n=137) | 0.13 s |
| noise 0.5 sd | 136/137 | 0.993 [0.960, 0.999] (95% Wilson, n=137) | 0.13 s |
| noise 1 sd | 136/137 | 0.993 [0.960, 0.999] (95% Wilson, n=137) | 0.13 s |
| noise 2 sd | 136/137 | 0.993 [0.960, 0.999] (95% Wilson, n=137) | 0.13 s |
| noise 4 sd | 136/137 | 0.993 [0.960, 0.999] (95% Wilson, n=137) | 0.13 s |


REAL_FAULTS_COMPLETE