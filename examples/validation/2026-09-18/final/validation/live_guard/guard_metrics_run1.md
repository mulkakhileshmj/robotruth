## Guard on live ACT rollouts (AlohaTransferCube)

- Episodes: 34 (4 failures, 20 successes, 10 unlabelled)

### Detection on true failures

- Detection rate: 0.500 [0.150, 0.850] (95% Wilson, n=4)
- Warning before episode end: mean 5.2 s, median 5.2 s (over 2 detected)
- Hard-limit stops on failures: 0

### False alarms on nominal operation

- False alarms per hour: 0.00 [0.00, 116.47] (0 alarms in 0.03 h, 95% Poisson)
- Nominal episodes with any alarm: 0.000 [0.000, 0.161] (95% Wilson, n=20) (this is the rate the conformal alpha bounds)
- Hard-limit stops on nominal episodes: 0

- 10 episodes without a truth label were ignored.

- Perturbed episodes (observation shift at t=3.0 s): 5/10 alerted after onset, 0 alerted before onset.
- Alerts on successful nominal episodes: 0 in 0.03 h (0.0/h)