## Guard on live ACT rollouts (AlohaTransferCube)

- Episodes: 34 (7 failures, 17 successes, 10 unlabelled)

### Detection on true failures

- Detection rate: 0.000 [0.000, 0.354] (95% Wilson, n=7)
- Hard-limit stops on failures: 0

### False alarms on nominal operation

- False alarms per hour: 0.00 [0.00, 136.37] (0 alarms in 0.03 h, 95% Poisson)
- Nominal episodes with any alarm: 0.000 [0.000, 0.184] (95% Wilson, n=17) (this is the rate the conformal alpha bounds)
- Hard-limit stops on nominal episodes: 0

- 10 episodes without a truth label were ignored.

- Perturbed episodes (observation shift at t=3.0 s): 0/10 alerted after onset, 0 alerted before onset.
- Alerts on successful nominal episodes: 0 in 0.03 h (0.0/h)