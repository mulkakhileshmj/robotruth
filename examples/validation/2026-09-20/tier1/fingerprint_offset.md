# Constant offsets: the fingerprint catches what the guard cannot

The same constant command offsets that the runtime guard needs about two standard deviations to notice, put through the unit fingerprint instead. The fingerprint regresses commanded against measured joint positions over a whole episode and reports a steady-state error per joint. An episode counts as detected when any joint's offset estimate falls more than three nominal standard deviations from the clean mean for that joint.

This is an offline, between-sessions check, not a per-step one. It is the right tool for drift and miscalibration; the guard is the right tool for stalls and erratic control. Neither detects a robot that moves normally and still fails the task, which is what the vision judge is for.

## Elvinky__pi05-piperx-7h-demo-dagger-full-episodes-20260914

120 episodes with both commanded and measured streams; 72 used for the nominal spread, 48 held out.

False alarms on clean held-out episodes: 3/48 = 0.062 [0.021, 0.168] (95% Wilson, n=48).

| injected offset | fingerprint detects | rate |
|---|---|---|
| 0.25 sd | 48/48 | 1.000 [0.926, 1.000] (95% Wilson, n=48) |
| 0.5 sd | 48/48 | 1.000 [0.926, 1.000] (95% Wilson, n=48) |
| 1 sd | 48/48 | 1.000 [0.926, 1.000] (95% Wilson, n=48) |
| 2 sd | 48/48 | 1.000 [0.926, 1.000] (95% Wilson, n=48) |

## jccj__so100_block_in_cup

50 episodes with both commanded and measured streams; 30 used for the nominal spread, 20 held out.

False alarms on clean held-out episodes: 0/20 = 0.000 [0.000, 0.161] (95% Wilson, n=20).

| injected offset | fingerprint detects | rate |
|---|---|---|
| 0.25 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 0.5 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 1 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 2 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |

## kantine__BotFails

133 episodes with both commanded and measured streams; 79 used for the nominal spread, 54 held out.

False alarms on clean held-out episodes: 2/54 = 0.037 [0.010, 0.125] (95% Wilson, n=54).

| injected offset | fingerprint detects | rate |
|---|---|---|
| 0.25 sd | 53/54 | 0.981 [0.902, 0.997] (95% Wilson, n=54) |
| 0.5 sd | 53/54 | 0.981 [0.902, 0.997] (95% Wilson, n=54) |
| 1 sd | 53/54 | 0.981 [0.902, 0.997] (95% Wilson, n=54) |
| 2 sd | 53/54 | 0.981 [0.902, 0.997] (95% Wilson, n=54) |

## lerobot__aloha_static_screw_driver

50 episodes with both commanded and measured streams; 30 used for the nominal spread, 20 held out.

False alarms on clean held-out episodes: 4/20 = 0.200 [0.081, 0.416] (95% Wilson, n=20).

| injected offset | fingerprint detects | rate |
|---|---|---|
| 0.25 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 0.5 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 1 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 2 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |

## lerobot__droid_1.0.1

119 episodes with both commanded and measured streams; 71 used for the nominal spread, 48 held out.

False alarms on clean held-out episodes: 3/48 = 0.062 [0.021, 0.168] (95% Wilson, n=48).

| injected offset | fingerprint detects | rate |
|---|---|---|
| 0.25 sd | 48/48 | 1.000 [0.926, 1.000] (95% Wilson, n=48) |
| 0.5 sd | 48/48 | 1.000 [0.926, 1.000] (95% Wilson, n=48) |
| 1 sd | 48/48 | 1.000 [0.926, 1.000] (95% Wilson, n=48) |
| 2 sd | 48/48 | 1.000 [0.926, 1.000] (95% Wilson, n=48) |

## lerobot__svla_so100_pickplace

50 episodes with both commanded and measured streams; 30 used for the nominal spread, 20 held out.

False alarms on clean held-out episodes: 6/20 = 0.300 [0.145, 0.519] (95% Wilson, n=20).

| injected offset | fingerprint detects | rate |
|---|---|---|
| 0.25 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 0.5 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 1 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 2 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |

## lerobot__svla_so101_pickplace

50 episodes with both commanded and measured streams; 30 used for the nominal spread, 20 held out.

False alarms on clean held-out episodes: 1/20 = 0.050 [0.009, 0.236] (95% Wilson, n=20).

| injected offset | fingerprint detects | rate |
|---|---|---|
| 0.25 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 0.5 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 1 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |
| 2 sd | 20/20 | 1.000 [0.839, 1.000] (95% Wilson, n=20) |

## sixpigs1__so100_pick_cube_in_box

100 episodes with both commanded and measured streams; 60 used for the nominal spread, 40 held out.

False alarms on clean held-out episodes: 4/40 = 0.100 [0.040, 0.231] (95% Wilson, n=40).

| injected offset | fingerprint detects | rate |
|---|---|---|
| 0.25 sd | 40/40 | 1.000 [0.912, 1.000] (95% Wilson, n=40) |
| 0.5 sd | 40/40 | 1.000 [0.912, 1.000] (95% Wilson, n=40) |
| 1 sd | 40/40 | 1.000 [0.912, 1.000] (95% Wilson, n=40) |
| 2 sd | 40/40 | 1.000 [0.912, 1.000] (95% Wilson, n=40) |


FINGERPRINT_OFFSET_COMPLETE