# Guard false-alarm rate on a large held-out nominal pool

Policy lerobot/act_aloha_sim_transfer_cube_human live in gym_aloha/AlohaTransferCube-v0. Calibration pool 150 saved nominal episodes (from the 0.1.4 run). Held-out episodes run fresh here: 400, of which 338 succeeded and 62 failed on their own. A false alarm is an alert raised during an episode the policy actually completed.

False alarms: 0/338 = 0.000 [0.000, 0.011] (95% Wilson, n=338), against a 0.05 bound.

For comparison, 0.1.4 measured 0/18 = 0.000 [0.000, 0.176] (95% Wilson, n=18), an interval wide enough to hide a 17 percent true rate.

Total wall time 10397 s.

GUARD_FA_COMPLETE