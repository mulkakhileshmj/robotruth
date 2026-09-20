# Policy CI: act_v18_s1 vs act_v18

Battery `515e96ff75282363` (200 scenarios), task `aloha_transfer_cube`, backend `gym_aloha.transfer_cube.v0`, evaluator `0.1.0`.

| | act_v18 | act_v18_s1 |
|---|---|---|
| Success | 85.0% [79.4, 89.3] | 85.0% [79.4, 89.3] |

Paired difference (B - A): 0.0% [0.0, 0.0]

**Verdict: INCONCLUSIVE (more trials needed)** (anytime-valid, n=200, diff +0.000 [-0.074, +0.074])

## Scenario diff

- Newly broken: 0 observed | noise floor NOT MEASURED: run the same policy twice first
- Fixed: 0

---
Pins hash A `59cdd82cfd86dc3a` B `59cdd82cfd86dc3a`. Run manifests `9a3294cbb211339e` / `40de1646011b9265`. A number without an interval is not a result.
