# robotruth claims audit

robotruth 0.1.0, generated 2026-09-18 14:37 UTC

**Verdict: WARN**

## Reported rates with intervals

What each number actually says once the interval is attached.

| task | policy | trials | rate | interval width |
|---|---|---|---|---|
| so101_eval | 1cam@20000 | 50 | 0.20 [0.11, 0.33] | 0.218 |
| so101_eval | 2cam@50000 | 50 | 0.02 [0.00, 0.10] | 0.101 |
| so101_eval | side_claw@20000 | 50 | 0.30 [0.19, 0.44] | 0.246 |

## Pairwise comparisons [INFO]

3 pairwise comparisons within tasks; 2 resolved at 95% confidence, 1 are inside the noise.

| task | A | B | diff (B-A) | interval | Fisher p | resolved |
|---|---|---|---|---|---|---|
| so101_eval | 1cam@20000 | 2cam@50000 | -0.180 | [-0.31, -0.06] | 0.008 | yes |
| so101_eval | 1cam@20000 | side_claw@20000 | 0.100 | [-0.07, +0.26] | 0.356 | no |
| so101_eval | 2cam@50000 | side_claw@20000 | 0.280 | [+0.14, +0.42] | 0.000 | yes |
