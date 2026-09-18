# Open VLM judge (open:Qwen/Qwen2.5-VL-7B-Instruct) on ur5fail execution test set

Episodes: 140 (140 decided, 0 abstained), robot time 1.17 h. Intervals are 95% Wilson.

| metric | value |
|---|---|
| abstain rate | 0.000 [0.000, 0.027] (n=140) |
| coverage | 1.000 [0.973, 1.000] (n=140) |
| accuracy on decided | 0.557 [0.474, 0.637] (n=140) |
| balanced accuracy on decided | 0.555 [0.442, 0.663] |
| success recall (TPR) | 0.435 [0.324, 0.552] (n=69) |
| failure recall (TNR) | 0.676 [0.561, 0.773] (n=71) |
| failure precision | 0.552 [0.447, 0.652] (n=87) |
| success bias (predicted minus true success rate, decided) | -0.114 |
| false alarms (failure called on a true success) | 39 |
| false alarm rate per true success | 0.565 [0.448, 0.676] (n=69) |
| false alarms per hour of robot time | 33.43 [26.49, 39.96] |

Balanced accuracy bounds are the mean of the two class-recall Wilson bounds, a rough guide, not a formal interval.
Reference: the best video-only judge on FailBench reaches 0.77 balanced accuracy overall and 0.52 on contact-rich tasks.


Parse or backend errors: 0/140. Truth from failure_mode; success tokens: ('success', 'no_failure', 'no failure', 'none', 'correct', 'ground_truth', 'ground truth').