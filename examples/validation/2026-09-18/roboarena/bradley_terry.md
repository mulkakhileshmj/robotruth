# RoboArena Bradley-Terry ranking

3284 sessions, 3284 usable pairwise outcomes, 0 skipped (unparsed layout).

| rank | policy | log-strength | bootstrap se | trials | binary success | mean partial |
|---|---|---|---|---|---|---|
| 1 | dreaming_zebra | +1.007 | 0.374 | 58 | 15/58 | 0.61 |
| 2 | pi05_droid | +0.515 | 0.245 | 564 | 89/564 | 0.48 |
| 3 | pi0_fast_droid | +0.409 | 0.253 | 739 | 93/739 | 0.45 |
| 4 | floppy_bagel_195k_temp1p0 | +0.347 | 1.890 | 8 | 1/8 | 0.48 |
| 5 | floppy_bagel_525k_temp1p0 | +0.245 | 0.389 | 33 | 4/33 | 0.38 |
| 6 | paligemma_vq_droid | +0.220 | 0.240 | 762 | 84/762 | 0.42 |
| 7 | paligemma_diffusion_droid | +0.210 | 0.251 | 751 | 73/751 | 0.41 |
| 8 | paligemma_fast_specialist_droid | +0.162 | 0.252 | 953 | 110/953 | 0.43 |
| 9 | paligemma_fast_droid | +0.140 | 0.247 | 951 | 115/951 | 0.44 |
| 10 | dam | -0.009 | 0.308 | 60 | 3/60 | 0.34 |
| 11 | floppy_bagel_420k | -0.048 | 0.423 | 29 | 2/29 | 0.33 |
| 12 | pi0_droid | -0.121 | 0.244 | 1009 | 75/1009 | 0.36 |
| 13 | floppy_bagel_285k_temp1p0 | -0.123 | 0.577 | 12 | 2/12 | 0.44 |
| 14 | floppy_bagel_90k_temp1p0 | -0.195 | 2.275 | 10 | 0/10 | 0.29 |
| 15 | paligemma_binning_droid | -2.759 | 0.274 | 629 | 2/629 | 0.03 |

Pairwise win probability P(row beats column):

| | dam | dreaming_zebra | floppy_bagel_195k_temp1p0 | floppy_bagel_285k_temp1p0 | floppy_bagel_420k | floppy_bagel_525k_temp1p0 | floppy_bagel_90k_temp1p0 | paligemma_binning_droid | paligemma_diffusion_droid | paligemma_fast_droid | paligemma_fast_specialist_droid | paligemma_vq_droid | pi05_droid | pi0_droid | pi0_fast_droid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dam | 0.50 | 0.27 | 0.41 | 0.53 | 0.51 | 0.44 | 0.55 | 0.94 | 0.45 | 0.46 | 0.46 | 0.44 | 0.37 | 0.53 | 0.40 |
| dreaming_zebra | 0.73 | 0.50 | 0.66 | 0.76 | 0.74 | 0.68 | 0.77 | 0.98 | 0.69 | 0.70 | 0.70 | 0.69 | 0.62 | 0.76 | 0.65 |
| floppy_bagel_195k_temp1p0 | 0.59 | 0.34 | 0.50 | 0.62 | 0.60 | 0.53 | 0.63 | 0.96 | 0.53 | 0.55 | 0.55 | 0.53 | 0.46 | 0.61 | 0.48 |
| floppy_bagel_285k_temp1p0 | 0.47 | 0.24 | 0.38 | 0.50 | 0.48 | 0.41 | 0.52 | 0.93 | 0.42 | 0.43 | 0.43 | 0.42 | 0.35 | 0.50 | 0.37 |
| floppy_bagel_420k | 0.49 | 0.26 | 0.40 | 0.52 | 0.50 | 0.43 | 0.54 | 0.94 | 0.44 | 0.45 | 0.45 | 0.43 | 0.36 | 0.52 | 0.39 |
| floppy_bagel_525k_temp1p0 | 0.56 | 0.32 | 0.47 | 0.59 | 0.57 | 0.50 | 0.61 | 0.95 | 0.51 | 0.53 | 0.52 | 0.51 | 0.43 | 0.59 | 0.46 |
| floppy_bagel_90k_temp1p0 | 0.45 | 0.23 | 0.37 | 0.48 | 0.46 | 0.39 | 0.50 | 0.93 | 0.40 | 0.42 | 0.41 | 0.40 | 0.33 | 0.48 | 0.35 |
| paligemma_binning_droid | 0.06 | 0.02 | 0.04 | 0.07 | 0.06 | 0.05 | 0.07 | 0.50 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.07 | 0.04 |
| paligemma_diffusion_droid | 0.55 | 0.31 | 0.47 | 0.58 | 0.56 | 0.49 | 0.60 | 0.95 | 0.50 | 0.52 | 0.51 | 0.50 | 0.42 | 0.58 | 0.45 |
| paligemma_fast_droid | 0.54 | 0.30 | 0.45 | 0.57 | 0.55 | 0.47 | 0.58 | 0.95 | 0.48 | 0.50 | 0.49 | 0.48 | 0.41 | 0.56 | 0.43 |
| paligemma_fast_specialist_droid | 0.54 | 0.30 | 0.45 | 0.57 | 0.55 | 0.48 | 0.59 | 0.95 | 0.49 | 0.51 | 0.50 | 0.49 | 0.41 | 0.57 | 0.44 |
| paligemma_vq_droid | 0.56 | 0.31 | 0.47 | 0.58 | 0.57 | 0.49 | 0.60 | 0.95 | 0.50 | 0.52 | 0.51 | 0.50 | 0.43 | 0.58 | 0.45 |
| pi05_droid | 0.63 | 0.38 | 0.54 | 0.65 | 0.64 | 0.57 | 0.67 | 0.96 | 0.58 | 0.59 | 0.59 | 0.57 | 0.50 | 0.65 | 0.53 |
| pi0_droid | 0.47 | 0.24 | 0.39 | 0.50 | 0.48 | 0.41 | 0.52 | 0.93 | 0.42 | 0.44 | 0.43 | 0.42 | 0.35 | 0.50 | 0.37 |
| pi0_fast_droid | 0.60 | 0.35 | 0.52 | 0.63 | 0.61 | 0.54 | 0.65 | 0.96 | 0.55 | 0.57 | 0.56 | 0.55 | 0.47 | 0.63 | 0.50 |