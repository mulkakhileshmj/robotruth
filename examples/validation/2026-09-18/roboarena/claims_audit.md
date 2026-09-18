# robotruth claims audit

robotruth 0.1.0, generated 2026-09-18 14:37 UTC

**Verdict: WARN**

## Reported rates with intervals

What each number actually says once the interval is attached.

| task | policy | trials | rate | interval width |
|---|---|---|---|---|
| ALL | dam | 60 | 0.05 [0.02, 0.14] | 0.120 |
| ALL | dreaming_zebra | 58 | 0.26 [0.16, 0.38] | 0.220 |
| ALL | floppy_bagel_195k_temp1p0 | 8 | 0.12 [0.02, 0.47] | 0.448 |
| ALL | floppy_bagel_285k_temp1p0 | 12 | 0.17 [0.05, 0.45] | 0.401 |
| ALL | floppy_bagel_420k | 29 | 0.07 [0.02, 0.22] | 0.201 |
| ALL | floppy_bagel_525k_temp1p0 | 33 | 0.12 [0.05, 0.27] | 0.225 |
| ALL | floppy_bagel_90k_temp1p0 | 10 | 0.00 [0.00, 0.28] | 0.278 |
| ALL | paligemma_binning_droid | 629 | 0.00 [0.00, 0.01] | 0.011 |
| ALL | paligemma_diffusion_droid | 751 | 0.10 [0.08, 0.12] | 0.042 |
| ALL | paligemma_fast_droid | 951 | 0.12 [0.10, 0.14] | 0.041 |
| ALL | paligemma_fast_specialist_droid | 953 | 0.12 [0.10, 0.14] | 0.041 |
| ALL | paligemma_vq_droid | 762 | 0.11 [0.09, 0.13] | 0.045 |
| ALL | pi05_droid | 564 | 0.16 [0.13, 0.19] | 0.060 |
| ALL | pi0_droid | 1009 | 0.07 [0.06, 0.09] | 0.032 |
| ALL | pi0_fast_droid | 739 | 0.13 [0.10, 0.15] | 0.048 |

## Pairwise comparisons [INFO]

105 pairwise comparisons within tasks; 32 resolved at 95% confidence, 73 are inside the noise.

| task | A | B | diff (B-A) | interval | Fisher p | resolved |
|---|---|---|---|---|---|---|
| ALL | dam | dreaming_zebra | 0.209 | [+0.08, +0.34] | 0.002 | yes |
| ALL | dam | floppy_bagel_195k_temp1p0 | 0.075 | [-0.06, +0.42] | 0.401 | no |
| ALL | dam | floppy_bagel_285k_temp1p0 | 0.117 | [-0.03, +0.40] | 0.191 | no |
| ALL | dam | floppy_bagel_420k | 0.019 | [-0.08, +0.17] | 0.659 | no |
| ALL | dam | floppy_bagel_525k_temp1p0 | 0.071 | [-0.04, +0.23] | 0.240 | no |
| ALL | dam | floppy_bagel_90k_temp1p0 | -0.050 | [-0.14, +0.23] | 1.000 | no |
| ALL | dam | paligemma_binning_droid | -0.047 | [-0.13, -0.01] | 0.005 | yes |
| ALL | dam | paligemma_diffusion_droid | 0.047 | [-0.04, +0.09] | 0.354 | no |
| ALL | dam | paligemma_fast_droid | 0.071 | [-0.02, +0.11] | 0.143 | no |
| ALL | dam | paligemma_fast_specialist_droid | 0.065 | [-0.02, +0.10] | 0.140 | no |
| ALL | dam | paligemma_vq_droid | 0.060 | [-0.03, +0.10] | 0.190 | no |
| ALL | dam | pi05_droid | 0.108 | [+0.02, +0.15] | 0.021 | yes |
| ALL | dam | pi0_droid | 0.024 | [-0.06, +0.06] | 0.616 | no |
| ALL | dam | pi0_fast_droid | 0.076 | [-0.01, +0.12] | 0.098 | no |
| ALL | dreaming_zebra | floppy_bagel_195k_temp1p0 | -0.134 | [-0.30, +0.23] | 0.668 | no |
| ALL | dreaming_zebra | floppy_bagel_285k_temp1p0 | -0.092 | [-0.27, +0.21] | 0.717 | no |
| ALL | dreaming_zebra | floppy_bagel_420k | -0.190 | [-0.32, -0.01] | 0.045 | yes |
| ALL | dreaming_zebra | floppy_bagel_525k_temp1p0 | -0.137 | [-0.28, +0.04] | 0.180 | no |
| ALL | dreaming_zebra | floppy_bagel_90k_temp1p0 | -0.259 | [-0.38, +0.03] | 0.102 | no |
| ALL | dreaming_zebra | paligemma_binning_droid | -0.255 | [-0.38, -0.16] | 0.000 | yes |
| ALL | dreaming_zebra | paligemma_diffusion_droid | -0.161 | [-0.29, -0.06] | 0.001 | yes |
| ALL | dreaming_zebra | paligemma_fast_droid | -0.138 | [-0.26, -0.04] | 0.007 | yes |
| ALL | dreaming_zebra | paligemma_fast_specialist_droid | -0.143 | [-0.27, -0.05] | 0.003 | yes |
| ALL | dreaming_zebra | paligemma_vq_droid | -0.148 | [-0.28, -0.05] | 0.003 | yes |
| ALL | dreaming_zebra | pi05_droid | -0.101 | [-0.23, -0.00] | 0.063 | yes |
| ALL | dreaming_zebra | pi0_droid | -0.184 | [-0.31, -0.09] | 0.000 | yes |
| ALL | dreaming_zebra | pi0_fast_droid | -0.133 | [-0.26, -0.03] | 0.009 | yes |
| ALL | floppy_bagel_195k_temp1p0 | floppy_bagel_285k_temp1p0 | 0.042 | [-0.32, +0.34] | 1.000 | no |
| ALL | floppy_bagel_195k_temp1p0 | floppy_bagel_420k | -0.056 | [-0.41, +0.13] | 0.530 | no |
| ALL | floppy_bagel_195k_temp1p0 | floppy_bagel_525k_temp1p0 | -0.004 | [-0.36, +0.18] | 1.000 | no |
| ALL | floppy_bagel_195k_temp1p0 | floppy_bagel_90k_temp1p0 | -0.125 | [-0.47, +0.17] | 0.444 | no |
| ALL | floppy_bagel_195k_temp1p0 | paligemma_binning_droid | -0.122 | [-0.47, -0.02] | 0.037 | yes |
| ALL | floppy_bagel_195k_temp1p0 | paligemma_diffusion_droid | -0.028 | [-0.37, +0.08] | 0.562 | no |
| ALL | floppy_bagel_195k_temp1p0 | paligemma_fast_droid | -0.004 | [-0.35, +0.10] | 1.000 | no |
| ALL | floppy_bagel_195k_temp1p0 | paligemma_fast_specialist_droid | -0.010 | [-0.36, +0.10] | 1.000 | no |
| ALL | floppy_bagel_195k_temp1p0 | paligemma_vq_droid | -0.015 | [-0.36, +0.09] | 1.000 | no |
| ALL | floppy_bagel_195k_temp1p0 | pi05_droid | 0.033 | [-0.31, +0.14] | 1.000 | no |
| ALL | floppy_bagel_195k_temp1p0 | pi0_droid | -0.051 | [-0.40, +0.05] | 0.464 | no |
| ALL | floppy_bagel_195k_temp1p0 | pi0_fast_droid | 0.001 | [-0.35, +0.11] | 1.000 | no |
| ALL | floppy_bagel_285k_temp1p0 | floppy_bagel_420k | -0.098 | [-0.38, +0.09] | 0.567 | no |
| ALL | floppy_bagel_285k_temp1p0 | floppy_bagel_525k_temp1p0 | -0.045 | [-0.34, +0.15] | 0.650 | no |
| ALL | floppy_bagel_285k_temp1p0 | floppy_bagel_90k_temp1p0 | -0.167 | [-0.45, +0.14] | 0.480 | no |
| ALL | floppy_bagel_285k_temp1p0 | paligemma_binning_droid | -0.163 | [-0.44, -0.04] | 0.002 | yes |
| ALL | floppy_bagel_285k_temp1p0 | paligemma_diffusion_droid | -0.069 | [-0.35, +0.05] | 0.333 | no |
| ALL | floppy_bagel_285k_temp1p0 | paligemma_fast_droid | -0.046 | [-0.33, +0.08] | 0.648 | no |
| ALL | floppy_bagel_285k_temp1p0 | paligemma_fast_specialist_droid | -0.051 | [-0.33, +0.07] | 0.640 | no |
| ALL | floppy_bagel_285k_temp1p0 | paligemma_vq_droid | -0.056 | [-0.34, +0.07] | 0.633 | no |
| ALL | floppy_bagel_285k_temp1p0 | pi05_droid | -0.009 | [-0.29, +0.12] | 1.000 | no |
| ALL | floppy_bagel_285k_temp1p0 | pi0_droid | -0.092 | [-0.37, +0.03] | 0.227 | no |
| ALL | floppy_bagel_285k_temp1p0 | pi0_fast_droid | -0.041 | [-0.32, +0.08] | 0.656 | no |
| ALL | floppy_bagel_420k | floppy_bagel_525k_temp1p0 | 0.052 | [-0.12, +0.21] | 0.676 | no |
| ALL | floppy_bagel_420k | floppy_bagel_90k_temp1p0 | -0.069 | [-0.22, +0.21] | 1.000 | no |
| ALL | floppy_bagel_420k | paligemma_binning_droid | -0.066 | [-0.22, -0.02] | 0.011 | yes |
| ALL | floppy_bagel_420k | paligemma_diffusion_droid | 0.028 | [-0.12, +0.08] | 1.000 | no |
| ALL | floppy_bagel_420k | paligemma_fast_droid | 0.052 | [-0.10, +0.11] | 0.565 | no |
| ALL | floppy_bagel_420k | paligemma_fast_specialist_droid | 0.046 | [-0.11, +0.10] | 0.764 | no |
| ALL | floppy_bagel_420k | paligemma_vq_droid | 0.041 | [-0.11, +0.10] | 0.760 | no |
| ALL | floppy_bagel_420k | pi05_droid | 0.089 | [-0.06, +0.15] | 0.290 | no |
| ALL | floppy_bagel_420k | pi0_droid | 0.005 | [-0.15, +0.06] | 1.000 | no |
| ALL | floppy_bagel_420k | pi0_fast_droid | 0.057 | [-0.10, +0.11] | 0.565 | no |
| ALL | floppy_bagel_525k_temp1p0 | floppy_bagel_90k_temp1p0 | -0.121 | [-0.27, +0.17] | 0.558 | no |
| ALL | floppy_bagel_525k_temp1p0 | paligemma_binning_droid | -0.118 | [-0.27, -0.04] | 0.000 | yes |
| ALL | floppy_bagel_525k_temp1p0 | paligemma_diffusion_droid | -0.024 | [-0.18, +0.05] | 0.556 | no |
| ALL | floppy_bagel_525k_temp1p0 | paligemma_fast_droid | -0.000 | [-0.15, +0.08] | 1.000 | no |
| ALL | floppy_bagel_525k_temp1p0 | paligemma_fast_specialist_droid | -0.006 | [-0.16, +0.07] | 0.786 | no |
| ALL | floppy_bagel_525k_temp1p0 | paligemma_vq_droid | -0.011 | [-0.16, +0.07] | 0.777 | no |
| ALL | floppy_bagel_525k_temp1p0 | pi05_droid | 0.037 | [-0.12, +0.12] | 0.805 | no |
| ALL | floppy_bagel_525k_temp1p0 | pi0_droid | -0.047 | [-0.20, +0.03] | 0.307 | no |
| ALL | floppy_bagel_525k_temp1p0 | pi0_fast_droid | 0.005 | [-0.15, +0.08] | 1.000 | no |
| ALL | floppy_bagel_90k_temp1p0 | paligemma_binning_droid | 0.003 | [-0.27, +0.01] | 1.000 | no |
| ALL | floppy_bagel_90k_temp1p0 | paligemma_diffusion_droid | 0.097 | [-0.18, +0.12] | 0.610 | no |
| ALL | floppy_bagel_90k_temp1p0 | paligemma_fast_droid | 0.121 | [-0.16, +0.14] | 0.618 | no |
| ALL | floppy_bagel_90k_temp1p0 | paligemma_fast_specialist_droid | 0.115 | [-0.16, +0.14] | 0.615 | no |
| ALL | floppy_bagel_90k_temp1p0 | paligemma_vq_droid | 0.110 | [-0.17, +0.13] | 0.612 | no |
| ALL | floppy_bagel_90k_temp1p0 | pi05_droid | 0.158 | [-0.12, +0.19] | 0.374 | no |
| ALL | floppy_bagel_90k_temp1p0 | pi0_droid | 0.074 | [-0.20, +0.09] | 1.000 | no |
| ALL | floppy_bagel_90k_temp1p0 | pi0_fast_droid | 0.126 | [-0.15, +0.15] | 0.622 | no |
| ALL | paligemma_binning_droid | paligemma_diffusion_droid | 0.094 | [+0.07, +0.12] | 0.000 | yes |
| ALL | paligemma_binning_droid | paligemma_fast_droid | 0.118 | [+0.10, +0.14] | 0.000 | yes |
| ALL | paligemma_binning_droid | paligemma_fast_specialist_droid | 0.112 | [+0.09, +0.13] | 0.000 | yes |
| ALL | paligemma_binning_droid | paligemma_vq_droid | 0.107 | [+0.09, +0.13] | 0.000 | yes |
| ALL | paligemma_binning_droid | pi05_droid | 0.155 | [+0.13, +0.19] | 0.000 | yes |
| ALL | paligemma_binning_droid | pi0_droid | 0.071 | [+0.05, +0.09] | 0.000 | yes |
| ALL | paligemma_binning_droid | pi0_fast_droid | 0.123 | [+0.10, +0.15] | 0.000 | yes |
| ALL | paligemma_diffusion_droid | paligemma_fast_droid | 0.024 | [-0.01, +0.05] | 0.139 | no |
| ALL | paligemma_diffusion_droid | paligemma_fast_specialist_droid | 0.018 | [-0.01, +0.05] | 0.238 | no |
| ALL | paligemma_diffusion_droid | paligemma_vq_droid | 0.013 | [-0.02, +0.04] | 0.448 | no |
| ALL | paligemma_diffusion_droid | pi05_droid | 0.061 | [+0.02, +0.10] | 0.001 | yes |
| ALL | paligemma_diffusion_droid | pi0_droid | -0.023 | [-0.05, +0.00] | 0.099 | no |
| ALL | paligemma_diffusion_droid | pi0_fast_droid | 0.029 | [-0.00, +0.06] | 0.084 | no |
| ALL | paligemma_fast_droid | paligemma_fast_specialist_droid | -0.006 | [-0.03, +0.02] | 0.723 | no |
| ALL | paligemma_fast_droid | paligemma_vq_droid | -0.011 | [-0.04, +0.02] | 0.544 | no |
| ALL | paligemma_fast_droid | pi05_droid | 0.037 | [+0.00, +0.07] | 0.043 | yes |
| ALL | paligemma_fast_droid | pi0_droid | -0.047 | [-0.07, -0.02] | 0.001 | yes |
| ALL | paligemma_fast_droid | pi0_fast_droid | 0.005 | [-0.03, +0.04] | 0.766 | no |
| ALL | paligemma_fast_specialist_droid | paligemma_vq_droid | -0.005 | [-0.04, +0.03] | 0.759 | no |
| ALL | paligemma_fast_specialist_droid | pi05_droid | 0.042 | [+0.01, +0.08] | 0.022 | yes |
| ALL | paligemma_fast_specialist_droid | pi0_droid | -0.041 | [-0.07, -0.02] | 0.002 | yes |
| ALL | paligemma_fast_specialist_droid | pi0_fast_droid | 0.010 | [-0.02, +0.04] | 0.546 | no |
| ALL | paligemma_vq_droid | pi05_droid | 0.048 | [+0.01, +0.09] | 0.013 | yes |
| ALL | paligemma_vq_droid | pi0_droid | -0.036 | [-0.06, -0.01] | 0.009 | yes |
| ALL | paligemma_vq_droid | pi0_fast_droid | 0.016 | [-0.02, +0.05] | 0.379 | no |
| ALL | pi05_droid | pi0_droid | -0.083 | [-0.12, -0.05] | 0.000 | yes |
| ALL | pi05_droid | pi0_fast_droid | -0.032 | [-0.07, +0.01] | 0.107 | no |
| ALL | pi0_droid | pi0_fast_droid | 0.052 | [+0.02, +0.08] | 0.000 | yes |
