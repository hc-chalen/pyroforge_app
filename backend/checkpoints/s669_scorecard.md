# S669 external benchmark — PyroForge ΔΔG

Dataset: **S669** (Pancotti et al., Briefings in Bioinformatics 23(2):bbab555 (2022)) — 669 variants on 94 proteins.
Regime split: 612 at ≤400 aa, 57 above.
Code 77a096e on mps.

Two correlations are reported and they disagree. **Pooled ρ** mixes
between-protein level differences with within-protein ranking; **within-protein
ρ** is the median over proteins of the ranking inside each one. The optimizer
only ever ranks mutations within a single protein, so within-protein ρ is the
product-relevant number — and `model_ddg`'s difference-embedding feature
deliberately cancels the per-protein offset that pooled ρ rewards.

| Method | class | pooled ρ | **within-protein ρ** | top-5 prec | antisym r | stab. AUC |
|---|---|---|---|---|---|---|
| MAESTRO | structure | 0.463 | **0.456** | 0.378 | -0.221 | 0.685 |
| ACDC-NN | structure | 0.454 | **0.429** | 0.356 | -0.980 | 0.711 |
| INPS-Seq | seq-only | 0.440 | **0.406** | 0.415 | -0.996 | 0.716 |
| INPS3D | structure | 0.439 | **0.543** | 0.385 | -0.498 | 0.679 |
| ACDC-NN-Seq | seq-only | 0.431 | **0.378** | 0.363 | -1.000 | 0.707 |
| DDGun3D | structure | 0.427 | **0.402** | 0.385 | -0.967 | 0.707 |
| DDGun | seq-only | 0.427 | **0.366** | 0.392 | -0.959 | 0.718 |
| PremPS | structure | 0.423 | **0.464** | 0.393 | -0.846 | 0.642 |
| DUET | structure | 0.416 | **0.284** | 0.348 | -0.118 | 0.665 |
| PopMusic | structure | 0.415 | **0.383** | 0.407 | -0.318 | 0.673 |
| SDM | structure | 0.392 | **0.371** | 0.378 | -0.403 | 0.669 |
| Dynamut | structure | 0.374 | **0.208** | 0.333 | -0.584 | 0.672 |
| ThermoNet | structure | 0.373 | **0.257** | 0.356 | -0.846 | 0.678 |
| mCSM | structure | 0.363 | **0.357** | 0.370 | -0.052 | 0.641 |
| I-Mutant3.0 | structure | 0.358 | **0.400** | 0.400 | -0.060 | 0.627 |
| SAAFEC-Seq | seq-only | 0.353 | **0.400** | 0.393 | -0.034 | 0.628 |
| I-Mutant3.0-Seq | seq-only | 0.349 | **0.429** | 0.363 | -0.476 | 0.656 |
| model_ddg | **ours** | 0.310 | **0.484** | 0.370 | -0.778 | 0.648 |
| FoldX | structure | 0.287 | **0.229** | 0.356 | -0.199 | 0.620 |
| esm2_zeroshot | baseline | 0.266 | **0.236** | 0.370 | -1.000 | 0.593 |
| MuPro | seq-only | 0.255 | **0.297** | 0.304 | -0.319 | 0.606 |
| random | baseline | 0.057 | **-0.068** | 0.274 | -1.000 | 0.527 |

## Regime split (model_ddg)

- ≤400 aa (in regime): ρ = 0.330 (n=612)
- >400 aa (out of regime): ρ = -0.164 (n=57)

## Paired comparisons (bootstrap over proteins)

- `pooled__model_ddg_vs_esm2_zeroshot`: Δρ = +0.044, 95% CI [-0.225, +0.278], sign-flip rate 0.385
- `within_protein__model_ddg_vs_esm2_zeroshot`: Δρ = +0.248, 95% CI [+0.051, +0.502], sign-flip rate 0.011
- `pooled__model_ddg_vs_random`: Δρ = +0.253, 95% CI [+0.079, +0.413], sign-flip rate 0.003
- `within_protein__model_ddg_vs_random`: Δρ = +0.552, 95% CI [+0.263, +0.755], sign-flip rate 0.000
- `pooled__model_ddg_vs_DDGun`: Δρ = -0.118, 95% CI [-0.345, +0.092], sign-flip rate 0.174
- `within_protein__model_ddg_vs_DDGun`: Δρ = +0.138, 95% CI [-0.089, +0.260], sign-flip rate 0.092
- `pooled__model_ddg_vs_INPS-Seq`: Δρ = -0.130, 95% CI [-0.366, +0.048], sign-flip rate 0.118
- `within_protein__model_ddg_vs_INPS-Seq`: Δρ = +0.078, 95% CI [-0.121, +0.246], sign-flip rate 0.154
- `pooled__model_ddg_vs_ACDC-NN-Seq`: Δρ = -0.121, 95% CI [-0.346, +0.074], sign-flip rate 0.154
- `within_protein__model_ddg_vs_ACDC-NN-Seq`: Δρ = +0.106, 95% CI [-0.126, +0.302], sign-flip rate 0.132
