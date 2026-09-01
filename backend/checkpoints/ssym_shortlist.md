# Ssym deep-protein shortlist test

The only external proteins with ≥20 scored mutations — 4 of them. S669 cannot support this test (~7 mutations per protein), which is why it is run here.

*Intervals resample mutations WITHIN a protein; with 4 qualifying proteins a bootstrap over proteins would have no power. Per-protein values are the result.*

| protein | n | stabilizing | base rate | model_ddg @5 | enrichment | zero-shot @5 | random @5 |
|---|---|---|---|---|---|---|---|
| 1L63A | 118 | 20 | 0.169 | **0.400** [0.00–1.00] | 2.36× | 0.600 | 0.167 |
| 2LZMA | 66 | 18 | 0.273 | **0.800** [0.20–1.00] | 2.93× | 0.800 | 0.276 |
| 1LZ1A | 61 | 14 | 0.230 | **0.400** [0.00–0.80] | 1.74× | 0.400 | 0.228 |
| 1EY0A | 28 | 11 | 0.393 | **0.400** [0.00–1.00] | 1.02× | 0.200 | 0.392 |

## Aggregate over the deep proteins

| k | model_ddg | zero-shot | random | internal grouped-CV |
|---|---|---|---|---|
| 1 | 0.250 (range 0.00–1.00) | 1.000 | 0.270 | 0.733 |
| 3 | 0.417 (range 0.00–0.67) | 0.667 | 0.267 | 0.741 |
| 5 | 0.500 (range 0.40–0.80) | 0.500 | 0.266 | 0.736 |
| 10 | 0.550 (range 0.30–0.80) | 0.350 | 0.264 | 0.703 |

Mean base rate across these proteins: **0.266**.

