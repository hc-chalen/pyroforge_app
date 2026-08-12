# PyroForge — data and model provenance

This document describes every dataset and trained model PyroForge depends on:
what each column is, where it came from, and what has and has not been verified.
It is the companion text for the Zenodo records, and it is deliberately explicit
about the limits of what can be attributed.

Source code lives in the repository. Data and weights are archived on Zenodo
because they are too large for git and because a DOI makes them citable.

| Record | Contents | Size |
|---|---|---|
| Data | `esm2_train_public.pt`, `esm2_test_public.pt`, `meltome_tm.npz` | ~3.0 GB |
| Models | `model_ddg.pt`, `model_thermo.pt`, `model_tm.joblib`, reports | ~11 MB |

> TODO: Fill in after publish zenodo
> Data: `10.5281/zenodo.XXXXXXXX` · Models: `10.5281/zenodo.XXXXXXXX`

---

## 1. Embedding datasets — `esm2_*_public.pt`

PyTorch pickles holding a dict. Load with:

```python
import torch
d = torch.load("esm2_train_public.pt", map_location="cpu", weights_only=False)
```

`esm2_train_public.pt` has 544,365 rows (499 parent proteins);
`esm2_test_public.pt` has 21,325 rows (54 parent proteins). The split is by
parent protein, so no protein appears in both.

### Features and row identity

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `embeddings` | N×1280 | float32 | ESM-2 650M, layer 33, mean-pooled over residues |
| `mutant_ids` | N | int64 | Unique per row |
| `parent_protein_ids` | N | int64 | Grouping key for the leakage-free split |
| `source_sequence_ids` | N | int64 | Wild-type sequence this row mutates |
| `target_sequence_ids` | N | int64 | The mutated sequence |
| `num_substitutions` | N | int16 | 0 marks a wild-type anchor row |
| `num_insertions` | N | int16 | 0 or 1 |
| `num_deletions` | N | int16 | 0 or 1 |
| `mutation_positions` | N×10 | int16 | Residue positions, `-1` padded |
| `mutation_source_aa` | N×10 | int8 | Index into `aa_vocab`, `-1` padded |
| `mutation_target_aa` | N×10 | int8 | Index into `aa_vocab`, `-1` padded |

`aa_vocab` is the string `ACDEFGHIKLMNPQRSTVWY`, so an amino-acid index `i` maps
to `aa_vocab[i]`. Up to 10 mutations are recorded per row.

**Sequences are not included.** The files carry embeddings and integer ids only.
This is the main limitation of the archive: rows cannot be mapped back to
sequences or to records in the upstream databases without the export tool
described in §4.

### Labels

`NaN` marks an absent measurement. Coverage differs sharply between the two
files because the parent split was not stratified by label.

| Key | Unit | train coverage | test coverage | Used by |
|---|---|---|---|---|
| `dg` | kcal/mol | 513,689 (94.4%) | 5,058 (23.7%) | `model_thermo.pt` |
| `ddg` | kcal/mol | 484,127 (88.9%) | 4,078 (19.1%) | `model_ddg.pt` |
| `tm` | °C | 3,331 (0.6%) | 1,027 (4.8%) | transfer diagnostic only |
| `trypsin_ml` | a.u. | 507,129 (93.2%) | 4,254 (19.9%) | — |
| `chymotrypsin_ml` | a.u. | 509,769 (93.6%) | 4,478 (21.0%) | — |

`ddg` is the primary optimization signal. Its sign convention is
**`ddg = dg(mutant) − dg(wild-type)`**, verified rather than assumed (§2).

`tm` in these files is *not* the Meltome data. It is a small, different assay
(folding Tm on designed domains) inherited from the upstream export. Absolute Tm
is trained from `meltome_tm.npz` instead; see §3 and §5.

`trypsin_ml` and `chymotrypsin_ml` are the raw proteolysis measurements from
which `dg` is derived. They are unused by any trainer but are retained because
their origin is established and they are the primary signal behind `dg`.

---

## 2. What was verified, and how

The upstream export script is not part of this project (§4), so the label columns
could not be matched row-by-row against the published source datasets — the
files carry no sequences to join on. Instead the columns were checked against
each other, on the principle that quantities from a single experiment obey known
relations. `backend/scripts/verify_data_provenance.py` reproduces all of this.

**`ddg` against `dg`.** For every mutant row with a wild-type anchor (same
`source_sequence_ids`, `num_substitutions == 0`), `dg(mutant) − dg(anchor)` was
compared to `ddg`:

| | train | test |
|---|---|---|
| Comparable rows | 434,128 | 2,520 |
| Pearson r | **0.9742** | **1.0000** |
| MAE | 0.133 kcal/mol | 0.043 kcal/mol |
| Within 0.5 kcal/mol | 95.3% | 100.0% |

The two columns are mutually consistent and share an origin, and the sign
convention is fixed by the fit rather than guessed.

**`dg` against the proteases.** Tsuboyama et al. 2023 derive folding stability
from trypsin and chymotrypsin proteolysis. Regressing `dg` on both columns:

| | train | test |
|---|---|---|
| r (`dg` ~ `trypsin_ml`) | 0.945 | 0.965 |
| r (`dg` ~ `chymotrypsin_ml`) | 0.901 | 0.959 |
| Joint R² | **0.906** | **0.952** |

`dg` is a near-linear function of the two proteolysis measurements, which is
consistent with the Tsuboyama method and identifies all three columns together.

**`ddg` antisymmetry.** Where both A→B and B→A exist at the same position of the
same parent (24 pairs in train), `r(forward, reverse) = −0.959` and 95.8% of
pairs have opposite signs — `ddg` behaves as a genuine thermodynamic difference.

**Physical ranges.** All 15 original columns fall inside the range their name
implies (ΔΔG within ±25 kcal/mol, Tm within 0–140 °C, Cm within 0–15 M, and so
on). No column showed a unit or scaling anomaly.

---

## 3. Absolute Tm dataset — `meltome_tm.npz`

The only dataset here whose provenance is fully traceable in code:
`backend/app/training/ingest_meltome.py` contains the download URL, the filtering
rules, and the embedding recipe.

```python
import numpy as np
M = np.load("meltome_tm.npz", allow_pickle=True)
```

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `emb` | 5106×1280 | float32 | Same ESM-2 recipe as §1 |
| `tm` | 5106 | float32 | Melting temperature, °C (31.3–99.0, mean 53.7) |
| `split` | 5106 | `<U5` | `train` (4,000) or `test` (1,106) |
| `source` | 5106 | `<U7` | `meltome` for all rows |

Built from the FLIP Meltome mixed split (Jarzab et al. 2020), filtered by length
and valid amino acids, de-duplicated by sequence, then embedded with the frozen
ESM2-650M mean-pooled recipe. Regenerate with:

```bash
cd backend && uv run pyroforge-ingest-meltome
```

Regeneration re-embeds ~5k proteins and takes GPU time, which is why the cache is
archived rather than left to be rebuilt.

---

## 4. Provenance limits — read this before reusing the data

**The script that produced `esm2_train.pt` and `esm2_test.pt` is not in this
repository.** It exported from an external database whose lineage began as
FireProtDB and had Tsuboyama 2023 and Human Domainome data merged into it. The
files record `scope: stability` and `available_in_scope: 4948558`, indicating a
~4.9M-row source of which these are subsets.

What this means concretely:

- `dg`, `ddg`, `trypsin_ml`, `chymotrypsin_ml` are attributed to Tsuboyama 2023
  on the evidence in §2 — a consistency argument, not a row-level join. The
  column scale, the ~500 parent proteins, and the proteolysis relation all agree
  with that source, but no id mapping was available to confirm it directly.
- `tm` is of FireProtDB lineage and is **not** attributed to a specific
  publication. Its 3,331 train values are the right order of magnitude for
  FireProtDB, but this was not verified. It is used only as a transfer
  diagnostic, never for training.
- Ten further columns present in the working files were **excluded from the
  archive**: `dtm`, `cm`, `m`, `dh`, `dhvh`, `dcp`, `domainome_ddg`,
  `domainome_ddg_std`, `domainome_fitness`, `domainome_fitness_std`. Every one
  has zero code references across the project and no established provenance.
  They remain only in the authors' working copy. `backend/scripts/export_public_dataset.py`
  performs the exclusion and records it inside each output file under
  `dropped_columns`.

Re-exporting from the original database was considered and rejected: it would
re-embed 544k sequences at significant GPU cost and would likely produce a
dataset that differs from the one the archived checkpoints were trained on,
invalidating the evaluation results below for no gain in attribution.

---

## 5. Trained models

Serving needs only these files plus the ESM-2 weights, which download
automatically. The embedding datasets are for training and evaluation.

| File | Trained on | Predicts |
|---|---|---|
| `model_ddg.pt` | `esm2_train.pt` → `ddg` | ΔΔG of a mutation (kcal/mol) |
| `model_thermo.pt` | `esm2_train.pt` → `dg` | Folding free energy (kcal/mol) |
| `improved/model_tm.joblib` | `meltome_tm.npz` → `tm` | Absolute melting temperature (°C) |

**`model_ddg.pt`** — MLP, `feature_spec: concat_mut_delta` (mutant embedding
concatenated with the wild-type difference, input dim 2560), widths
[256, 256, 256], dropout 0.324. Split: `parent_group_validation:parent_protein_ids`,
seed 42, 494,370 train rows / 49,995 val rows, 449 / 50 parents.
Held-out residual σ(ΔΔG) = 0.776 kcal/mol, which sets the served confidence
interval and the significance gate for mutation selection.

**`model_thermo.pt`** — MLP, widths [128, 1024, 128, 1024], dropout 0.415,
architecture from Hyperband NAS (`architecture_source: autopytorch_nas`). Same
parent-group split. Residual σ(dG) = 1.251 kcal/mol; calibration slope 0.996,
intercept −0.167.

**`model_tm.joblib`** — PCA-100 + Ridge+GBM on Meltome only, 4,000 training
proteins. Meltome held-out: **r = 0.810, MAE = 5.93 °C** (n=1,106), residual
σ(Tm) = 7.67 °C.

Transfer to the §1 `tm` assay is deliberately poor — r = 0.185 pooled, Spearman
0.007, MAE 14.9 °C (n=1,027). This is recorded, not hidden: whole-protein
lysate Tm and folding Tm on designed domains are different measurements, and
mixing them did not help. It is the reason absolute Tm is trained on Meltome
alone.

### Held-out scorecard

| Output | Parent-validation | Held-out test |
|---|---|---|
| ΔΔG (`model_ddg`) | r=0.758, MAE=0.596, n=39,264 | r=0.688, MAE=0.591, n=2,676 |
| absolute dG = dG(wt)+ΔΔG | r=0.656, MAE=1.003, n=42,473 | r=0.738, MAE=1.048, n=3,561 |
| absolute Tm (`model_tm`) | — | r=0.810, MAE=5.928, n=1,106 |

Within-protein ΔΔG correlation has median r=0.79 across the 22 validation
proteins. The held-out test file covers only 3 proteins, so its pooled numbers
are noisy — read the validation column for ΔΔG reliability.

---

## 6. Citations

Cite the archive and the original data sources together. The embeddings are a
derived work; the underlying measurements are not ours.

- **ESM-2** — Lin et al., *Evolutionary-scale prediction of atomic-level protein
  structure with a language model*, **Science 379:1123–1130 (2023)**.
  Model `facebook/esm2_t33_650M_UR50D`.
- **Tsuboyama 2023 mega-scale** — Tsuboyama et al., *Mega-scale experimental
  analysis of protein folding stability in biology and design*,
  **Nature 620:434–444 (2023)**, doi:10.1038/s41586-023-06328-6.
  Source of `dg`, `ddg`, `trypsin_ml`, `chymotrypsin_ml`.
- **Meltome Atlas** — Jarzab et al., *Meltome atlas — thermal proteome
  stability across the tree of life*, **Nature Methods 17:495–503 (2020)**,
  doi:10.1038/s41592-020-0801-4. Source of `meltome_tm.npz`, via the FLIP
  mixed split.
- **FireProtDB** — the lineage of the upstream export and the likely origin of
  `tm`; see §4 for why this is not stated as a confirmed attribution.

---

## 7. Reproducing

```bash
cd backend && uv sync

# verify the label columns are internally consistent (§2)
uv run python scripts/verify_data_provenance.py

# inspect any .pt file's schema and coverage
uv run python scripts/inspect_pt.py

# rebuild the archive copies from the working files (§4)
uv run python scripts/export_public_dataset.py

# retrain
uv run pyroforge-train           # model_thermo.pt   (needs esm2_train.pt)
uv run pyroforge-train-ddg       # model_ddg.pt      (needs esm2_train.pt)
uv run pyroforge-ingest-meltome  # rebuild meltome_tm.npz
uv run pyroforge-train-tm        # model_tm.joblib   (needs meltome_tm.npz)
```

Training requires `parent_protein_ids` for the leakage-free split. Point
`DATA_PATH` at `esm2_train.pt` (or `esm2_train_public.pt`); the retired
`esm2_50k.pt` subset lacks that column and is rejected with an explicit error.

`release/MANIFEST.json`, written by the export script, carries the SHA-256 of
each archived file for integrity checking against the Zenodo record.
