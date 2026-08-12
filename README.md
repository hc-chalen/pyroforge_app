# PyroForge

**PyroForge is a web app for improving the heat-stability of proteins.**

You paste in a protein sequence and it will:

- **Predict** its melting temperature (Tm) and folding stability,
- **Scan** every possible single-residue mutation and return a *ranked shortlist*
  of candidates predicted to make the protein more stable,
- **Fold & visualize** the 3-D structure with a stability heatmap.

It is built around **ESM-2**, a protein language model, with a FastAPI backend
and a React + TypeScript frontend.

---

## How it works (the core idea)

Predicting protein stability from scratch is hard. PyroForge takes a shortcut
that works surprisingly well:

1. **Turn the protein into numbers.** A large pre-trained protein language model,
   **ESM-2 (650M)**, reads the amino-acid sequence and produces a single
   1280-number vector (an "embedding") that captures what the protein "is like".
   ESM-2 is **frozen** — we never retrain it, we just use it as a feature
   extractor.

2. **Predict stability from the embedding.** Small, fast models sit on top of
   those embeddings and predict the numbers we actually care about.

3. **Find better mutations.** To score a mutation, we compare the embedding of
   the mutant to the embedding of the original (wild-type) protein. Doing this
   for every position and every amino acid gives a full "mutation map".

### The three prediction models

| What it predicts | Meaning | Trained on |
|------------------|---------|------------|
| **Tm** | Absolute melting temperature (°C) — how hot before it unfolds | Meltome Atlas, via `meltome_tm.npz` (5,106 proteins) |
| **dG** | Folding free energy — how stably it folds | `esm2_train.pt` → `dg` (Tsuboyama 2023) |
| **ΔΔG** | How much a *mutation* changes stability (kcal/mol) | `esm2_train.pt` → `ddg` (Tsuboyama 2023) |

The Tm model reads the Meltome cache **only** — it does not touch the
`esm2_*.pt` files, whose sparse `tm` column is a different assay kept purely as a
transfer diagnostic. Full column-by-column provenance, including what could and
could not be attributed, is in **[DATA.md](DATA.md)**.

**ΔΔG is the key signal for optimization** — it tells you whether a mutation
helps or hurts. One nice property: flipping a mutation around should flip the
sign of its effect (mutating A→B is the opposite of B→A). The ΔΔG model is
trained to respect this automatically.

> **How much to trust the shortlist.** Measured on parent-held-out proteins
> (`uv run pyroforge-calibrate-ddg`), the ΔΔG model *ranks* mutations well within
> a protein — median Spearman ≈ 0.76, and **74 % of a top-5 shortlist was truly
> stabilizing against a 22 % base rate** (95 % CI 61–85 %, n = 22 proteins). But
> that is an average: per protein it ranged from 0 % to 100 %, and fewer than half
> the shortlist was stabilizing for 5 of the 22. The model's *absolute* ΔΔG is
> also biased low. So treat the output as **candidates to screen**, not as
> verified stabilizing mutations, and order the wild type as a control.

> **How a sequence is scored:** a wild-type sequence shows its absolute **Tm**
> and **dG**. A mutant shows its absolute **Tm** plus the **ΔΔG** of the change.

---

## Architecture

```
┌──────────────┐     HTTP      ┌──────────────┐     jobs     ┌──────────────┐
│   Frontend   │  ──────────>  │   FastAPI    │  ─────────>  │ Celery worker│
│ React + Vite │  <──────────  │   backend    │  <─────────  │ (ESM-2, ML)  │
└──────────────┘               └──────────────┘   via Redis  └──────────────┘
```

- **Frontend** — the web UI (paste sequences, run predictions, view results).
- **Backend (FastAPI)** — the API. Fast predictions run here directly.
- **Celery worker** — runs the slow jobs (optimization scans, structure folding)
  in the background so the UI stays responsive.
- **Redis** — the message queue that connects the backend and the worker.

### Repository layout

```
pyroforge/
├── backend/                 # FastAPI app + Celery worker + ML training code
│   └── app/
│       ├── api/             # API routes & request/response schemas
│       ├── core/            # config, Celery setup
│       ├── data/            # dataset loading (leakage-free train/test split)
│       ├── engines/         # prediction, optimization, folding, registry
│       ├── models/          # the small PyTorch models
│       ├── services/        # ESM-2 service, model loading
│       ├── tasks/           # background (Celery) jobs
│       └── training/        # scripts that train the models
├── frontend/                # React + TypeScript + Vite web client
├── docker-compose.yml       # runs the whole stack with one command
├── docker-compose.train.yml # overlay that mounts the dataset for retraining
├── DATA.md                  # dataset & model provenance, Zenodo DOIs
├── esm2_train.pt            # training data (embeddings + labels, ~2.9 GB) — Zenodo
└── esm2_test.pt             # held-out test data (~110 MB) — Zenodo
```

The `esm2_*.pt` files are **training/evaluation data only** and are not in git.
To just run the app you don't need them — you only need the trained models in
`backend/checkpoints/` (shipped with the repo) plus the ESM-2 weights, which
download automatically the first time you run it. Download them from Zenodo only
if you want to retrain or re-evaluate; see [DATA.md](DATA.md).

---

## Running the app (Docker — recommended)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/).
From the repo root:

```bash
docker compose up --build
```

Wait until the containers are healthy, then open **http://localhost:5173**.

| Service   | URL                     | Role                          |
|-----------|-------------------------|-------------------------------|
| Frontend  | http://localhost:5173   | The web UI                    |
| API       | http://localhost:8000   | FastAPI backend               |
| Redis     | localhost:6379          | Job queue (starts on its own) |
| Worker    | (no UI)                 | Runs background jobs          |

Check the backend is ready:

```bash
curl http://localhost:8000/health
# {"status":"ok","models_ready":"true"}
```

> The first build takes a few minutes, and the **first run downloads the ESM-2
> weights (~2.5 GB)** — this is a one-time download that is then cached.

### Using the app

1. Open http://localhost:5173
2. Create an analysis and paste a protein sequence (10–400 residues).
3. **Run Prediction** — get the absolute Tm for the sequence.
4. **Run Optimization** — scan every single-residue mutation and get a ranked
   shortlist of candidates to screen, with the measured hit rate shown alongside
   (runs in the background; the page updates when it's done).
5. **Fold & Visualize** — see the predicted 3-D structure with a stability
   heatmap.
6. Optional: registry search, ordering quote, and embedding comparison.

---

## Deploying on a GPU server

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

The app is served on port `3002` and is **unauthenticated** — anyone who can
reach that port can queue GPU work, so keep it on a trusted network or put your
own access control in front of it.

See **[docs/OPERATIONS.md](docs/OPERATIONS.md)** for first-time setup, the
systemd unit, and what to check before exposing it publicly.

---

## Running locally (without Docker)

Useful for development with instant reload. You need Python 3.12 +
[uv](https://docs.astral.sh/uv/), Node.js 22+, and Redis. Open four terminals:

```bash
# 1 — Redis (the job queue)
docker compose up redis            # or: brew services start redis

# 2 — Backend API
cd backend && uv sync
uv run uvicorn app.main:app --reload --port 8000

# 3 — Background worker
cd backend && uv run celery -A app.core.celery_app worker --loglevel=info

# 4 — Frontend
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173.

---

## Training the models

The trained models ship with the repo (via Git LFS), so you normally don't need
to train anything. If you want to reproduce them, download the datasets from
Zenodo ([DATA.md](DATA.md)) into the repo root, then run from `backend/` after
`uv sync`:

```bash
# dG model (folding free energy)          — needs esm2_train.pt
uv run pyroforge-train

# ΔΔG model (mutation effects)            — needs esm2_train.pt
uv run pyroforge-train-ddg
# Characterise it: fits the level calibration and measures the within-protein
# ranking quality the optimizer reports. Re-run after every train-ddg, or the
# app has no measured hit rate to show.
uv run pyroforge-calibrate-ddg

# Absolute Tm model: download + embed the Meltome data, then fit
uv run pyroforge-ingest-meltome
uv run pyroforge-train-tm
```

Training automatically uses a GPU if one is available (NVIDIA CUDA or Apple
Silicon), otherwise the CPU.

To retrain inside Docker, add the training overlay so the dataset is mounted —
the base stack deliberately doesn't mount it, since serving never reads it:

```bash
docker compose -f docker-compose.yml -f docker-compose.train.yml up
```

### Verifying the data

The label columns' provenance was established by internal consistency checks
rather than by joining against the source databases (the files carry no
sequences). Both scripts are reproducible:

```bash
cd backend && uv run python scripts/verify_data_provenance.py
```

```bash
cd backend && uv run python scripts/inspect_pt.py
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Health check + model readiness |
| POST | `/api/v1/predict` | Predict absolute Tm (and optional ΔΔG) |
| POST | `/api/v1/optimize` | Start a mutation-scan optimization job |
| GET  | `/api/v1/jobs/{id}` | Check the status of a background job |
| POST | `/api/v1/visualize/fold` | Predict the 3-D structure (ESMFold) |
| POST | `/api/v1/visualize/heatmap` | Per-residue stability heatmap data |
| POST | `/api/v1/registry/search` | Search similar iGEM parts |
| POST | `/api/v1/procure/quote` | Codon-optimize + get an ordering quote |
| POST | `/api/v1/embed/compare` | Compare two sequences' embeddings |

---

## Data and large files

This repository holds **source code**. The embedding datasets and trained model
weights are archived on **Zenodo**, which gives each upload a citable DOI and is
the recommended long-term archive for iGEM teams.

| Record | Contents | DOI |
|--------|----------|-----|
| Data | `esm2_train_public.pt`, `esm2_test_public.pt`, `meltome_tm.npz` (~3.0 GB) | _pending_ |
| Models | `model_ddg.pt`, `model_thermo.pt`, `model_tm.joblib` + reports (~11 MB) | _pending_ |

**[DATA.md](DATA.md)** documents every column, what was verified about it, and —
importantly — what could not be attributed. Read it before reusing the data.

## Data sources & references

- **ESM-2** (protein language model) — Lin et al., *Evolutionary-scale
  prediction of atomic-level protein structure…*, **Science 379:1123–1130
  (2023)**; model `facebook/esm2_t33_650M_UR50D`.
- **Meltome Atlas** (melting temperatures) — Jarzab et al., **Nature Methods
  17:495–503 (2020)**, doi:10.1038/s41592-020-0801-4. Source of the absolute-Tm
  training data, via the FLIP mixed split.
- **Tsuboyama 2023 mega-scale** (folding stability & mutation effects) —
  Tsuboyama et al., **Nature 620:434–444 (2023)**,
  doi:10.1038/s41586-023-06328-6. Source of the `dg` and `ddg` labels.
- **FireProtDB** — the lineage of the upstream database export the `esm2_*.pt`
  files came from. The export script is not part of this project, so some
  columns could not be attributed to a specific publication; see
  [DATA.md §4](DATA.md).
