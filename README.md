# PyroForge

**PyroForge is a web app for improving the heat-stability of proteins.**

You paste in a protein sequence and it will:

- **Predict** its melting temperature (Tm) and folding stability,
- **Scan** every possible single-residue mutation to find changes that make the
  protein *more* stable,
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
| **Tm** | Absolute melting temperature (°C) — how hot before it unfolds | Meltome Atlas (thousands of proteins) |
| **dG** | Folding free energy — how stably it folds | Tsuboyama 2023 mega-scale dataset |
| **ΔΔG** | How much a *mutation* changes stability (kcal/mol) | Tsuboyama 2023 mega-scale dataset |

**ΔΔG is the key signal for optimization** — it tells you whether a mutation
helps or hurts. One nice property: flipping a mutation around should flip the
sign of its effect (mutating A→B is the opposite of B→A). The ΔΔG model is
trained to respect this automatically.

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
├── esm2_train.pt            # training data (embeddings + labels, ~2.7 GB)
├── esm2_50k.pt              # smaller training subset (~250 MB, used by Docker)
└── esm2_test.pt             # held-out test data (~110 MB, for evaluation)
```

The `esm2_*.pt` files are **training/evaluation data only**. To just run the app
you don't need them — you only need the trained models in `backend/checkpoints/`
(shipped with the repo) plus the ESM-2 weights, which download automatically the
first time you run it.

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
4. **Run Optimization** — scan every single-residue mutation for stabilizing
   changes (runs in the background; the page updates when it's done).
5. **Fold & Visualize** — see the predicted 3-D structure with a stability
   heatmap.
6. Optional: registry search, ordering quote, and embedding comparison.

---

## Trial deployment: login gate & usage monitoring

The shared GPU-server deployment currently sits behind a **temporary login
page**, with per-account usage monitoring alongside it. Both live entirely in
`auth/` and `monitoring/` plus two compose overlays — no auth code exists in
`backend/app/**` or `frontend/src/**`, and removing the gate later is a
delete-two-directories operation.

See **[docs/OPERATIONS.md](docs/OPERATIONS.md)** for setup, managing accounts,
reading the usage reports, and the teardown checklist.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
               -f docker-compose.auth.yml -f docker-compose.monitor.yml up -d
```

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
to train anything. If you want to reproduce them, run from `backend/` after
`uv sync`:

```bash
# dG model (folding free energy)
uv run pyroforge-train

# ΔΔG model (mutation effects — the main optimization signal)
uv run pyroforge-train-ddg

# Absolute Tm model: download + embed the Meltome data, then fit
uv run pyroforge-ingest-meltome
uv run pyroforge-train-tm
```

Training automatically uses a GPU if one is available (NVIDIA CUDA or Apple
Silicon), otherwise the CPU.

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

## Data sources & references

- **ESM-2** (protein language model) — Lin et al., *Evolutionary-scale
  prediction of atomic-level protein structure…*, **Science 379:1123–1130
  (2023)**; model `facebook/esm2_t33_650M_UR50D`.
- **Meltome Atlas** (melting temperatures) — Jarzab et al., **Nature Methods
  17:495–503 (2020)**, doi:10.1038/s41592-020-0801-4.
- **Tsuboyama 2023 mega-scale** (folding stability & mutation effects) —
  Tsuboyama et al., **Nature 620:434–444 (2023)**,
  doi:10.1038/s41586-023-06328-6.
