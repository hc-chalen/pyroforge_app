# PyroForge

Protein-thermostability platform: paste a protein sequence, predict its melting
temperature and folding stability, scan every single-residue mutation for
stabilizing changes, and fold/visualize the result. FastAPI + Celery backend,
React + TypeScript frontend, ESM-2 protein language model at the core.

All predictors sit on top of **frozen, mean-pooled ESM2-650M embeddings**
(1280-d, one vector per sequence). Small models on top of those embeddings
predict the thermodynamic targets. Every accuracy number below is **held-out**
(leakage-free: whole parent proteins are held out, never individual mutants).

## Repository layout

```
pyroforge/
├── backend/                 # FastAPI app + Celery workers + training/eval code
│   ├── app/
│   │   ├── api/             # routes + schemas
│   │   ├── core/            # config, Celery app
│   │   ├── data/            # dataset loading + leakage-free parent split
│   │   ├── engines/         # prediction, optimization, folding, registry, procurement
│   │   ├── models/          # torch models + the Tm regressor
│   │   ├── services/        # ESM-2 service, model store, codon/Twist helpers
│   │   ├── tasks/           # Celery jobs
│   │   └── training/        # training + evaluation pipeline (see below)
│   ├── checkpoints/         # served weights (Git LFS) + reports/plots (see "Models & data")
│   ├── tests/               # fast, data-free unit tests
│   ├── scripts/             # one-off helpers (iGEM parts fetch)
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                # React + TypeScript + Vite client
├── docs/                    # design documents (PDF spec)
├── docker-compose.yml
├── esm2_train.pt            # full training embeddings + labels (~2.7 GB, local training)
├── esm2_50k.pt              # 50k-row training subset (~250 MB, used by Docker)
└── esm2_test.pt             # held-out test embeddings (~110 MB, evaluation only)
```

**The three `esm2_*.pt` files are training/evaluation data, not runtime
dependencies.** Serving (prediction + optimization) only needs the trained
checkpoints in `backend/checkpoints/` plus the ESM-2 weights (downloaded
automatically on first run). See [Hosting](#hosting-notes).

## Models & data

| Target | Served model | Training database | Held-out accuracy | Status |
|--------|--------------|-------------------|-------------------|--------|
| **ΔΔG** (mutation stability) | `model_ddg.pt` — MLP on difference embedding Δ = emb(mut) − emb(wt), **antisymmetric** | **Tsuboyama 2023 mega-scale** (~484k ΔΔG rows, 266 proteins; proteolysis ΔG) | grouped-CV pooled r≈0.66, within-protein median ρ≈0.75 | **reliable** |
| **dG** (folding free energy) | `model_thermo.pt` — dG-only ensemble of MLP towers | Tsuboyama 2023 mega-scale | r≈0.64, ρ≈0.82, MAE≈1.05 kcal/mol | ranks well; served for wild-type |
| **Absolute Tm** | `improved/model_tm.joblib` — PCA + SVR/stacking (AutoML-selected, `TmAutoMLRegressor`) | **Meltome Atlas** (FLIP `mixed_split`, thousands of proteins) | r≈0.83, MAE≈5.5 °C | **reliable** |

**Why the ΔΔG model is antisymmetric.** Physically, ΔΔG(A→B) = −ΔΔG(B→A). A
forward-only model violated this badly (bias −1.29 kcal/mol; corr(fwd, −rev) =
0.27). Training on every mutation *and its exact reverse* fixes it (bias ≈ 0;
corr ≈ 0.95) at no measurable accuracy cost on the 266-protein cross-validation.
This is the model now served, and `train_ddg.py` reproduces it — see
`antisymmetric_augment`.

**Retired (not served):** the `model_thermo` **Tm head** (curated
FireProtDB/ProThermDB-style Tm — only ~237 proteins, r≈0.39); absolute Tm now
comes from the Meltome-trained `model_tm`. The former direct-ΔTm model
(`model_dtm`, r≈0.16) and the **ΔTm-derived-from-ΔΔG** path have both been removed
— ΔTm added no information beyond ΔΔG (it was a strictly monotone function of it)
and had no held-out validity as a °C prediction. A mutant's stability change is
reported as **ΔΔG**; its absolute Tm comes from `model_tm` on the mutant sequence.

**Why two databases?** No single public dataset has both large-scale *mutational*
ΔΔG and broad *cross-protein* Tm, and the two are different assays (proteolysis
folding ΔG vs thermal melting). Each target therefore trains on the largest
appropriate database. Optimization ranks mutations on ΔΔG, which comes entirely
from the ~0.5M-row Tsuboyama mutant/wild-type measurements.

**How a sequence is scored at inference.** A **wild-type** shows absolute **Tm**
(`model_tm`) and **dG** (`model_thermo`). A **mutant** shows absolute **Tm**
(`model_tm` run on the mutant sequence) and **ΔΔG** (`model_ddg` on the difference
embedding). ΔΔG is the trustworthy mutation-effect signal — a single point
mutation barely moves the mean-pooled embedding, so read the mutant Tm as the
absolute Tm of that sequence, not as evidence of a thermal shift.

### Data sources & references

**Absolute Tm — Meltome Atlas, obtained via FLIP (downloaded by this repo).**
`pyroforge-ingest-meltome` fetches the FLIP benchmark's *meltome* mixed split from
`http://data.bioembeddings.com/public/FLIP/fasta/meltome/mixed_split.fasta`
(27,951 sequences; headers `TARGET=<Tm °C> SET=train|test`), filters to 20–300 aa,
de-duplicates, and ESM2-embeds 5,106 proteins. **FLIP** is a machine-learning
benchmark that *repackages* several experimental datasets into standardized
train/test splits; its *meltome* task is the **Meltome Atlas** data (melting
temperatures across 13 species by thermal proteome profiling). So FLIP is the
download format; Meltome Atlas is the underlying experiment.
- Meltome Atlas — Jarzab et al., *Meltome atlas—thermal proteome stability across the kingdoms of life*, **Nature Methods 17:495–503 (2020)**, doi:10.1038/s41592-020-0801-4.
- FLIP — Dallago et al., *FLIP: Benchmark tasks in fitness landscape inference for proteins*, **NeurIPS 2021 Datasets & Benchmarks**; data host `data.bioembeddings.com/public/FLIP/`.

**ΔΔG / dG — Tsuboyama mega-scale (NOT downloaded here; pre-existing `esm2_*.pt`).**
The `esm2_train.pt` / `esm2_test.pt` files were exported from an external
"stability-scope" database (metadata `scope=stability`, `available_in_scope≈4.95M`)
before this work; there is no download script for them in the repo. Their 484k
ΔΔG / 514k dG rows and fields `domainome_ddg` / `chymotrypsin_ml` / `trypsin_ml`
identify the source as the **Tsuboyama 2023 mega-scale** proteolysis dataset. The
original data is public on **Zenodo (doi:10.5281/zenodo.7992926)** and mirrored at
HuggingFace `RosettaCommons/MegaScale`, if you need to regenerate or cite it.
> ⚠️ The design docs label this "FireProtDB", but classic FireProtDB is ~16k
> curated entries with no proteolysis/domainome fields — the large ΔΔG/dG is
> Tsuboyama, not FireProtDB. Confirm the exact export query before citing for iGEM.
- Tsuboyama et al., *Mega-scale experimental analysis of protein folding stability in biology and design*, **Nature 620:434–444 (2023)**, doi:10.1038/s41586-023-06328-6.
- ESM-2 — Lin et al., *Evolutionary-scale prediction of atomic-level protein structure…*, **Science 379:1123–1130 (2023)**; model `facebook/esm2_t33_650M_UR50D`.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended), **or** local installs of Python 3.12 + [uv](https://docs.astral.sh/uv/), Node.js 22+, and Redis
- For Docker: `esm2_50k.pt` in the repo root (mounted as the training data). For
  full local training: `esm2_train.pt` + `esm2_test.pt` in the repo root.
- ~3 GB free disk for the ESM-2 weights (downloaded on first run).

## Quick start (Docker — recommended)

### 1. Train the models (first time only)

Serving requires these checkpoints in `backend/checkpoints/`:

| Checkpoint | Produced by | Required? |
|------------|-------------|-----------|
| `model_thermo.pt` | `pyroforge-train` | **yes** (API refuses to start without it) |
| `model_ddg.pt` | `pyroforge-train-ddg` | recommended (else ΔΔG falls back to dG(mut)−dG(wt)) |
| `improved/model_tm.joblib` | `pyroforge-ingest-meltome` → `pyroforge-train-tm` | recommended (else Tm uses the weak retired head) |

`model_thermo.pt` and `model_ddg.pt` ship with the repo via **Git LFS**, so a
clone with LFS enabled (`git lfs install`) already has them — only
`model_tm.joblib` needs producing. To retrain any of them, or if you cloned
without LFS, run the commands below from `backend/`:

```bash
cd backend
uv sync

# 1) model_thermo (served dG ensemble). Add --use-nas to build it from the
#    autoPyTorch NAS best config (see "Architecture search" below).
uv run pyroforge-train --epochs 30 --batch-size 256

# 2) model_ddg — the served ΔΔG model (antisymmetric). NOT produced by step 1.
uv run pyroforge-train-ddg

# 3) absolute Tm: download + ESM2-embed Meltome Atlas, then fit model_tm
uv run pyroforge-ingest-meltome --max-proteins 4000 --max-len 300
uv run pyroforge-train-tm
```

> Each `pyroforge-*` command has an equivalent `python -m app.training.<module>`
> form. Training and ESM-2 embedding use the GPU automatically (`device="auto"`
> → CUDA / Apple-Silicon MPS / CPU).

### 2. Start all services

From the repo root:

```bash
docker compose up --build
```

`docker compose up` runs the **dev stack**: `docker-compose.yml` (production base)
with `docker-compose.override.yml` auto-merged on top, giving hot reload
(`uvicorn --reload`), the Vite dev server, and source bind-mounts. See
[Production vs dev stacks](#production-vs-dev-stacks) to run the deploy stack or
use a GPU.

Wait until all four containers are healthy (the backend has a `/health`
healthcheck; the worker and frontend wait for it via `depends_on: service_healthy`):

| Service        | URL                      | Role                                      |
|----------------|--------------------------|-------------------------------------------|
| Frontend       | http://localhost:5173    | React UI                                  |
| API            | http://localhost:8000    | FastAPI backend                           |
| Redis          | localhost:6379           | Celery broker (started automatically)     |
| Celery worker  | (no UI)                  | Runs optimization, folding, registry jobs |

Verify the API is up and models are loaded:

```bash
curl http://localhost:8000/health
# {"status":"ok","models_ready":"true"}
```

If `models_ready` is `"false"`, `model_thermo.pt` is missing — complete step 1.

### 3. Use the app

1. Open http://localhost:5173
2. Create or select an analysis and paste a protein sequence (10–400 residues for folding)
3. **Run Optimization** — site-saturation ΔΔG scan (async; polls job status automatically)
4. **Run Prediction** — absolute Tm on the sequence; ΔΔG appears after optimization
5. **Fold & Visualize** — ESMFold structure + thermal heatmap (async)
6. Optional: Registry search, procurement quote, embedding comparison

### Docker build time

The first `docker compose up --build` (or any rebuild after backend code
changes) typically takes **1–3 minutes** (`uv sync`, a torch import sanity
check, and `npm install`). On the **first run** the backend downloads the ESM-2
weights (~2.5 GB) unless they are already in `~/.cache/torch/` — `docker-compose.yml`
mounts that cache automatically on macOS/Linux. Subsequent `docker compose up`
(without `--build`) reuses cached layers and starts quickly.

### Production vs dev stacks

The Compose setup is split so the same images run in both modes:

| Command | Stack | Backend | Frontend |
|---------|-------|---------|----------|
| `docker compose up` | **dev** (base + auto-merged `docker-compose.override.yml`) | `uvicorn --reload`, source bind-mounted | Vite dev server (HMR) on 5173 |
| `docker compose -f docker-compose.yml up` | **production** | `uvicorn` (no reload), baked image | static bundle served by **nginx** on 5173 |
| `DEVICE=cuda docker compose -f docker-compose.yml -f docker-compose.gpu.yml up` | **production + GPU** | same, ESM-2/optimization on CUDA | nginx |

- **Device.** `DEVICE` defaults to `cpu` (correct everywhere; **required** on Apple
  Silicon — ESM-2 is numerically degenerate on MPS). On an NVIDIA host, set
  `DEVICE=cuda` and add `docker-compose.gpu.yml`, which reserves the GPU (needs the
  NVIDIA Container Toolkit). Don't combine `docker-compose.gpu.yml` with the dev
  override.
- **Frontend.** The production image is a multi-stage build (`tsc && vite build` →
  nginx); `VITE_API_URL` is baked in at build time (default `http://localhost:8000`,
  override with `--build-arg`/the `VITE_API_URL` env). The dev override swaps in the
  Vite dev server, which reads `VITE_API_URL` at runtime.

## Training pipeline (reference)

Run from `backend/` after `uv sync`. Data paths default to `esm2_train.pt` /
`esm2_test.pt` in the repo root; Docker overrides the training file to
`esm2_50k.pt` via the `DATA_PATH` env var.

| Step | Command | Output | Trains on |
|------|---------|--------|-----------|
| dG head | `uv run pyroforge-train` | `model_thermo.pt`, `training_report.json` | Tsuboyama |
| ΔΔG | `uv run pyroforge-train-ddg` | `model_ddg.pt`, `ddg_training_report.json` | Tsuboyama (+ reverse augmentation) |
| Tm ingest | `uv run pyroforge-ingest-meltome` | `improved/meltome_tm.npz` | Meltome Atlas |
| Tm fit | `uv run pyroforge-train-tm` | `improved/model_tm.joblib`, `improved/tm_report.json` | Meltome Atlas |

Async retraining of `model_thermo` is also exposed via the API:

```bash
curl -X POST http://localhost:8000/api/v1/admin/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 30, "batch_size": 256, "lr": 0.001}'
```

## Evaluating model performance

`pyroforge-evaluate` scores every served model on the held-out test file using
the same anchored-baseline construction the app uses, and writes an anchored
scatter grid, a per-protein reliability panel, a summary bar chart, a metrics
JSON and a markdown scorecard to `backend/checkpoints/plots/`:

```bash
cd backend
uv run pyroforge-evaluate
```

Latest held-out results:

| Target | Eval set | Result |
|--------|----------|--------|
| ΔΔG | 266-protein grouped 5-fold CV (434,601 mutants) | pooled r≈0.66 [0.61, 0.70], within-protein median ρ≈0.75; antisymmetry corr 0.95 |
| dG | `esm2_test.pt` (5,058 rows) | r≈0.64, ρ≈0.82, MAE≈1.05 kcal/mol |
| Absolute Tm | 1,106 held-out Meltome proteins | r≈0.83, MAE≈5.5 °C (AutoML stack) |

Baselines from the ΔΔG cross-validation confirm the difference embedding is
load-bearing: a ridge baseline gets pooled r≈0.42 and a mutation-only MLP (no
Δemb) gets r≈0.37, versus r≈0.66 for the served design. Additional plots
(cross-validation, antisymmetry, Tm learning curve, scorecards) live in
`backend/checkpoints/improved/plots/`.

### Architecture search for the served dG model

`pyroforge-autopytorch` runs Hyperband neural architecture search (the same BOHB
multi-fidelity approach as [Auto-PyTorch Tabular](https://arxiv.org/abs/2006.13799))
over the dG tower, searching layer widths/depth, dropout, learning rate, weight
decay and batch size. `pyroforge-autopytorch-compare` renders PyTorch-vs-NAS
comparison plots, and its report also feeds the read-only "Model design" panel in
the UI (`GET /api/v1/models/nas-report`).

The search result can either stay a benchmark **or drive the served model**:
retrain `model_thermo` with `pyroforge-train --use-nas` to build the served dG
ensemble from the NAS best config. The search space (default widths
`128,256,512,768,1024`, 2–5 layers) is wide enough to reach and beat the
hand-picked `[1024,512,256,128]` tower.

```bash
# 1) search the dG architecture (cap rows so the search explores many configs)
uv run pyroforge-autopytorch --time-limit 1800 --max-train-rows 50000
# 2) (optional) benchmark plots: hand-designed tower vs NAS
uv run pyroforge-autopytorch-compare
# 3) promote the NAS architecture into the served model
uv run pyroforge-train --use-nas
```

> The official `autoPyTorch` PyPI package pins sklearn<0.25 and is incompatible
> with this backend's Python 3.12 environment, so the tuner reimplements the same
> multi-fidelity MLP search methodology.

## Local development (without Docker)

Use this for hot reload without rebuilding images. You need **four terminals**.

> **Do not mix Docker and local dev.** If `docker compose up` is running, stop it
> first (`docker compose down`); otherwise Docker holds ports 5173/8000, Vite
> falls back to 5174, and the API rejects that origin.

```bash
# Terminal 1 — Redis
docker compose up redis            # or: brew services start redis

# Terminal 2 — API
cd backend && uv sync
uv run uvicorn app.main:app --reload --port 8000

# Terminal 3 — Celery worker  (do NOT also run a Docker worker — they compete for jobs)
# The worker defaults to the non-forking "solo" pool (see note below); passing
# --pool=solo explicitly is optional but makes the intent obvious.
cd backend && uv run celery -A app.core.celery_app worker --loglevel=info --pool=solo

# Terminal 4 — Frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173. The frontend talks to the API at
`http://localhost:8000` (override with `VITE_API_URL`).

First-time backend extras:

```bash
cd backend
cp .env.example .env                          # optional: set TWIST_API_KEY for live quotes
uv run python scripts/fetch_igem_parts.py     # optional: seed the iGEM registry DB
```

## Tests

Fast, data-free unit tests (no `esm2_*.pt` needed) guard the leakage-free parent
split and WT anchoring:

```bash
cd backend
uv run python tests/test_dataset_split.py     # or: uv run pytest tests/ (if pytest installed)
```

## Hosting notes

- **Serving needs checkpoints, not data files.** A hosted API only requires
  `backend/checkpoints/` (populated by training) and network access to download
  the ESM-2 weights once. The `esm2_*.pt` files are for (re)training/evaluation.
- **GPU.** In-process, `device="auto"` selects CUDA, then Apple-Silicon MPS, then
  CPU (and ESM-2 is forced off MPS → CPU — see the device note above). On CPU,
  ESM-2 embedding is ~3 sequences/s; a GPU is strongly recommended for
  optimization (which embeds one mutant per position). **In Docker** the device is
  explicit: `DEVICE` defaults to `cpu`; run the production stack with
  `docker-compose.gpu.yml` and `DEVICE=cuda` to use an NVIDIA GPU (see
  [Production vs dev stacks](#production-vs-dev-stacks)).
- **Concurrency.** The Celery worker runs one job at a time (`--pool=solo
  --concurrency=1`), matching the single in-memory ESM-2 model. Scale by running
  more worker containers, not higher concurrency — each holds its own ~2.5 GB copy
  of the model.
- **ESM-2 weights** (~2.5 GB) download on first use to `~/.cache/torch/`; mount
  or bake that into the image to avoid re-downloading.
- **Ports:** 5173 (frontend), 8000 (API), 6379 (Redis). Set `VITE_API_URL` if the
  API is not on `localhost:8000`.
- **Health check:** `GET /health` returns `models_ready` — wire it to your
  orchestrator's readiness probe.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + model readiness |
| POST | `/api/v1/predict` | Predict absolute Tm (and optional ΔΔG) |
| POST | `/api/v1/optimize` | Async site-saturation optimization |
| GET | `/api/v1/jobs/{id}` | Poll async job status |
| POST | `/api/v1/visualize/fold` | Async ESMFold structure prediction |
| POST | `/api/v1/visualize/heatmap` | Per-residue heatmap data |
| POST | `/api/v1/registry/search` | Async MMseqs2 + iGEM parts search |
| POST | `/api/v1/procure/quote` | Codon optimization + Twist quote |
| POST | `/api/v1/embed/compare` | ESM-2 embedding similarity (original vs optimized) |
| GET | `/api/v1/models/nas-report` | Read-only architecture-search report |
| POST | `/api/v1/admin/train` | Async retraining of model_thermo |

## Troubleshooting

### Why are Prediction / Optimization disabled?

Both buttons stay greyed out until **all** of these are true:

1. **A sequence is entered** in the active analysis
2. **The API is reachable** at http://localhost:8000
3. **`models_ready` is `"true"`** — `model_thermo.pt` loaded successfully

| `curl http://localhost:8000/health` | Cause | Fix |
|---------------|-------|-----|
| Connection refused | API not running | Start backend (Docker or `uvicorn` locally) |
| `models_ready: "false"` | `model_thermo.pt` missing | Run training (step 1) |
| `models_ready: "true"` but buttons disabled | No sequence in the sidebar | Paste a sequence and select the analysis |

Fold & Visualize only needs a sequence (no trained models), so it can be enabled
while Prediction and Optimization are not. A red **"Backend unreachable"** banner
means the frontend cannot reach the API — fix the backend first.

### Celery worker crashes on macOS (`WorkerLostError: signal 6 (SIGABRT)`)

If Prediction/Optimization jobs fail with `Worker exited prematurely: signal 6
(SIGABRT)`, the worker was started with Celery's default **prefork** pool. Tasks
run PyTorch/ESM-2, which is **not fork-safe on macOS** — a forked child aborts as
soon as it touches torch. The backend now defaults the worker to the non-forking
**solo** pool (`worker_pool="solo"` in `app/core/celery_app.py`), so restarting
the worker fixes it. If you override the pool on the command line, use
`--pool=solo` (or `--pool=threads`) on macOS; avoid `--pool=prefork`.

### Common Docker issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `backend-1 exited with code 1` / `uvicorn: no such file or directory` | Host `backend/` volume overwrote the container virtualenv | Use the current `docker-compose.yml` (`uv run uvicorn` + `/app/.venv` volume) |
| Celery: `Connection refused` to `localhost:6379` | Redis not running | `docker compose up redis` or full `docker compose up` |
| `models_ready: "false"` inside Docker | Empty checkpoint volume | Ensure `backend/checkpoints/model_thermo.pt` exists on the host |

## Environment

Copy `backend/.env.example` to `backend/.env` and set `TWIST_API_KEY` for live
Twist quotes.
