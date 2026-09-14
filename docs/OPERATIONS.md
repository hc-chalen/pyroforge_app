# PyroForge — deployment operations

Running the Docker Compose stack on a GPU server. The app is open and
unauthenticated: whoever can reach the port can use it.

## Quick reference

| What you want | Command | Open |
|---|---|---|
| Dev, hot reload | `docker compose up` | `:5173` |
| Production, locally | `docker compose -f docker-compose.yml up` | `:3002` |
| The real server | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` | `:3002` |

`docker-compose.override.yml` (the dev stack) is auto-merged by plain
`docker compose up`, which is why the first and second rows differ.

---

## How it fits together

```
              ┌──────────────────────────────────────────────┐
  browser ──► │ :3002  frontend (nginx, SPA + /api/ proxy)   │  ← only public port
              └───────────────────┬──────────────────────────┘
                                  │ /api/
                                  ▼
                                        ┌─► worker-fast   (GPU 0) predict, registry
                    :8000 backend ──► redis ─► worker-heavy  (GPU 0) optimize, train
                                        └─► worker-io           fold ──► esmfold (GPU 1)
                                                                    └─► api.esmatlas.com (fallback)
```

`:3002` is the *only* published port in the production stack. The backend and
Redis are not published on the host at all — they are reachable only from other
containers on the compose network. (The dev stack additionally publishes `:8000`
and `:6379` on loopback; see `docker-compose.override.yml`.)

**Analyses are stored in the browser, not on the server.** The frontend keeps
them in `localStorage` under `pyroforge-analyses`, so there is no per-user
server state to manage or back up. Two people sharing the same browser profile
on the same machine share that storage.

---

## First-time setup on the server

```bash
cd ~/pyroforge_app

# 1. Environment
cp .env.example .env        # set DEVICE=cuda on the GPU host

# 2. ESMFold weights, once (~10 GB into ~/.cache/huggingface). The esmfold
#    container mounts them and never downloads — see "Self-hosted ESMFold" below.

# 3. Bring the stack up (first boot downloads ESM-2 weights, ~2.5 GB)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# 4. Build the iGEM parts snapshot the Registry search runs against. Needs the
#    Registry SQL dump copied in first — see "The iGEM parts snapshot" below.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  cp sql_parts.sql.gz backend:/tmp/sql_parts.sql.gz
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  exec backend uv run pyroforge-fetch-parts --from-sql /tmp/sql_parts.sql.gz
```

> **After a `git pull`, rebuild before running anything new.** The production
> stack runs baked images with no source bind-mount, so pulled code is not in the
> containers until you rebuild — and `pyroforge-fetch-parts` in particular is a
> console script generated during `uv sync` at *image build* time, so a new entry
> point fails with `Failed to spawn: pyroforge-fetch-parts` until then:
>
> ```bash
> docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
> ```
>
> Pass the same `-f` flags to every `docker compose` call. Without them compose
> merges `docker-compose.override.yml` (the dev stack) and you are configuring a
> different set of services than the one you are running.

### Self-hosted ESMFold

3D structures come from `esmfold`, a container on its own card (`ESMFOLD_GPU`,
default 1) built from [services/esmfold](../services/esmfold). It answers the same
request as `api.esmatlas.com/foldSequence/v1/pdb/`, and the public API stays
configured as the fallback for when it is down.

Why not the public API alone: it has no SLA, its repository was archived in August
2024, and in September 2026 it returned `504` after 30 s for every sequence it had
not folded before — only previously folded sequences still came back.

The container runs offline and mounts the weights from the host, so fetch them
once, as the user whose `HOME` the stack runs with (`HOME` in
`deploy/pyroforge.service` if you use it). `benchmark.py` downloads them and
measures the card at the same time:

```bash
python3 -m venv ~/esmfold-venv && source ~/esmfold-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.17.0 accelerate==1.15.0
CUDA_VISIBLE_DEVICES=1 python services/esmfold/benchmark.py --out ~/benchmark_out
```

Measured on an RTX 2080 Ti (22 GB), 2026-09-14:

| Residues | Fold time | Peak VRAM |
|---|---|---|
| 84 | 3.3 s | 7.98 GB |
| 100 | 2.2 s | 8.00 GB |
| 200 | 4.4 s | 8.20 GB |
| 300 | 11.2 s | 8.53 GB |
| 400 | 22.8 s | 8.99 GB |

The service folds **one sequence at a time**: a second concurrent fold on the same
card competes for the same compute and adds no throughput. Up to
`ESMFOLD_MAX_WAITING` (8) requests wait behind it; past that it answers `503`, and
the worker retries and then moves to the fallback. 8 is sized so the last request
in a full queue of 400-residue folds (~205 s) still finishes inside the worker's
300 s read timeout.

If one card stops being enough, add a second card rather than more concurrency on
this one — the worker does not yet spread load across two `esmfold` containers.

### The iGEM parts snapshot

Registry Interoperability searches a **local protein snapshot** of the iGEM
Registry, built by step 3 into `./backend/data/` (bind-mounted into the backend
and every worker, so it survives an image rebuild and all containers see the
same file). Until it exists the engine reports itself as `not_provisioned` and
the UI says so — it will not answer queries against a stand-in database.

**Step 3 no longer works on its own.** `https://parts.igem.org/fasta/parts/All_Parts`
returns **403 from iGEM's own Apache** — not from the CDN in front of it, which is
the difference that matters: the origin is refusing the path, so no retry,
user-agent or different network brings it back. The endpoint is gone, not blocked.

What the Registry does still publish is a **mysqldump of the `parts` table**,
linked from `registry.igem.org`. It carries more than the FASTA did (part type and
status as well as the sequence), so deleted and non-coding parts are dropped before
translation instead of after. Download it in a browser, then:

```bash
docker compose cp sql_parts.sql.gz backend:/tmp/sql_parts.sql.gz
docker compose exec backend uv run pyroforge-fetch-parts --from-sql /tmp/sql_parts.sql.gz
```

Expect roughly (measured on the 2026-09 dump, ~52 MB gzipped):

```
84,285 rows in dump; 75,672 with a usable sequence (6,484 deleted/discontinued, 2,129 empty sequence)
Wrote 22,294 protein parts → /app/data/igem_parts.fasta
```

The drop from 75,672 to 22,294 is expected and is not a failure: most Registry
parts are promoters, RBSs, terminators and scars with no ORF, and 33,955 of the
translated proteins are exact duplicates of another part. Parsing the dump takes
about 15 s; translation a couple of minutes.

If you have an older `All_Parts.fasta` from before the endpoint was withdrawn, that
route still exists and produces an equivalent snapshot:

```bash
docker compose exec backend uv run pyroforge-fetch-parts --from-file /tmp/All_Parts.fasta
```

The Registry distributes DNA; the search is over protein. The script translates
the longest ORF of each part, drops non-coding parts and anything outside
30–2000 aa, and de-duplicates. Check what you got:

```bash
cat backend/data/igem_parts.manifest.json
```

Every search result carries `database.n_parts` and the snapshot date, and the
card prints them — so "no homologs" is always readable next to how much was
actually searched.

> **Sizing.** With more than 2,000 parts the pairwise fallback is refused as too
> slow and mmseqs2 becomes mandatory. It is installed in the backend image, so
> this only bites if you run outside Docker.

To start it automatically on boot, install the systemd unit — edit
`WorkingDirectory` and the `HOME`/`DEVICE` values in it first:

```bash
sudo cp deploy/pyroforge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now pyroforge
```

### Check it worked

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3002/                  # 200, the SPA
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3002/api/v1/admin/train # 403, blocked
docker compose exec backend curl -s localhost:8000/health                        # backend
docker compose exec worker-io wget -qO- http://esmfold:8000/health               # esmfold, as the worker sees it
docker compose exec worker-io sh -c \
  'wget -qO- --post-data=MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ http://esmfold:8000/foldSequence/v1/pdb/ | head -3'
```

---

## What the frontend nginx enforces

`:3002` is the only published port in the production stack — the backend and
Redis are reachable only over the compose network, so
[frontend/nginx.conf](../frontend/nginx.conf) is the single chokepoint and
carries the protections that used to live in the trial's auth gateway:

| Rule | Why |
|---|---|
| `POST /api/v1/admin/train` → 403 | Full retrain; monopolises the GPU. Still runnable server-side, see below. |
| GPU endpoints: 10/min per IP, burst 5 | The Celery pool is `--concurrency=1`; an unthrottled client can queue hours of work and lock everyone out. Covers `optimize`, `predict`, `predict/job`, `visualize/fold`, `visualize/heatmap`, `embed/compare`, `registry/search`. |
| `/api/v1/jobs/…`: 180/min per IP, burst 60 | Status polling runs every 2s per job — deliberately generous. |
| Other `/api/`: 60/min per IP | Cheap reads. |
| 10 concurrent connections per IP | Stops one client parking connections on the 300s proxy timeout. |
| `client_max_body_size 1m` | The largest legitimate request is a 1000-residue sequence, ~1 KB. |

Throttled requests get `429`. To retrain, or to reach the API unthrottled, go
through the container rather than the host:

```bash
docker compose exec backend curl -X POST localhost:8000/api/v1/admin/train
```

---

## ⚠️ Before exposing it publicly

- **No authentication.** The rate limits slow abuse down; they are not access
  control. Anyone who can reach `:3002` can queue GPU work. Keep the deployment
  on a trusted network (a tailnet, a campus VPN) or put your own access control
  in front of it.
- **Docker's published ports bypass `ufw`/`firewalld`.** `3002:80` writes
  straight into the Docker iptables chain, ahead of your firewall rules — a
  `ufw deny` will *not* close it. To genuinely restrict it, bind the port to one
  interface in `docker-compose.yml`, e.g. `"100.65.250.65:3002:80"` for a
  tailnet-only deployment.
- **Plain `http://` is readable in transit** on a shared LAN. Put TLS in front
  before handing the link out beyond a trusted network.
- **Sequences can reach a third party.** `/api/v1/visualize/fold` goes to the
  self-hosted `esmfold` service first. While that is down or its queue is full,
  the worker falls back to `api.esmatlas.com` and the sequence leaves the server.
  Set `ESMFOLD_FALLBACK_API_URL=` (empty) in `.env` to keep sequences here and
  accept failed folds during those windows instead.
- **Nothing here quotes a price.** `TWIST_API_KEY` no longer exists, and
  `/api/v1/procure/quote` returns no `price_usd` and no `lead_time_days` — both
  used to be invented locally (a hard-coded per-bp band table, and
  `10 if length < 1500 else 15`), and the second carried no estimate flag at all.
  The card shows what the app actually computes — the CDS, its CAI against a stated
  reference, GC content, feasibility flags — and links out. Pricing and ordering
  are Twist's, not ours.
- **Containers run as root** (neither Dockerfile sets `USER`), and the backend
  bind-mounts `./backend/checkpoints` and `~/.cache/torch` read-write. Worth
  fixing before this runs anywhere untrusted.

---

## Job monitoring

`POST /api/v1/optimize` returns immediately and the real GPU work happens in the
Celery worker afterwards, so the API log is not where job time shows up.

```bash
docker compose logs -f worker-heavy     # optimize/train: starts, failures, durations
docker compose logs -f worker-fast      # predict/registry
docker compose logs -f worker-io        # folding; a fallback to the public API logs a warning here
docker compose logs -f esmfold          # fold service: model load, 503 queue-full, 499 callers that gave up
docker compose ps                       # container health
docker stats                            # GPU-host CPU/memory pressure

# Queue backlog. A number that keeps climbing on `heavy` means jobs arrive
# faster than one worker drains them — scale it: `--scale worker-heavy=2`.
docker compose exec redis redis-cli llen fast
docker compose exec redis redis-cli llen heavy
docker compose exec redis redis-cli llen io
```

Rate-limit rejections show up as `429` in the frontend's access log:

```bash
docker compose logs frontend | grep ' 429 '
```
