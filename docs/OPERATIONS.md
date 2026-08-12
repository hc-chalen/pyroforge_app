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
                    :8000 backend ──► redis ──► celery-worker (GPU)
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

# 2. Bring the stack up (first boot downloads ESM-2 weights, ~2.5 GB)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

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
- **Sequences are sent to a third party.** `/api/v1/visualize/fold` posts the
  sequence to `api.esmatlas.com` (ESMFold). Unpublished sequences leave the
  server; say so to your users, or point `ESMFOLD_API_URL` at your own instance.
- **Leave `TWIST_API_KEY` unset** on a public deployment. With it set,
  `/api/v1/procure/quote` spends your Twist quota on behalf of anonymous callers.
- **Containers run as root** (neither Dockerfile sets `USER`), and the backend
  bind-mounts `./backend/checkpoints` and `~/.cache/torch` read-write. Worth
  fixing before this runs anywhere untrusted.

---

## Job monitoring

`POST /api/v1/optimize` returns immediately and the real GPU work happens in the
Celery worker afterwards, so the API log is not where job time shows up.

```bash
docker compose logs -f celery-worker    # job starts, failures, durations
docker compose ps                       # container health
docker stats                            # GPU-host CPU/memory pressure
```

Rate-limit rejections show up as `429` in the frontend's access log:

```bash
docker compose logs frontend | grep ' 429 '
```
