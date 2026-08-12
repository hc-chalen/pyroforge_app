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

Only `:3002` is bound publicly; the backend and Redis stay on `127.0.0.1`.

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
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3002/       # 200, the SPA
curl -s http://localhost:8000/health                                  # backend, loopback only
```

---

## ⚠️ Before exposing it publicly

- **No authentication.** Anyone who can reach `:3002` can queue GPU work. Keep
  the deployment on a trusted network (a tailnet, a campus VPN) or put your own
  access control in front of it if that is not acceptable.
- **`POST /api/v1/admin/train` kicks off a full retrain** and will monopolise
  the GPU for as long as it runs. It is reachable through the `/api/` proxy like
  any other endpoint. If the deployment is publicly reachable, block that path
  in the frontend nginx config or drop the route.
- **Plain `http://` is readable in transit** on a shared LAN. Put TLS in front
  before handing the link out beyond a trusted network.

---

## Job monitoring

`POST /api/v1/optimize` returns immediately and the real GPU work happens in the
Celery worker afterwards, so the API log is not where job time shows up.

```bash
docker compose logs -f celery-worker    # job starts, failures, durations
docker compose ps                       # container health
docker stats                            # GPU-host CPU/memory pressure
```
