# PyroForge — trial deployment operations

## Quick reference

**Which command gives you a login page?** Only the ones passing
`-f docker-compose.auth.yml`. Compose auto-merges `docker-compose.override.yml`
(the dev stack) but never the auth overlay, so plain `docker compose up` is
unchanged from before.

| What you want | Command | Open | Login? |
|---|---|---|---|
| Dev, hot reload | `docker compose up` | `:5173` | **no** |
| Test the login locally | `docker compose -f docker-compose.yml -f docker-compose.auth.yml up` | `:3002` | **yes** |
| The real server | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.auth.yml -f docker-compose.monitor.yml up -d` | `:3002` | **yes** |

Before the last two work once: `cp auth/users.example.yml auth/users.yml` and
put `AUTH_SECRET_KEY=...` in `.env` (see [First-time setup](#first-time-setup-on-the-server)).

| Task | Command |
|---|---|
| Add a user | `docker compose -f docker-compose.yml -f docker-compose.auth.yml run --rm --no-deps auth-service python hashpw.py NAME` |
| Remove a user | delete their line in `auth/users.yml` (takes effect immediately) |
| Who used what | `python3 monitoring/usage_report.py` |
| Traffic dashboard | tunnel `7890`, open `http://localhost:7890` |
| Job durations | tunnel `5555`, open `http://localhost:5555` |
| Remove the gate | [teardown checklist](#removing-the-login-layer-after-the-trial) |

---

Everything in this document covers the **temporary access gate** and **usage
monitoring** added for the limited-access trial on the school GPU server. Both
are designed to be deleted in one go afterwards — see
[Removing the login layer](#removing-the-login-layer-after-the-trial).

The application itself is untouched by any of it: there is no auth code in
`backend/app/**` or `frontend/src/**`, and no user column in any data structure.

---

## How it fits together

```
              ┌──────────────────────────────────────────────┐
  browser ──► │ :3002  auth-gateway (nginx)                  │  ← only public port
              │        every request → auth_request ─────────┼──► auth-service
              │        no valid session → /_auth/login       │    (login page,
              └───────────────────┬──────────────────────────┘     bcrypt, cookie)
                                  │ authenticated
                                  ▼
                    :3001 frontend (nginx, loopback)
                                  │ /api/
                                  ▼
                    :8000 backend ──► redis ──► celery-worker (GPU)
```

Two things worth knowing:

- **Analyses are stored in the browser, not on the server.** The frontend keeps
  them in `localStorage` under `pyroforge-analyses`. Accounts therefore need no
  server-side data separation — two users never see each other's results, and
  removing the login layer later does not touch anyone's saved work. The one
  exception: two people sharing the *same browser profile on the same machine*
  share that storage, because it is keyed by browser, not by account.
- **The gate is the only thing standing between the internet and the GPU.**
  Only `:3002` is bound publicly; the frontend, backend, Redis and both
  dashboards are all on `127.0.0.1`.

---

## First-time setup on the server

```bash
cd ~/pyroforge_app

# 1. Accounts file (never committed — auth/users.yml is gitignored)
cp auth/users.example.yml auth/users.yml

# 2. Cookie signing secret
cp .env.example .env
echo "AUTH_SECRET_KEY=$(openssl rand -base64 32)" >> .env

# 3. Create a real account (see "Managing accounts" below), then DELETE the
#    demo account from auth/users.yml before anyone else can reach the server.

# 4. Bring the stack up
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
               -f docker-compose.auth.yml -f docker-compose.monitor.yml up -d
```

To start it automatically on boot, install the systemd unit — it already
includes both overlays:

```bash
sudo cp deploy/pyroforge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now pyroforge
```

### Check it worked

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3002/          # 302 → login
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3002/api/v1/predict  # 401
```

---

## Managing accounts

Give each person their own account. It costs nothing and it is what makes the
usage report meaningful.

**Add a user:**

```bash
docker compose -f docker-compose.yml -f docker-compose.auth.yml \
  run --rm --no-deps auth-service python hashpw.py alice
```

It prompts for a password (never passed on the command line, so it stays out of
your shell history) and prints a line to paste under `users:` in
`auth/users.yml`. The file is re-read on every login attempt — **no restart
needed, and nobody gets logged out.**

**Remove a user:** delete their line. This blocks future logins but does *not*
kill an existing session.

**Force everyone out immediately** (e.g. at the end of the trial):

```bash
sed -i "s|^AUTH_SECRET_KEY=.*|AUTH_SECRET_KEY=$(openssl rand -base64 32)|" .env
docker compose ... up -d --force-recreate auth-service
```

Changing the signing key invalidates every session at once.

### Sessions

14 days by default (`AUTH_SESSION_DAYS`), and a session in active use renews
itself once past the halfway mark, so it is really a 7-day idle timeout. If a
session does expire mid-visit, the page reload shows the login form; API calls
made in the meantime return `401 {"detail": "Not authenticated..."}` rather than
a confusing HTML response.

---

## ⚠️ Transport security

Session cookies and passwords are only as private as the connection carrying
them. **Over plain `http://` on a shared LAN, both are readable in transit.**

- **On Tailscale** (your `100.65.x.x` address suggests this): the tailnet is
  already encrypted end-to-end, so plain http between tailnet members is fine.
  Keep `AUTH_COOKIE_SECURE=false`.
- **On a bare LAN/campus IP**: put TLS in front before handing out accounts,
  then set `AUTH_COOKIE_SECURE=true` in `.env` and restart `auth-service`.

Do not set `AUTH_COOKIE_SECURE=true` without HTTPS — browsers would refuse to
send the cookie and nobody could log in.

---

## Monitoring usage

Three views, none of them public. Open the two dashboards over an SSH tunnel:

```bash
ssh -L 7890:localhost:7890 -L 5555:localhost:5555 you@gpu-server
```

### 1. Per-account usage — the main one

```bash
python3 monitoring/usage_report.py              # everything so far
python3 monitoring/usage_report.py --since 7    # last week
python3 monitoring/usage_report.py --format csv > usage.csv
```

Standard library only; no install, no container. Shows per account: request
count, active days, GPU-job count broken down by kind (optimize / fold /
predict / …), p50 and p95 latency, last seen, plus a daily-active-accounts
chart. It also flags accounts seen from more than three addresses, which
usually means a password has been shared around.

### 2. Traffic shape — GoAccess, <http://localhost:7890>

Requests over time, busiest endpoints, status codes, slow requests. Regenerated
every 60s from the same log. GoAccess has no per-account panel — that is what
the script above is for.

### 3. Job durations — Flower, <http://localhost:5555>

The Celery view: how long optimisation and folding jobs actually take, which
fail, and whether the worker is keeping up. This is where real GPU time shows
up, since `POST /api/v1/optimize` returns immediately and the work happens in
the worker afterwards. To attribute a long job to a person, match its start
time against the gateway log.

### The raw log

`logs/usage.log`, one JSON object per request:

```json
{"ts":"2026-07-26T13:49:38+00:00","date":"2026-07-26","time":"13:49:38",
 "user":"alice","ip":"100.65.250.3","method":"POST","path":"/api/v1/optimize",
 "status":200,"bytes":412,"req_time":0.031,"upstream_time":"0.030","ua":"..."}
```

`user` is empty for unauthenticated requests (mostly the login page itself).
`date`/`time` are split out because GoAccess's `strptime` cannot read the
`+00:00` offset in `ts`.

**Rotation:** the log grows unbounded. At a handful of users it is a few MB a
month, but for a long trial add a rotation rule:

```
sudo tee /etc/logrotate.d/pyroforge <<'EOF'
/home/codingteam/pyroforge_app/logs/usage.log {
    weekly
    rotate 8
    compress
    missingok
    notifempty
    copytruncate
}
EOF
```

`copytruncate` matters — nginx holds the file open, so a plain rotation would
leave it writing to the rotated inode.

---

## Things deliberately locked down

- **`POST /api/v1/admin/train` returns 403 at the gateway.** It kicks off a full
  retrain and would monopolise the GPU. Run it on the server against the
  loopback backend instead: `curl -X POST 127.0.0.1:8000/api/v1/admin/train`.
- **Login is rate-limited** to 12/min per IP with a burst of 5; beyond that the
  gateway returns 503.
- **Failed logins never say whether the username exists**, and a bcrypt
  comparison runs even for unknown users so response timing cannot be used to
  enumerate accounts.
- **The `next=` redirect parameter only accepts local paths**, so the login form
  cannot be used to bounce someone to another site.

---

## Removing the login layer (after the trial)

The whole feature is two paths plus three lines. Nothing in `backend/app/**` or
`frontend/src/**` needs to change, and no user data is affected.

```bash
cd ~/pyroforge_app

# 1. Stop the stack
sudo systemctl stop pyroforge   # or: docker compose ... down

# 2. Delete the feature
rm -rf auth/ monitoring/ logs/
rm docker-compose.auth.yml docker-compose.monitor.yml
```

**3.** In `docker-compose.yml`, give the public port back to the frontend:

```yaml
  frontend:
    ports:
      - "3002:80"        # was 127.0.0.1:3001:80 during the trial
```

**4.** In `deploy/pyroforge.service`, drop `-f docker-compose.auth.yml
-f docker-compose.monitor.yml` from both `ExecStart` and `ExecStop`.

**5.** In `.env` / `.env.example`, delete the `AUTH_*` block.

```bash
sudo systemctl daemon-reload && sudo systemctl start pyroforge
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config   # sanity check
```

Then remove this file. Grep for leftovers with:

```bash
grep -rn "auth-gateway\|auth-service\|AUTH_SECRET\|_auth/" --exclude-dir=node_modules .
```

> **Note for whoever does this:** users' saved analyses live in their own
> browsers' `localStorage`, not on the server, so none of the above touches
> them. The stack after removal is exactly what it was before the trial: the
> frontend container serving the SPA on `:3002` and proxying `/api/` to the
> backend.
