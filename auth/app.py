"""PyroForge access gate — a standalone session-auth service.

This service exists ONLY to keep the trial deployment non-public. It is
deliberately isolated from the application:

  * it knows nothing about PyroForge's API, models or data;
  * PyroForge knows nothing about it (no imports, no shared config);
  * the nginx gateway in front of the app calls `/verify` as an `auth_request`
    subrequest and never routes application traffic through this process.

Deleting the `auth/` directory and `docker-compose.auth.yml` removes the whole
feature. See docs/OPERATIONS.md ("Removing the login layer").

Endpoints
    GET  /verify   internal — 200 + `X-Auth-User` when the session is valid, else 401
    GET  /login    the login page
    POST /login    form submit → sets the session cookie, redirects to `next`
    GET  /logout   clears the session cookie
    GET  /healthz  liveness probe
"""

from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import bcrypt
import yaml
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

logger = logging.getLogger("pyroforge.auth")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

COOKIE_NAME = "pyroforge_session"
_SALT = "pyroforge-session-v1"

USERS_FILE = Path(os.environ.get("AUTH_USERS_FILE", "/app/users.yml"))
SESSION_DAYS = int(os.environ.get("AUTH_SESSION_DAYS", "14"))
COOKIE_SECURE = os.environ.get("AUTH_COOKIE_SECURE", "false").lower() == "true"
MAX_AGE = int(timedelta(days=SESSION_DAYS).total_seconds())

# A dummy hash used to equalise timing between "no such user" and "wrong
# password", so the login form cannot be used to enumerate valid usernames.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt(rounds=12))

_secret = os.environ.get("AUTH_SECRET_KEY", "")
if not _secret:
    # Fail soft rather than refusing to boot, but make the consequence loud:
    # every restart invalidates all sessions and forces everyone to log in again.
    _secret = secrets.token_urlsafe(32)
    logger.warning(
        "AUTH_SECRET_KEY is not set — generated an ephemeral one. All sessions "
        "will be invalidated on restart. Set AUTH_SECRET_KEY in .env."
    )

serializer = URLSafeTimedSerializer(_secret, salt=_SALT)

# The template ships a `demo` account whose password is published in
# users.example.yml, so that local testing works out of the box. On a reachable
# server that is a free account for anyone who reads the repo — warn loudly, on
# startup and on every use, until it is removed.
DEMO_ACCOUNT = "demo"

_LOGIN_TEMPLATE = (Path(__file__).parent / "login.html").read_text(encoding="utf-8")


def load_users() -> dict[str, bytes]:
    """Read username → bcrypt hash from the users file.

    Read on every login rather than cached at startup so that adding a user is
    a file edit, with no container restart and no dropped sessions.
    """
    if not USERS_FILE.is_file():
        # A missing bind-mount source makes Docker create a *directory* here,
        # which is the usual cause — hence is_file() rather than exists().
        logger.error(
            "Users file %s is missing or not a file — all logins will fail. "
            "Did you `cp auth/users.example.yml auth/users.yml` first?",
            USERS_FILE,
        )
        return {}
    try:
        raw = yaml.safe_load(USERS_FILE.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        logger.exception("Users file %s is not valid YAML — all logins will fail.", USERS_FILE)
        return {}

    users = raw.get("users") or {}
    if not isinstance(users, dict):
        logger.error("Users file %s: 'users' must be a mapping of name → hash.", USERS_FILE)
        return {}
    return {str(name): str(h).encode("utf-8") for name, h in users.items()}


def check_password(username: str, password: str) -> bool:
    users = load_users()
    stored = users.get(username)
    # Always run one bcrypt comparison, present or not — see _DUMMY_HASH.
    to_check = stored if stored is not None else _DUMMY_HASH
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), to_check)
    except ValueError:
        logger.error("User %r has a malformed bcrypt hash in %s.", username, USERS_FILE)
        return False
    return ok and stored is not None


def safe_next(raw: str | None) -> str:
    """Constrain the post-login redirect to a path on this site.

    Rejects absolute URLs, scheme-relative `//evil.com` and backslash variants
    so the login form cannot be turned into an open redirect.
    """
    if not raw or not raw.startswith("/"):
        return "/"
    if raw.startswith("//") or raw.startswith("/\\"):
        return "/"
    return raw


def set_session_cookie(response: Response, username: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        serializer.dumps(username),
        max_age=MAX_AGE,
        httponly=True,
        samesite="lax",
        # Only send over HTTPS when the deployment actually has TLS; forcing
        # this on a plain-http LAN deployment would silently break every login.
        secure=COOKIE_SECURE,
        path="/",
    )


def read_session(request: Request) -> str | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        return serializer.loads(token, max_age=MAX_AGE)
    except SignatureExpired:
        return None
    except BadSignature:
        logger.warning("Rejected a session cookie with a bad signature.")
        return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    users = load_users()
    logger.info("Loaded %d account(s) from %s", len(users), USERS_FILE)
    if DEMO_ACCOUNT in users:
        logger.warning(
            "=" * 70
            + "\nSECURITY: the example %r account is enabled. Its password is "
            "published\nin auth/users.example.yml — anyone who reads the repo can "
            "log in.\nRemove it from auth/users.yml before this server is "
            "reachable by others.\n" + "=" * 70,
            DEMO_ACCOUNT,
        )
    yield


app = FastAPI(
    title="PyroForge auth",
    lifespan=lifespan,
    # No API docs: this service is only ever called by nginx and the login form.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    users = load_users()
    return {
        "status": "ok",
        "users": str(len(users)),
        # Surfaced so `docker compose ps`-style checks can catch it too.
        "example_account_enabled": str(DEMO_ACCOUNT in users).lower(),
    }


@app.get("/verify")
def verify(request: Request) -> Response:
    """`auth_request` target for the nginx gateway.

    200 → nginx serves the request, and copies `X-Auth-User` into the access log
    so usage can be attributed per account. 401 → nginx redirects to /login (or
    returns JSON for API calls).
    """
    username = read_session(request)
    if username is None:
        return Response(status_code=401)

    response = Response(status_code=200)
    response.headers["X-Auth-User"] = username

    # Sliding expiry: re-issue once the session is past its halfway point, so an
    # actively-used session never expires mid-task. The gateway forwards this
    # Set-Cookie to the browser.
    try:
        serializer.loads(request.cookies[COOKIE_NAME], max_age=MAX_AGE // 2)
    except SignatureExpired:
        set_session_cookie(response, username)
    except BadSignature:  # pragma: no cover — read_session already validated it
        pass
    return response


def render_login(next_url: str, error: str = "", status_code: int = 200) -> HTMLResponse:
    error_block = (
        f'<p class="error" role="alert">{error}</p>' if error else '<p class="error"></p>'
    )
    html = _LOGIN_TEMPLATE.replace("{{ERROR}}", error_block).replace(
        "{{NEXT}}", next_url.replace('"', "&quot;")
    )
    return HTMLResponse(html, status_code=status_code)


@app.get("/login")
def login_page(request: Request, next: str = "/") -> Response:
    # Already signed in → skip the form.
    if read_session(request) is not None:
        return RedirectResponse(safe_next(next), status_code=303)
    return render_login(safe_next(next))


@app.post("/login")
def login_submit(
    username: str = Form(default=""),
    password: str = Form(default=""),
    next: str = Form(default="/"),
) -> Response:
    target = safe_next(next)
    username = username.strip()

    if not check_password(username, password):
        logger.info("Failed login for %r", username)
        # One message for both cases — never reveal whether the user exists.
        return render_login(target, "Incorrect username or password.", status_code=401)

    if username == DEMO_ACCOUNT:
        logger.warning(
            "SECURITY: someone logged in as the example %r account. Its password "
            "is published in auth/users.example.yml — remove it from "
            "auth/users.yml now.",
            DEMO_ACCOUNT,
        )

    logger.info("Successful login for %r", username)
    response = RedirectResponse(target, status_code=303)
    set_session_cookie(response, username)
    return response


@app.get("/logout")
def logout() -> Response:
    response = RedirectResponse("/_auth/login", status_code=303)
    response.delete_cookie(COOKIE_NAME, path="/")
    return response
