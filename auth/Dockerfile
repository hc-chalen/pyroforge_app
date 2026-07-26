# ---------------------------------------------------------------------------
# PyroForge access gate — auth service image.
#
# Intentionally standalone: it does not build on the backend image and shares no
# code with it, so `rm -rf auth/` removes the feature without touching the app.
# Dependencies are pinned inline; there is no lock file to maintain for ~5 small
# packages.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.115.6" \
    "uvicorn[standard]==0.34.0" \
    "python-multipart==0.0.20" \
    "itsdangerous==2.2.0" \
    "bcrypt==4.2.1" \
    "PyYAML==6.0.2"

COPY app.py login.html hashpw.py ./

# Drop privileges — this process is the one exposed to unauthenticated traffic.
RUN useradd --system --uid 10001 authsvc
USER authsvc

EXPOSE 9000

# Single worker: state is a signed cookie, so there is nothing to share, and the
# load is a handful of logins per day.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
