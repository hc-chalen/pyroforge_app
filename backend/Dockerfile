FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install MMseqs2. Upstream ships one tarball per instruction set, so the build
# must be picked to match the image architecture — the avx2 one is x86_64-only.
# On Apple Silicon it does not fail loudly at install time: Docker Desktop's
# Rosetta layer runs the x86 binary right up to the first AVX2 instruction and
# then the process dies with SIGILL, surfacing as a registry-search error.
# TARGETARCH is supplied automatically by BuildKit.
ARG TARGETARCH
RUN case "$TARGETARCH" in \
        arm64) MMSEQS_BUILD=arm64 ;; \
        amd64) MMSEQS_BUILD=avx2 ;; \
        *) echo "No MMseqs2 build mapped for TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac \
    && wget -q "https://mmseqs.com/latest/mmseqs-linux-${MMSEQS_BUILD}.tar.gz" -O /tmp/mmseqs.tar.gz \
    && mkdir -p /opt/mmseqs \
    && tar -xzf /tmp/mmseqs.tar.gz -C /opt/mmseqs --strip-components=1 \
    && ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs \
    && rm /tmp/mmseqs.tar.gz \
    # Fail the build here rather than at runtime if the binary cannot execute.
    && mmseqs version

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
RUN uv run python -c "import matplotlib; import torch; import fastapi"

RUN mkdir -p checkpoints cache data cache/structures checkpoints/plots

ENV MMSEQS_BIN=/usr/local/bin/mmseqs
ENV PATH="/app/.venv/bin:$PATH"

# Production default: no --reload (single-process, no file watcher). The dev
# stack re-adds --reload via docker-compose.override.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
