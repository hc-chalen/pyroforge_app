FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install MMseqs2
RUN wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mmseqs.tar.gz \
    && mkdir -p /opt/mmseqs \
    && tar -xzf /tmp/mmseqs.tar.gz -C /opt/mmseqs --strip-components=1 \
    && ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs \
    && rm /tmp/mmseqs.tar.gz

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
