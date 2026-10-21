FROM python:3.12-slim

# Triton JIT-compiles its CUDA driver stubs on the first fold (the warm-up in
# folding.py). python:3.12-slim ships no compiler, so without these the service
# dies with: RuntimeError: Failed to find C compiler. Please specify via CC …
# python3-dev is for Python.h; gcc + libc6-dev for the actual compile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

# torch from the CUDA 13.0 index: the same build the backend image already runs on
# this host's RTX 2080 Ti cards (driver 580, CUDA 13.0). The wheel bundles the CUDA
# runtime; the NVIDIA Container Toolkit supplies the driver.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY folding.py server.py ./

# Weights are mounted at /hf, not baked in — ~8 GB re-sent on every build otherwise.
# Offline, so a missing mount fails at startup instead of silently starting a
# 10 GB download inside the container.
ENV HF_HOME=/hf \
    HF_HUB_OFFLINE=1

EXPOSE 8000

# Exactly one worker: each additional worker loads another copy of the model.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
