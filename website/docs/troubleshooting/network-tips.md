---
title: Network Tips
sidebar_label: Network Tips
---

This guide shows how to build and run in restricted or slow network environments without modifying repo files. You’ll use small local override files and a compose override so the codebase stays clean.

What you’ll solve:

- Hugging Face model downloads blocked/slow
- Go modules fetching blocked during Docker build
- PyPI access for the mock-vLLM test image

## TL;DR: Choose your path

- Fastest and most reliable: use local models in `./models` and skip HF network entirely.
- Otherwise: mount an HF cache + set mirror env vars via a compose override.
- For building: use an override Dockerfile to set Go mirrors (examples provided).
- For mock-vllm: use an override Dockerfile to set pip mirror (examples provided).

You can mix these based on your situation.

## 1. Hugging Face models

The router will download embedding models on first run unless you provide them locally. Prefer Option A if possible.

### Option A — Use local models (no external network)

1) Download the required model(s) with any reachable method (VPN/offline) into the repo’s `./models` folder. Example layout:

   - `models/all-MiniLM-L12-v2/`
   - `models/category_classifier_modernbert-base_model`

2) In `config/config.yaml`, point to the local path. Example:

   ```yaml
   bert_model:
     # point to a local folder under /app/models (already mounted by compose)
     model_id: /app/models/all-MiniLM-L12-v2
   ```

3) No extra env is required. `docker-compose.yml` already mounts `./models:/app/models:ro`.

### Option B — Use HF cache + mirror

Create a compose override to persist cache and use a regional mirror (example below uses a China mirror). Save as `docker-compose.override.yml` in the repo root:

```yaml
services:
  semantic-router:
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface
      - HF_HUB_ENABLE_HF_TRANSFER=1
      - HF_ENDPOINT=https://hf-mirror.com  # example mirror endpoint (China)
```

Optional: pre-warm cache on the host (only if you have `huggingface_hub` installed):

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2")
PY
```

## 2. Build with Go mirrors (Dockerfile override)

When building `Dockerfile.extproc`, the Go stage may hang on `proxy.golang.org`. Create an override Dockerfile that enables mirrors without touching the original.

1) Create `Dockerfile.extproc.cn` at repo root with this content:

```Dockerfile
# syntax=docker/dockerfile:1

FROM rust:1.85 AS rust-builder
RUN apt-get update && apt-get install -y make build-essential pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY tools/make/ tools/make/
COPY Makefile ./
COPY candle-binding/Cargo.toml candle-binding/
COPY candle-binding/src/ candle-binding/src/
RUN make rust

FROM golang:1.24 AS go-builder
WORKDIR /app

# Go module mirrors (example: goproxy.cn)
ENV GOPROXY=https://goproxy.cn,direct
ENV GOSUMDB=sum.golang.google.cn

RUN mkdir -p src/semantic-router
COPY src/semantic-router/go.mod src/semantic-router/go.sum src/semantic-router/
COPY candle-binding/go.mod candle-binding/semantic-router.go candle-binding/

# Pre-download modules to fail fast if mirrors are unreachable
RUN cd src/semantic-router && go mod download && \
    cd /app/candle-binding && go mod download

COPY src/semantic-router/ src/semantic-router/
COPY --from=rust-builder /app/candle-binding/target/release/libcandle_semantic_router.so /app/candle-binding/target/release/

ENV CGO_ENABLED=1
ENV LD_LIBRARY_PATH=/app/candle-binding/target/release
RUN mkdir -p bin && cd src/semantic-router && go build -o ../../bin/router cmd/main.go

FROM quay.io/centos/centos:stream9
WORKDIR /app
COPY --from=go-builder /app/bin/router /app/extproc-server
COPY --from=go-builder /app/candle-binding/target/release/libcandle_semantic_router.so /app/lib/
COPY config/config.yaml /app/config/
ENV LD_LIBRARY_PATH=/app/lib
EXPOSE 50051
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
```

2) Point compose to the override Dockerfile by extending `docker-compose.override.yml`:

```yaml
services:
  semantic-router:
    build:
      dockerfile: Dockerfile.extproc.cn
```

## 3. Mock vLLM (PyPI mirror via Dockerfile override)

For the optional testing profile, create an override Dockerfile to configure pip mirrors.

1) Create `tools/mock-vllm/Dockerfile.cn`:

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Pip mirror (example: TUNA mirror in China)
RUN python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

2) Extend `docker-compose.override.yml` to use the override Dockerfile for `mock-vllm`:

```yaml
services:
  mock-vllm:
    build:
      dockerfile: Dockerfile.cn
```

## 4. Build and run

With the overrides in place, build and run normally (Compose will auto-merge):

```bash
# Build all images with overrides
docker compose -f docker-compose.yml -f docker-compose.override.yml build

# Run router + envoy
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# If you need the testing profile (mock-vllm)
docker compose -f docker-compose.yml -f docker-compose.override.yml --profile testing up -d
```

## 5. Troubleshooting

- Go modules still time out:
  - Verify `GOPROXY` and `GOSUMDB` are present in the go-builder stage logs.
  - Try a clean build: `docker compose build --no-cache`.

- HF models still download slowly:
  - Prefer Option A (local models).
  - Ensure the cache volume is mounted and `HF_ENDPOINT`/`HF_HUB_ENABLE_HF_TRANSFER` are set.

- PyPI slow for mock-vllm:
  - Confirm the CN Dockerfile is being used for that service.
