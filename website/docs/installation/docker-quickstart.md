---
sidebar_position: 3
---

# Install with Docker Compose

Run Semantic Router + Envoy locally using Docker Compose v2.

## Prerequisites

- Docker Engine, see more in [Docker Engine Installation](https://docs.docker.com/engine/install/) 
- Docker Compose v2 (use the `docker compose` command, not the legacy `docker-compose`)

  Docker Compose Plugin Installation(if missing), see more in [Docker Compose Plugin Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

  ```bash
  # For Ubuntu and Debian, run:
  sudo apt-get update
  sudo apt-get install -y docker-compose-plugin

  # For RPM-based distributions, run:
  sudo yum update
  sudo yum install docker-compose-plugin

  # Verify
  docker compose version
  ```

- Ensure ports 8801, 50051, 19000 are free

## Install and Run with Docker Compose v2

**1. Clone the repo and move into it (from your workspace root)**

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
```

**2. Download required models (classification models)**

```bash
make download-models
```

This downloads the classification models used by the router:

- Category classifier (ModernBERT-base)
- PII classifier (ModernBERT-base)
- Jailbreak classifier (ModernBERT-base)

Note: The BERT similarity model defaults to a remote Hugging Face model. See Troubleshooting for offline/local usage.

**3. Start the services with Docker Compose v2**

```bash
# Start core services (semantic-router + envoy)
docker compose up --build

# Or run in background (recommended)
docker compose up --build -d

# With testing profile (includes mock vLLM). Use testing config to point router at the mock endpoint:
# (CONFIG_FILE is read by the router entrypoint; the file is mounted from ./config)
CONFIG_FILE=/app/config/config.testing.yaml docker compose --profile testing up --build
```

**4. Verify**

- Semantic Router (gRPC): localhost:50051
- Envoy Proxy: http://localhost:8801
- Envoy Admin: http://localhost:19000

## Common Operations

```bash
# View service status
docker compose ps

# Follow logs for the router service
docker compose logs -f semantic-router

# Exec into the router container
docker compose exec semantic-router bash

# Stop and clean up containers
docker compose down
```

## Troubleshooting

**1. Router exits immediately with a Hugging Face DNS/download error**

Symptoms (from `docker compose logs -f semantic-router`):

```
Failed to initialize BERT: request error: https://huggingface.co/... Dns Failed: resolve dns name 'huggingface.co:443'
```

Why: `bert_model.model_id` in `config/config.yaml` points to a remote model (`sentence-transformers/all-MiniLM-L12-v2`). If the container cannot resolve or reach the internet, startup fails.

Fix options:

- Allow network access in the container (online):

  - Ensure your host can resolve DNS, or add DNS servers to the `semantic-router` service in `docker-compose.yml`:

    ```yaml
    services:
      semantic-router:
        # ...
        dns:
          - 1.1.1.1
          - 8.8.8.8
    ```

  - If behind a proxy, set `http_proxy/https_proxy/no_proxy` env vars for the service.

- Use a local copy of the model (offline):

  1. Download `sentence-transformers/all-MiniLM-L12-v2` to `./models/sentence-transformers/all-MiniLM-L12-v2/` on the host.
  2. Update `config/config.yaml` to use the local path (mounted into the container at `/app/models`):

      ```yaml
      bert_model:
        model_id: "models/sentence-transformers/all-MiniLM-L12-v2"
        threshold: 0.6
        use_cpu: true
      ```

  3. Recreate services: `docker compose up -d --build`

Extra tip: If you use the testing profile, also pass the testing config so the router targets the mock service:

```bash
CONFIG_FILE=/app/config/config.testing.yaml docker compose --profile testing up --build
```

**2. Envoy/Router up but requests fail**

- Ensure `mock-vllm` is healthy (testing profile only):
  - `docker compose ps` should show mock-vllm healthy; logs show 200 on `/health`.
- Verify the router config in use:
  - Router logs print `Starting vLLM Semantic Router ExtProc with config: ...`. If it shows `/app/config/config.yaml` while testing, you forgot `CONFIG_FILE`.
- Basic smoke test via Envoy (OpenAI-compatible):
  - Send a POST to `http://localhost:8801/v1/chat/completions` with `{"model":"auto", "messages":[{"role":"user","content":"hi"}]}` and check that the mock responds with `[mock-openai/gpt-oss-20b]` content when testing profile is active.

**3. DNS problems inside containers**

If DNS is flaky in your Docker environment, add DNS servers to the `semantic-router` service in `docker-compose.yml`:

```yaml
services:
  semantic-router:
    # ...
    dns:
      - 1.1.1.1
      - 8.8.8.8
```

For corporate proxies, set `http_proxy`, `https_proxy`, and `no_proxy` in the service `environment`.

Make sure 8801, 50051, 19000 are not bound by other processes. Adjust ports in `docker-compose.yml` if needed.
