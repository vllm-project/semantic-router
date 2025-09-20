# Install with Docker Compose

Run Semantic Router + Envoy locally using Docker Compose v2.

## Prerequisites

- Docker Engine and Docker Compose v2 (use the `docker compose` command, not the legacy `docker-compose`)

   ```bash
   # Verify
   docker compose version
   ```

   Install Docker Compose v2 for Ubuntu(if missing), see more in [Docker Compose Plugin Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

   ```bash
   # Remove legacy v1 if present (optional)
   sudo apt-get remove -y docker-compose || true

   sudo apt-get update
   sudo apt-get install -y ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --yes --dearmor -o /etc/apt/keyrings/docker.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

   docker compose version
   ```

- Ensure ports 8801, 50051, 19000 are free

## Install and Run with Docker Compose v2

1) Clone the repo and move into it (from your workspace root):

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
```

2) Download required models (classification models):

```bash
make download-models
```

This downloads the classification models used by the router:

- Category classifier (ModernBERT-base)
- PII classifier (ModernBERT-base)
- Jailbreak classifier (ModernBERT-base)

Note: The BERT similarity model defaults to a remote Hugging Face model. See Troubleshooting for offline/local usage.

3) Start the services with Docker Compose v2:

```bash
# Start core services (semantic-router + envoy)
docker compose up --build

# Or run in background (recommended)
docker compose up --build -d

# With testing profile (includes mock vLLM)
docker compose --profile testing up --build
```

4) Verify

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

### 1) Router exits immediately with a Hugging Face DNS/download error

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

### 2) Port already in use

Make sure 8801, 50051, 19000 are not bound by other processes. Adjust ports in `docker-compose.yml` if needed.
