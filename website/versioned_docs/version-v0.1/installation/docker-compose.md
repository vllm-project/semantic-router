---
sidebar_position: 3
---

# Install with Docker Compose

This guide provides step-by-step instructions for deploying the vLLM Semantic Router with Envoy AI Gateway on Docker Compose.

## Common Prerequisites

- **Docker Engine:** see more in [Docker Engine Installation](https://docs.docker.com/engine/install/)

- **Clone repo:**

  ```bash
  git clone https://github.com/vllm-project/semantic-router.git
  cd semantic-router
  ```

- **Download classification models (≈1.5GB, first run only):**

  ```bash
  # Tips: If you encounter this error 'hf: command not found', run 'pip install huggingface_hub hf_transfer'.
  make download-models
  ```

  This downloads the classification models used by the router:

  - Category classifier (ModernBERT-base)
  - PII classifier (ModernBERT-base)
  - Jailbreak classifier (ModernBERT-base)

---

### Requirements

- Docker Compose v2 (`docker compose` command, not the legacy `docker-compose`)

  Install Docker Compose Plugin (if missing), see more in [Docker Compose Plugin Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

  ```bash
  # For Debian / Ubuntu
  sudo apt-get update
  sudo apt-get install -y docker-compose-plugin

  # For RHEL / CentOS / Fedora
  sudo yum update -y
  sudo yum install -y docker-compose-plugin

  # Verify
  docker compose version
  ```

- Ensure ports 8801, 50051, 19000, 3000 and 9090 are free

### Start Services

```bash
# Core (router + envoy)
docker compose -f deploy/docker-compose/docker-compose.yml up --build

# Detached (recommended once OK)
docker compose -f deploy/docker-compose/docker-compose.yml up -d --build

# Include mock vLLM + testing profile (points router to mock endpoint)
CONFIG_FILE=/app/config/testing/config.testing.yaml \
  docker compose -f deploy/docker-compose/docker-compose.yml --profile testing up --build
```

### Verify

- gRPC: `localhost:50051`
- Envoy HTTP: `http://localhost:8801`
- Envoy Admin: `http://localhost:19000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin` for first login)

### Common Operations

```bash
# View service status
docker compose ps

# Follow logs for the router service
docker compose logs -f semantic-router

# Exec into the router container
docker compose exec semantic-router bash

# Recreate after config change
docker compose -f deploy/docker-compose/docker-compose.yml up -d --build

# Stop and clean up containers
docker compose down
```
