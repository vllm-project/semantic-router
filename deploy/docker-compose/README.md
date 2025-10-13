# Main Runtime Compose Stack

This directory contains the primary `docker-compose.yml` used to run the semantic-router stack (router + envoy + optional mock-vllm + observability).

## Path Layout
Because this file lives under `deploy/docker-compose/`, all relative paths to repository resources go two levels up (../../) back to repo root.

Example mappings:

- `../../config` -> mounts to `/app/config` inside containers
- `../../models` -> shared model files
- `../../tools/observability/...` -> Prometheus / Grafana provisioning assets

## Profiles

- `testing` : enables `mock-vllm` and `llm-katan`
- `llm-katan` : only `llm-katan`

## Services and Ports

These host ports are exposed when you bring the stack up:

- Dashboard: http://localhost:8700 (Semantic Router Dashboard)
- Envoy proxy: http://localhost:8801
- Envoy admin: http://localhost:19000
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Open WebUI: http://localhost:3001
- Mock vLLM (testing profile): http://localhost:8000
- LLM Katan (testing/llm-katan profiles): http://localhost:8002

## Common Commands

```bash
# Bring up core stack
docker compose -f deploy/docker-compose/docker-compose.yml up --build

# With testing profile (adds mock-vllm & llm-katan)
docker compose -f deploy/docker-compose/docker-compose.yml --profile testing up --build

# Tear down
docker compose -f deploy/docker-compose/docker-compose.yml down
```

After the stack is healthy, open the Dashboard at http://localhost:8700.

## Overrides
You can place a `docker-compose.override.yml` at repo root and combine:

```bash
docker compose -f deploy/docker-compose/docker-compose.yml -f docker-compose.override.yml up -d
```

## Related Stacks

- Local observability only: `tools/observability/docker-compose.obs.yml`
- Tracing stack: `tools/tracing/docker-compose.tracing.yaml`

## Dashboard

The Dashboard service provides a single entry point to explore and manage the stack:

- URL: http://localhost:8700
- Health check: http://localhost:8700/healthz
- Features:
  - Router status and config visibility (reads `config/config.yaml` mounted into the container)
  - Embedded Grafana dashboard (live metrics from Prometheus)
  - Links to Envoy, Prometheus, and Open WebUI Playground

### How it is wired

From `docker-compose.yml`:

- Service name: `dashboard`
- Image: `ghcr.io/vllm-project/semantic-router/dashboard:latest` (override with `DASHBOARD_IMAGE`)
- Build fallback: builds from `dashboard/backend/Dockerfile` at repo root
- Port: `8700:8700`
- Volumes: mounts `../../config` as `/app/config` (read-write) so the Dashboard can read your active `config.yaml`
- Depends on: `semantic-router`, `grafana`, `prometheus`, `openwebui`, `pipelines`

Environment variables passed to Dashboard:

- `DASHBOARD_PORT=8700`
- `TARGET_GRAFANA_URL=http://grafana:3000`
- `TARGET_PROMETHEUS_URL=http://prometheus:9090`
- `TARGET_ROUTER_API_URL=http://semantic-router:8080`
- `TARGET_ROUTER_METRICS_URL=http://semantic-router:9190/metrics`
- `TARGET_OPENWEBUI_URL=http://openwebui:8080`
- `ROUTER_CONFIG_PATH=/app/config/config.yaml`

Note: The Router’s HTTP API (8080) and metrics (9190) are only available inside the Docker network (not published to host). The Dashboard connects to them internally so you don’t need to expose extra ports.

### Credentials and defaults

- Grafana is provisioned with `admin/admin` (see `GF_SECURITY_ADMIN_*` env in compose). For convenient embedding, anonymous viewing is enabled and suitable for local development only.
- Open WebUI is published at http://localhost:3001 and defaults to routing OpenAI-compatible traffic to the Pipelines service.

### Customizing

- Use `DASHBOARD_IMAGE` to point at a different prebuilt Dashboard image if desired:
  - Example (bash): `export DASHBOARD_IMAGE=ghcr.io/your-org/semantic-router-dashboard:tag`
- Edit `config/config.yaml` to change Router behavior; the Dashboard reads that file from the mounted `config/` directory.
- If a port conflicts on your machine, change the left-hand port mapping in `docker-compose.yml` (e.g., `"3000:3000"` -> `"33000:3000"`).

### Troubleshooting

- Dashboard not loading charts: ensure Prometheus and Grafana containers are up and healthy.
- Grafana asks for login: use `admin/admin` or enable anonymous as already configured in compose; for production, disable anonymous and set strong credentials.
- Health checks failing initially can be normal while services warm up; Compose has `start_period` set to lessen flakiness.
