# Main Runtime Compose Stack

This directory contains the primary `docker-compose.yml` used to run the Semantic Router stack end-to-end:

- Envoy proxy (ExtProc integration)
- Semantic Router (extproc)
- Observability (Prometheus + Grafana)
- Dashboard (unified UI: config, monitoring, topology, playground)
- Open WebUI + Pipelines (for the Playground tab)
- Optional test services (mock-vllm, llm-katan via profiles)

## Path Layout

Because this file lives under `deploy/docker-compose/`, all relative paths to repository resources go two levels up (../../) back to repo root.

Example mappings:

- `../../config` -> mounts to `/app/config` inside containers
- `../../models` -> shared model files
- `../../tools/observability/...` -> Prometheus / Grafana provisioning assets
- `./addons/*` -> Compose-local Envoy/Grafana/Prometheus configs and Open WebUI pipeline

## Services & Ports

- `envoy` (ports: 8801, 19000)
- `semantic-router` (port: 50051 for gRPC ExtProc; has internal health on 8080)
- `prometheus` (port: 9090)
- `grafana` (port: 3000)
- `openwebui` (port: 3001 → 8080 in-container)
- `pipelines` (no host port by default)
- `dashboard` (port: 8700)
- `mock-vllm` (port: 8000; profile: testing)
- `llm-katan` (port: 8002 → 8000; profiles: testing, llm-katan)

## Profiles

- `testing` : enables `mock-vllm` and `llm-katan`
- `llm-katan` : enables only `llm-katan`

## Quick Start

Preferred way is via Makefile wrappers (they set COMPOSE_FILE and project name for you):

```bash
# Start core stack (add REBUILD=1 to force image rebuild)
make docker-compose-up

# Start with testing profile
make docker-compose-up-testing

# Start with llm-katan profile
make docker-compose-up-llm-katan

# Rebuild and start (core stack)
make docker-compose-rebuild

# Stop all services
make docker-compose-down
```

Equivalent raw docker compose commands:

```bash
# Core stack
docker compose -f deploy/docker-compose/docker-compose.yml up -d --build

# With testing profile
docker compose -f deploy/docker-compose/docker-compose.yml --profile testing up -d --build

# Tear down
docker compose -f deploy/docker-compose/docker-compose.yml down
```

## Overrides

You can place a `docker-compose.override.yml` at repo root and combine:

```bash
docker compose -f deploy/docker-compose/docker-compose.yml -f docker-compose.override.yml up -d
```

Typical overrides include:

- Changing published ports
- Switching images to local builds
- Overriding environment variables for services

## Dashboard Integration

The `dashboard` service exposes a unified UI at http://localhost:8700 with:

- Monitoring: iframe embed of Grafana
- Config: `GET /api/router/config/all` and `POST /api/router/config/update` mapped to `/app/config/config.yaml`
- Topology: visualizes routing/config
- Playground: iframe embed of Open WebUI

Environment variables set in Compose:

- `TARGET_GRAFANA_URL=http://grafana:3000`
- `TARGET_PROMETHEUS_URL=http://prometheus:9090`
- `TARGET_ROUTER_API_URL=http://semantic-router:8080`
- `TARGET_ROUTER_METRICS_URL=http://semantic-router:9190/metrics`
- `TARGET_OPENWEBUI_URL=http://openwebui:8080`
- `ROUTER_CONFIG_PATH=/app/config/config.yaml`

Volumes:

- `../../config:/app/config:rw` — allows the dashboard to read/write the config file

Image selection:

- Uses `DASHBOARD_IMAGE` if provided; otherwise builds from `dashboard/backend/Dockerfile` at `docker compose up` time.

## Open WebUI + Pipelines

- `openwebui` is exposed at http://localhost:3001 (proxied via the Dashboard too)
- `pipelines` mounts `./addons/vllm_semantic_router_pipe.py` into `/app/pipelines/` for easy integration

## Networking

All services join the `semantic-network` bridge network with a fixed subnet to make in-network lookups stable. Host-published ports are listed above under Services & Ports.

## Troubleshooting

- Dashboard shows Grafana not configured: ensure Grafana is healthy and `TARGET_GRAFANA_URL` is correct
- Config update returns 500: verify `../../config` is mounted read/write and not a read-only ConfigMap/volume
- Envoy not ready: check `envoy` health at http://localhost:19000/ready and the config in `./addons/envoy.yaml`

## Related Stacks

- Local observability only: `tools/observability/docker-compose.obs.yml`
- Tracing stack: `tools/tracing/docker-compose.tracing.yaml`

## Related Stacks

- Local observability only: `tools/observability/docker-compose.obs.yml`
- Tracing stack: `tools/tracing/docker-compose.tracing.yaml`
