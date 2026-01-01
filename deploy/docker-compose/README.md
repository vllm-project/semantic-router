# Main Runtime Compose Stack

This directory contains the primary `docker-compose.yml` used to run the Semantic Router stack end-to-end:

- Envoy proxy (ExtProc integration)
- Semantic Router (extproc)
- Observability (Prometheus + Grafana + Jaeger)
- Dashboard (unified UI: config, monitoring, topology, playground)
- Optional test services (mock-vllm, llm-katan via profiles)

## Path Layout

Because this file lives under `deploy/docker-compose/`, all relative paths to repository resources go two levels up (../../) back to repo root.

Example mappings:

- `../../config` -> mounts to `/app/config` inside containers
- `../../models` -> shared model files
- `../../tools/observability/...` -> Prometheus / Grafana provisioning assets
- `./addons/*` -> Compose-local Envoy/Grafana/Prometheus configs

## Services & Ports

- `envoy` (ports: 8801, 19000)
- `semantic-router` (port: 50051 for gRPC ExtProc; has internal health on 8080)
- `prometheus` (port: 9090)
- `grafana` (port: 3000)
- `jaeger` (ports: 4318, 16686)
- `dashboard` (port: 8700)
- `mock-vllm` (port: 8000; profile: testing)
- `llm-katan` (port: 8002; profiles: testing, llm-katan)

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
- Jaeger: http://localhost:16686 (tracing UI)
- Mock vLLM (testing profile): http://localhost:8000
- LLM Katan (testing/llm-katan profiles): http://localhost:8002

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

After the stack is healthy, open the Dashboard at http://localhost:8700.

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
- Playground: built-in chat playground calling the router API

Environment variables set in Compose:

- `TARGET_GRAFANA_URL=http://grafana:3000`
- `TARGET_PROMETHEUS_URL=http://prometheus:9090`
- `TARGET_JAEGER_URL=http://jaeger:16686`
- `TARGET_ROUTER_API_URL=http://semantic-router:8080`
- `TARGET_ROUTER_METRICS_URL=http://semantic-router:9190/metrics`
- `ROUTER_CONFIG_PATH=/app/config/config.yaml`

Volumes:

- `../../config:/app/config:rw` — allows the dashboard to read/write the config file

Image selection:

- Uses `DASHBOARD_IMAGE` if provided; otherwise builds from `dashboard/backend/Dockerfile` at `docker compose up` time.

## Observability Stack

The stack includes a complete observability solution:

### Prometheus

- **URL**: http://localhost:9090
- **Configuration**: `./addons/prometheus.yaml`
- **Data Retention**: 15 days
- **Storage**: Persistent volume `prometheus-data`

### Grafana

- **URL**: http://localhost:3000
- **Credentials**: admin/admin
- **Configuration**:
  - Datasources: Prometheus and Jaeger
  - Dashboard: LLM Router dashboard
  - Storage: Persistent volume `grafana-data`

### Jaeger (Distributed Tracing)

- **URL**: http://localhost:16686
- **OTLP Endpoint**: http://localhost:4318 (gRPC)
- **Configuration**: OTLP collector enabled
- **Integration**: Semantic Router sends traces via OTLP

## Networking

All services join the `semantic-network` bridge network with a fixed subnet to make in-network lookups stable. Host-published ports are listed above under Services & Ports.

## Troubleshooting

- Dashboard shows Grafana not configured: ensure Grafana is healthy and `TARGET_GRAFANA_URL` is correct
- Config update returns 500: verify `../../config` is mounted read/write and not a read-only ConfigMap/volume
- Envoy not ready: check `envoy` health at http://localhost:19000/ready and the config in `./addons/envoy.yaml`

## Related Stacks

- Local observability only: `tools/observability/docker-compose.obs.yml`
- Tracing stack: `tools/tracing/docker-compose.tracing.yaml`
