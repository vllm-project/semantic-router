# Main Runtime Compose Stack

This directory contains the primary `docker-compose.yml` used to run the semantic-router stack (router + envoy + optional mock-vllm + observability: Prometheus/Grafana + Jaeger tracing).

## Path Layout

Because this file lives under `deploy/docker-compose/`, all relative paths to repository resources go two levels up (../../) back to repo root.

Example mappings:

- `../../config` -> mounts to `/app/config` inside containers
- `../../models` -> shared model files
- `../../tools/observability/...` -> Prometheus / Grafana provisioning assets

## Profiles

- `testing` : enables `mock-vllm` and `llm-katan`
- `llm-katan` : only `llm-katan`

## Common Commands

```bash
# Bring up core stack
docker compose -f deploy/docker-compose/docker-compose.yml up --build

# With testing profile (adds mock-vllm & llm-katan)
docker compose -f deploy/docker-compose/docker-compose.yml --profile testing up --build

# Tear down
docker compose -f deploy/docker-compose/docker-compose.yml down
```

## Overrides

You can place a `docker-compose.override.yml` at repo root and combine:

```bash
docker compose -f deploy/docker-compose/docker-compose.yml -f docker-compose.override.yml up -d
```

## Related Stacks

- Local observability only: `tools/observability/docker-compose.obs.yml`
- Tracing stack (standalone, dev): `tools/tracing/docker-compose.tracing.yaml`

## Tracing & Grafana

- Jaeger UI: http://localhost:16686
- Grafana: http://localhost:3000 (admin/admin)
  - Prometheus datasource (default) for metrics
  - Jaeger datasource for exploring traces (search service `vllm-semantic-router`)

By default, the router container uses `config/config.tracing.yaml` (enabled tracing, exporter to Jaeger).
Override with `CONFIG_FILE=/app/config/config.yaml` if you don’t want tracing.
