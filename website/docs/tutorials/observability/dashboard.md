# Semantic Router Dashboard

The Semantic Router Dashboard combines a landing/setup flow with an authenticated operator control plane for the router. It now spans configuration lifecycle, live testing, observability, debugging, evaluation, and agent operations across local development, Docker Compose, and Kubernetes deployments.

- Landing, login, and setup surfaces for first-run bootstrap
- Unified operator workspace for dashboard, monitoring, tracing, status, logs, replay, config, playground, and topology
- Evaluation, ratings, builder, and ML model-selection workflows
- OpenClaw and MCP surfaces for agent workspaces, servers, and tool execution
- Single backend proxy that normalizes auth, CORS, and CSP across Grafana, Prometheus, Jaeger, and router APIs

## What’s inside

### Frontend (React + TypeScript + Vite)

A modern SPA with:

- React 18 + TypeScript + Vite
- React Router for client-side routing
- CSS Modules, dark/light theme with persistence
- Collapsible sidebar to jump across sections
- Topology visualization powered by React Flow

Primary surfaces:

- Entry flow: `/`, `/login`, `/auth/transition`, `/setup`
- Operator home: `/dashboard`, `/monitoring`
- Configuration and testing: `/config`, `/config/:section`, `/playground`, `/playground/fullscreen`, `/topology`
- Debugging and runtime insight: `/tracing`, `/status`, `/logs`, `/replay`
- Quality and optimization: `/evaluation`, `/ml-setup`, `/ratings`, `/builder`
- Agent and admin surfaces: `/clawos`, `/users`

### Backend (Go HTTP Server)

- Serves the frontend build and auth-protected SPA routes
- Reverse proxies Grafana, Prometheus, Jaeger, router APIs, and selected service APIs
- Exposes setup, config lifecycle, tooling, evaluation, ML pipeline, MCP, and OpenClaw APIs

Key routes:

- Health and setup: `/healthz`, `/api/settings`, `/api/setup/state`, `/api/setup/import-remote`, `/api/setup/validate`, `/api/setup/activate`
- Auth and admin: `/api/auth/*`, `/api/admin/users`, `/api/admin/permissions`, `/api/admin/audit-logs`
- Config lifecycle: `/api/router/config/all`, `/api/router/config/yaml`, `/api/router/config/update`, `/api/router/config/deploy/preview`, `/api/router/config/deploy`, `/api/router/config/rollback`, `/api/router/config/versions`, `/api/router/config/defaults`, `/api/router/config/defaults/update`
- Tooling and operator APIs: `/api/tools-db`, `/api/tools/web-search`, `/api/tools/open-web`, `/api/tools/fetch-raw`, `/api/status`, `/api/logs`, `/api/topology/test-query`
- Evaluation and ML workflows: `/api/evaluation/*`, `/api/ml-pipeline/*`
- Agent integrations: `/api/mcp/*`, `/api/openclaw/*`, `/embedded/grafana/*`, `/embedded/prometheus/*`, `/embedded/jaeger*`, `/metrics/router`

The proxy strips/overrides `X-Frame-Options` and adjusts `Content-Security-Policy` to allow `frame-ancestors 'self'`, enabling safe embedding under the dashboard origin.

## Environment variables

Common environment variables:

- `DASHBOARD_PORT` (8700)
- `TARGET_GRAFANA_URL`
- `TARGET_PROMETHEUS_URL`
- `TARGET_JAEGER_URL`
- `TARGET_ROUTER_API_URL` (http://localhost:8080)
- `TARGET_ROUTER_METRICS_URL` (http://localhost:9190/metrics)
- `TARGET_ENVOY_URL` (required for Playground chat via Envoy)
- `ROUTER_CONFIG_PATH` (../../config/config.yaml)
- `DASHBOARD_STATIC_DIR` (../frontend)
- `DASHBOARD_READONLY`
- `DASHBOARD_SETUP_MODE`
- `ML_SERVICE_URL`
- `DASHBOARD_AUTH_DB_PATH`, `DASHBOARD_JWT_SECRET`, `DASHBOARD_ADMIN_EMAIL`, `DASHBOARD_ADMIN_PASSWORD`, `DASHBOARD_ADMIN_NAME`
- `OPENCLAW_ENABLED`, `OPENCLAW_URL`, `OPENCLAW_DATA_DIR`, `OPENCLAW_TOKEN`

Note: The config update API writes to `ROUTER_CONFIG_PATH`. In containers/Kubernetes, this path must be writable (not a read-only ConfigMap). Mount a writable volume if you need runtime edits to persist.

## Quick start

### Docker Compose (recommended)

The dashboard is integrated into the main Compose file.

```bash
# From the repository root
make docker-compose-up
```

Then open in browser:

- Dashboard: http://localhost:8700
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## Related docs

- [Installation Configuration](../../installation/configuration.md)
- [Observability Metrics](./metrics.md)
- [Distributed Tracing](./distributed-tracing.md)
