# Semantic Router Dashboard

The Dashboard is an authenticated control and observability client for a
Semantic Router deployment. Its React frontend and Go backend provide a
product UI over Router Management, plus focused Dashboard workflows such as
Playground, evaluation, vLLM-SR Agent, and optional OpenClaw integration.

The ownership boundary is deliberate:

- Router Management owns Models, Recipes, Entrypoints, identity, access
  policy, quotas, API keys, usage, and request logs.
- `vllm-sr serve` owns runtime bootstrap and infrastructure.
- The Dashboard does not read, write, activate, deploy, or roll back Router
  YAML. It is also not an inference proxy; applications call the Router's
  public API directly.

## Local development

Run the frontend and backend from separate terminals:

```bash
make dashboard-dev-frontend
```

```bash
TARGET_ROUTER_API_URL=http://127.0.0.1:8080 \
TARGET_ENVOY_URL=http://127.0.0.1:8899 \
make dashboard-dev-backend
```

Open `http://127.0.0.1:3001`. Vite proxies Dashboard API requests to port
`8700`. Start the Router data plane separately with `vllm-sr serve`.

## Build and test

```bash
make dashboard-check
```

The gate runs frontend lint and type checking, frontend unit tests, backend
lint and tests, and the Dashboard Go module tidy check. The backend is a
separate Go module; root-level `go test ./...` does not cover it.

## Runtime connections

The backend accepts matching command-line flags for these environment
variables. Defaults live in [`backend/config/config.go`](backend/config/config.go).

| Variable | Purpose |
| --- | --- |
| `DASHBOARD_PORT` | Dashboard HTTP port. |
| `DASHBOARD_STATIC_DIR` | Built frontend assets. |
| `TARGET_ROUTER_API_URL` | Router Management origin used by the backend proxy. |
| `DASHBOARD_ROUTER_PUBLIC_URL` | Browser-reachable Router inference origin. |
| `TARGET_ENVOY_URL` | Inference origin used by trusted Dashboard workflows. |
| `TARGET_ROUTER_METRICS_URL` | Router metrics endpoint. |
| `TARGET_GRAFANA_URL` | Optional Grafana origin. |
| `TARGET_PROMETHEUS_URL` | Optional Prometheus origin. |
| `TARGET_JAEGER_URL` | Optional Jaeger origin. |
| `TARGET_FLEET_SIM_URL` | Optional Fleet Simulator origin. |

Feature controls include `DASHBOARD_READONLY`, `EVALUATION_ENABLED`,
`ML_PIPELINE_ENABLED`, and `OPENCLAW_ENABLED`. Router resource
mutations still require the exact Router Management capability such as
`routing.manage`; Dashboard roles are not a substitute for Router
authorization.

## Authentication

Provision the first administrator with `DASHBOARD_ADMIN_EMAIL`,
`DASHBOARD_ADMIN_PASSWORD`, and optionally `DASHBOARD_ADMIN_NAME`. Public
first-admin registration is disabled by default and can be enabled only with
`DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true` in a controlled environment.

Configure the Dashboard's Router identity exchange with the
`DASHBOARD_ISSUER_*` and `DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE` settings.
Router Management validates the resulting principal, namespace, and
capabilities on every management request.

Persistent Dashboard-owned state includes authentication, evaluation, and
optional workflow databases. Mount their configured paths on durable storage
when those workflows must survive restarts.

## Architecture

```text
Browser
  -> React SPA
  -> Go Dashboard API
       -> Router Management API
       -> Router public inference API
       -> optional monitoring and workflow services
       -> Dashboard-owned authentication and workflow stores
```

- [`frontend/src/app/`](frontend/src/app/) owns routing and authentication
  gates.
- [`frontend/src/pages/`](frontend/src/pages/) owns page orchestration.
- [`backend/router/`](backend/router/) registers Dashboard and proxy routes.
- [`backend/handlers/`](backend/handlers/) implements Dashboard-owned
  workflows.

Keep user-facing workflows in the website documentation and keep this file
focused on Dashboard development and deployment boundaries.
