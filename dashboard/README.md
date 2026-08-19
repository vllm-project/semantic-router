# Semantic Router Dashboard

The Dashboard is the authenticated control and observability UI for a Semantic
Router deployment. It combines a React frontend with a Go backend that serves
the SPA, stores dashboard state, and proxies Router, Envoy, Grafana,
Prometheus, Jaeger, and Fleet Simulator endpoints.

Use it to:

- complete first-run model and recipe setup;
- inspect, edit, validate, deploy, and roll back Router configuration;
- import and activate Recipe packages;
- test routes in the Playground and inspect the selected path;
- view topology, logs, evaluations, and monitoring tools;
- manage security policies, ML selection workflows, MCP tools, and optional
  OpenClaw workers when those features are enabled.

The Dashboard is a control plane, not an inference proxy. Applications should
send inference requests to Envoy.

## Local development

Start the frontend and backend in separate terminals from the repository root:

```bash
make dashboard-dev-frontend
```

```bash
ROUTER_CONFIG_PATH="$PWD/config/config.yaml" \
TARGET_ROUTER_API_URL=http://127.0.0.1:8080 \
TARGET_ENVOY_URL=http://127.0.0.1:8899 \
make dashboard-dev-backend
```

Open `http://127.0.0.1:3001`. Vite proxies backend requests to port `8700`.
The Router and Envoy must be running for live config, status, and Playground
operations.

For the complete local stack, use the CLI instead:

```bash
vllm-sr serve --config config/config.yaml
vllm-sr dashboard
```

The current installation and first-run workflow is documented in
[`website/docs/installation/installation.md`](../website/docs/installation/installation.md).

## Build and test

```bash
make dashboard-build
make dashboard-check
make dashboard-test-backend
```

`dashboard-check` is the single entrypoint for dashboard quality, and the required
`Dashboard` CI workflow runs the **same target** — nothing here is CI-only, and
nothing in CI is missing locally. Run it before pushing. It runs, in order:

| Step | What it covers |
| --- | --- |
| `dashboard-lint` | ESLint on the frontend, golangci-lint on the backend |
| `dashboard-type-check` | TypeScript type checking (frontend + Knowledge Map) |
| `dashboard-test-frontend` | Frontend unit tests |
| `dashboard-test-backend` | `go test ./...` on `dashboard/backend` |
| `dashboard-go-mod-tidy` | Verifies `go.mod` / `go.sum` are tidy |

The dashboard backend is a **separate Go module**, so `go test ./...` from the
repository root does not cover it. Use `make dashboard-test-backend`, or run
`go test ./...` from `dashboard/backend` directly.

Some backend tests shell out to the `vllm-sr` CLI (for example to regenerate Envoy
config), so they need its Python dependencies importable:

```bash
pip install -e src/vllm-sr
```

Without it, those tests fail with `ModuleNotFoundError`. CI installs the same package.

`dashboard-check` runs plain `go test`. The race detector roughly doubles the
runtime, which is a poor trade on every PR, so it is deliberately **not** in the
always-on gate. Run it locally before pushing concurrency-sensitive work — anything
touching shared state, goroutines, caches, or resolvers:

```bash
cd dashboard/backend && go test ./... -race
```

## Runtime configuration

The backend accepts matching command-line flags for these environment
variables. Defaults are defined in
[`backend/config/config.go`](backend/config/config.go).

| Variable | Purpose |
| --- | --- |
| `DASHBOARD_PORT` | Backend listen port; default `8700`. |
| `DASHBOARD_STATIC_DIR` | Built frontend assets. |
| `ROUTER_CONFIG_PATH` | Canonical Router YAML read or updated by config APIs. |
| `DASHBOARD_CONFIG_DIR` | Directory for config versions and related state. |
| `TARGET_ROUTER_API_URL` | Router management API; default `http://localhost:8080`. |
| `TARGET_ROUTER_METRICS_URL` | Router Prometheus endpoint. |
| `TARGET_ENVOY_URL` | Inference endpoint used by Playground and route probes. |
| `TARGET_GRAFANA_URL` | Optional Grafana base URL. |
| `TARGET_PROMETHEUS_URL` | Optional Prometheus base URL. |
| `TARGET_JAEGER_URL` | Optional Jaeger base URL. |
| `TARGET_FLEET_SIM_URL` | Optional Fleet Simulator service URL. |

Feature controls:

| Variable | Purpose |
| --- | --- |
| `DASHBOARD_READONLY` | Hard-disable all config mutation. |
| `DASHBOARD_RUNTIME_CONFIG_WRITABLE` | Allow mutation of the mounted runtime config surface. |
| `DASHBOARD_RECIPE_STORE_WRITABLE` | Allow Recipe package import. |
| `DASHBOARD_SETUP_MODE` | Enable the trusted first-run setup flow. |
| `EVALUATION_ENABLED` | Enable evaluation jobs. |
| `ML_PIPELINE_ENABLED` | Enable benchmark, training, and config-generation jobs. |
| `ML_TRAINING_DIR` | Training script directory for subprocess mode. |
| `ML_SERVICE_URL` | Use an external ML service instead of local subprocesses. |
| `MCP_ENABLED` | Enable MCP server and tool management. |
| `OPENCLAW_ENABLED` | Enable OpenClaw provisioning and room workflows. |

Persistent SQLite paths include `DASHBOARD_AUTH_DB_PATH`,
`EVALUATION_DB_PATH`, `DASHBOARD_WORKFLOW_DB_PATH`, and
`DASHBOARD_CONFIG_PROJECTION_DB_PATH`. Mount writable persistent storage for
any state that must survive a container restart.

## Authentication and write safety

Set a stable `DASHBOARD_JWT_SECRET` and provision the first administrator with
`DASHBOARD_ADMIN_EMAIL`, `DASHBOARD_ADMIN_PASSWORD`, and optionally
`DASHBOARD_ADMIN_NAME`. Public web-form bootstrap is disabled by default; only
set `DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true` in a controlled first-run environment.

Read-only mode and the two writable-surface flags are independent. A read-only
ConfigMap, GitOps-owned config, or read-only Recipe store should be reflected in
the matching flag so the UI does not offer operations the runtime cannot
persist.

Some local workflows can manage containers. Do not mount a container-runtime
socket unless users with Dashboard access are allowed to control that runtime.
See the [security hardening guide](../website/docs/installation/security-hardening.md)
for the deployment boundary.

## Architecture

```text
Browser
  -> React SPA (dashboard/frontend)
  -> Go API and reverse proxy (dashboard/backend)
       -> Router management API
       -> Envoy inference listener
       -> optional monitoring and simulator services
       -> local SQLite and config/Recipe storage
```

- [`frontend/src/app/`](frontend/src/app/) owns routing, authentication gates,
  and the application shell.
- [`frontend/src/pages/`](frontend/src/pages/) owns page orchestration.
- [`backend/router/`](backend/router/) registers public and authenticated API
  routes.
- [`backend/handlers/`](backend/handlers/) implements control-plane workflows.
- [`backend/recipe/`](backend/recipe/) validates and materializes Recipe
  packages.
- [`wizmap/`](wizmap/) builds the embedded knowledge-map view.

Keep detailed user workflows in the website and keep this README focused on
developing and operating the Dashboard itself.
