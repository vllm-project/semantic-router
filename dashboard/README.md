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

`dashboard-check` runs frontend lint, type checking, unit tests, and the backend
module-tidy check. `dashboard-test-backend` runs the Go test suite.

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

## Environment-agnostic configuration

The backend exposes a single port (default 8700) and proxies to targets defined via environment variables. This keeps frontend URLs stable and avoids CORS by same-origining everything under the dashboard host.

Required env vars (with sensible defaults per environment):

- `DASHBOARD_PORT` (default: 8700)
- `TARGET_GRAFANA_URL`
- `TARGET_PROMETHEUS_URL`
- `TARGET_ROUTER_API_URL` (router `:8080`)
- `TARGET_ROUTER_METRICS_URL` (router `:9190/metrics`)
- `TARGET_ENVOY_URL` — Envoy proxy URL for chat completions (e.g., `http://envoy:8801`). Required for Playground chat to work.

Optional:

- `ROUTER_CONFIG_PATH` (default: `../../config/config.yaml`) — path to the router config file used by the config APIs and Tools DB.
- `DASHBOARD_STATIC_DIR` — override static assets directory (defaults to `../frontend`).
- `ML_SERVICE_URL` — URL of the Python ML service sidecar for HTTP mode (e.g., `http://ml-service:8686`). If not set, the dashboard uses subprocess mode (runs Python scripts directly).
- `ML_PIPELINE_ENABLED` — set to `true` to enable ML pipeline features in Docker Compose/K8s deployments.
  Note: The backend already adjusts frame-busting headers (X-Frame-Options/CSP) to allow embedding from the dashboard origin; no extra env flag is required.

Recommended upstream settings for embedding:

- Grafana: set `GF_SECURITY_ALLOW_EMBEDDING=true` and prefer `access: proxy` datasource (already configured)

## URL strategy (stable, user-facing)

- Dashboard Home (Landing): `http://<host>:8700/`
- Monitoring tab: iframe `src="/embedded/grafana/d/<dashboard-uid>?kiosk&theme=light"`
- Config tab: frontend fetch `GET /api/router/config/all` (demo edit modals; see note above)
- Topology tab: client fetch of `GET /api/router/config/all` to render the flow graph
- Playground tab: built-in chat UI calling the router API (`POST /api/router/v1/chat/completions`)

## Deployment matrix

1. Local dev (router and observability on host)

- Use `tools/observability/docker-compose.obs.yml` to start Prometheus (9090) and Grafana (3000) on host network
- Start dashboard backend locally (port 8700)
- Env examples:
  - `TARGET_GRAFANA_URL=http://localhost:3000`
  - `TARGET_PROMETHEUS_URL=http://localhost:9090`
  - `TARGET_ROUTER_API_URL=http://localhost:8080`
  - `TARGET_ROUTER_METRICS_URL=http://localhost:9190/metrics`

2. Docker Compose (all-in-one)

- Reuse services defined in `deploy/docker-compose/docker-compose.yml` (Dashboard included by default)
- Env examples (inside compose network):
  - `TARGET_GRAFANA_URL=http://grafana:3000`
  - `TARGET_PROMETHEUS_URL=http://prometheus:9090`
  - `TARGET_ROUTER_API_URL=http://semantic-router:8080`
  - `TARGET_ROUTER_METRICS_URL=http://semantic-router:9190/metrics`

3. Kubernetes

- Install/confirm Prometheus and Grafana via existing manifests in `deploy/kubernetes/observability` (repository root)
- Deploy the dashboard via manifests under the repository-level `deploy/kubernetes/` (or create one similar to the Compose setup)
- Configure the dashboard Deployment with in-cluster URLs:
  - `TARGET_GRAFANA_URL=http://grafana.<ns>.svc.cluster.local:3000`
  - `TARGET_PROMETHEUS_URL=http://prometheus.<ns>.svc.cluster.local:9090`
  - `TARGET_ROUTER_API_URL=http://semantic-router.<ns>.svc.cluster.local:8080`
  - `TARGET_ROUTER_METRICS_URL=http://semantic-router.<ns>.svc.cluster.local:9190/metrics`
- Expose the dashboard via Ingress/Gateway to the outside; upstreams remain ClusterIP

## Security & access control

- Dashboard auth uses JWTs from `Authorization: Bearer <token>` for protected `/api/*` and `/embedded/*` requests.
- Protected embedded entry URLs may also carry `authToken=<token>`. Login and bootstrap responses set an HttpOnly `vsr_session` cookie, and logout revokes newly issued server-side session ids before clearing that cookie.
- Frame embedding: backend strips/overrides `X-Frame-Options` and `Content-Security-Policy` headers from upstreams to permit `frame-ancestors 'self'` only.
- **Security Policy page** (`/security`, accessible via Manager dropdown): allows admins to define role-to-model RBAC mappings and per-role rate-limit tiers. On save, the dashboard translates these into canonical router config (`routing.signals.role_bindings`, `routing.decisions`, and `global.services.ratelimit`), merges them into the running `config.yaml`, and triggers a hot-reload so the router enforces the new policy immediately. Requires the `security.manage` permission for writes; `config.read` is sufficient for viewing. See [security-hardening.md](../website/docs/installation/security-hardening.md) for full details.
- **Dashboard RBAC permissions**: `feedback.submit`, `replay.read`, and `security.manage` extend the built-in role/permission matrix. Only admin-role users receive `security.manage` by default.
- Auth users, roles, permissions, audit logs, workflow state, and session ids use SQLite under `./data` by default. In containers or Kubernetes, mount `/app/data` or set `DASHBOARD_AUTH_DB_PATH` and `DASHBOARD_WORKFLOW_DB_PATH` to persistent paths if you need state to survive restarts.
- The current SQLite auth/session store is single-replica local state. Run one dashboard replica unless you add a shared production auth/session store.
- Future: OIDC login on dashboard and signed proxy sessions to embedded services.

## Runtime status and version reporting

- `/api/status` is the dashboard's live runtime summary endpoint. It is protected by dashboard auth and requires the logs/observability read permission.
- The status response reports the dashboard backend version in tag form, such as `v0.3.0`, `v0.3.0-dev.<sha>`, or `v0.3.0-nightly.<date>.<sha>`.
- Version values are injected into release dashboard images from the pushed `v<version>` tag. Non-release dashboard images derive their version from `src/vllm-sr/pyproject.toml` plus CI context. Local source runs fall back to `src/vllm-sr/pyproject.toml` plus Go VCS metadata when available.
- When the dashboard backend is running but Router or Envoy is not reachable, `/api/status` still reports the Dashboard service as `running` and marks Router as not running instead of returning an empty `0/0` service list.

Write access warning for config updates:

- The `POST /api/router/config/update` endpoint writes to the mounted config path. In Docker/K8s this may be read-only if sourced from a ConfigMap. Use a writable volume, bind-mount, or external configuration service if you need runtime persistence.

## Extensibility

- New panels: add tabs/components to `frontend/`
- New integrations: add target env vars and a new `/embedded/<service>` route in backend proxy
- Topology: customize nodes/edges in `TopologyPage.tsx` (React Flow)
- Metrics aggregation: add `/api/metrics` in backend to produce derived KPIs from Prometheus

## Implementation notes

— Backend: Go server with reverse proxies for `/embedded/*` and `/api/router/*`, plus `/api/router/config/all`
— Frontend: SPA with embedded observability + built-in chat playground + structured config viewer
— K8s manifests: Deployment + Service + ConfigMap; optional Ingress (add per cluster)
— Future: OIDC, per-route RBAC, metrics summary endpoint

## Quick Start

### Method 1: Start with Docker Compose (Recommended)

The Dashboard is integrated into the main Compose stack, requiring no extra configuration:

```bash
# From the project root directory
docker compose -f deploy/docker-compose/docker-compose.yml up -d --build
```

After startup, access:

- **Dashboard**: http://localhost:8700
- **Grafana** (direct access): http://localhost:3000 (admin/admin)
- **Prometheus** (direct access): http://localhost:9090

### Method 2: Local Development Mode

When developing the Dashboard code locally:

```bash
# 1) Start Observability locally (Prometheus + Grafana on host network)
docker compose -f tools/observability/docker-compose.obs.yml up -d

# 2) Install frontend dependencies and run Vite dev server
cd dashboard/frontend
npm install
npm run dev
# Vite runs at http://localhost:3001 and proxies /api, /embedded and /healthz to http://localhost:8700

# 3) Start the Dashboard backend in another terminal
cd dashboard/backend
export TARGET_GRAFANA_URL=http://localhost:3000
export TARGET_PROMETHEUS_URL=http://localhost:9090
export TARGET_ROUTER_API_URL=http://localhost:8080
export TARGET_ROUTER_METRICS_URL=http://localhost:9190/metrics
export ROUTER_CONFIG_PATH=../../config/config.yaml
go run main.go -port=8700 -static=../frontend/dist -config=$ROUTER_CONFIG_PATH

# Tip: If your router runs inside Docker Compose, point TARGET_* to the container hostnames instead.
```

### Method 3: Rebuild Dashboard Only

For a quick rebuild after code changes:

```bash
# Rebuild the dashboard service
docker compose -f deploy/docker-compose/docker-compose.yml build dashboard

# Restart the dashboard
docker compose -f deploy/docker-compose/docker-compose.yml up -d dashboard

# View logs
docker logs -f semantic-router-dashboard
```

## Testing and CI checks

### The fast gate: `make dashboard-check`

Run this before pushing. It is the single entrypoint for dashboard quality, and the
required `Dashboard` CI workflow runs the **same target** — nothing here is CI-only,
and nothing in CI is missing locally.

```bash
make dashboard-check
```

It runs, in order:

| Step | What it covers |
| --- | --- |
| `dashboard-lint` | ESLint on the frontend, golangci-lint on the backend |
| `dashboard-type-check` | TypeScript type checking (frontend + Knowledge Map) |
| `dashboard-test-frontend` | Frontend unit tests |
| `dashboard-test-backend` | `go test ./...` on `dashboard/backend` |
| `dashboard-go-mod-tidy` | Verifies `go.mod` / `go.sum` are tidy |

### Running just the backend tests

```bash
make dashboard-test-backend          # from the repo root
cd dashboard/backend && go test ./... # equivalent, run directly
```

The dashboard backend is a **separate Go module**, so `go test ./...` from the repo
root does not cover it — use one of the two commands above.

Some backend tests shell out to the `vllm-sr` CLI (for example to regenerate Envoy
config), so they need its Python dependencies importable:

```bash
pip install -e src/vllm-sr
```

Without it, those tests fail with `ModuleNotFoundError`. CI installs the same package.

### Race detection is a local step, not part of the fast gate

`dashboard-check` runs plain `go test`. The race detector roughly doubles the runtime,
which is a poor trade on every PR, so it is deliberately **not** in the always-on gate.
Run it locally before pushing concurrency-sensitive work — anything touching shared
state, goroutines, caches or resolvers:

```bash
cd dashboard/backend && go test ./... -race
```

### Building

```bash
make dashboard-build   # frontend + backend, same target CI runs
```

## Deployment Details

### Docker Compose Integration Notes

- The Dashboard service is integrated as a default service in `deploy/docker-compose/docker-compose.yml`.
- No additional overlay files are needed; the compose file will start all services.
- The Dashboard depends on the `semantic-router` (for health checks), `grafana`, and `prometheus` services.

### Dockerfile Build

- A **3-stage multi-stage build** is defined in `dashboard/backend/Dockerfile`:
  1. **Node.js stage**: Builds the React frontend with Vite (`npm run build` → `dist/`)
  2. **Go builder stage**: Compiles the backend binary with multi-architecture support
  3. **Alpine runtime stage**: Combines backend + frontend dist in minimal image
- An independent Go module `dashboard/backend/go.mod` isolates backend dependencies.
- Frontend production build (`dist/`) is packaged into the image at `/app/frontend`.
- **Multi-architecture support**: The Dockerfile supports both AMD64 and ARM64 architectures.
- **Pre-built images**: Available at `ghcr.io/vllm-project/semantic-router/dashboard` with tags for releases and latest.

### Grafana Embedding Support

Grafana is already configured for embedding in `deploy/docker-compose/docker-compose.yml`:

```yaml
- GF_SECURITY_ALLOW_EMBEDDING=true
- GF_SECURITY_COOKIE_SAMESITE=lax
```

The Dashboard reverse proxy will automatically clean up `X-Frame-Options` and adjust CSP headers to ensure the iframe loads correctly.

Default dashboard path in Monitoring tab: `/d/llm-router-metrics/llm-router-metrics`.

### Health Check

The Dashboard provides a `/healthz` endpoint for container health checks:

```bash
curl http://localhost:8700/healthz
# Returns: {"status":"healthy","service":"semantic-router-dashboard"}
```

### Kubernetes deployment

Example deployment notes (adapt these to your cluster setup):

- Deployment using args `-port=8700 -static=/app/frontend -config=/app/config/config.yaml`
- Service (ClusterIP) exposing port 80 → container port 8700
- ConfigMap/Secret for upstream targets (`TARGET_*` env) and your router config file

Quick start:

```bash
# Set your namespace and apply
kubectl create ns vllm-semantic-router-system --dry-run=client -o yaml | kubectl apply -f -
# Apply your manifests under deploy/kubernetes/
kubectl -n vllm-semantic-router-system apply -f deploy/kubernetes/

# Port-forward for local testing
kubectl -n vllm-semantic-router-system port-forward svc/semantic-router-dashboard 8700:80
# Open http://localhost:8700
```

Notes:

- Configure environment variables to match your in-cluster service DNS names and namespace.
- For Helm deployments, `dashboard.persistence.enabled=true` mounts `/app/data` and wires the auth/session and workflow SQLite paths into that persistent volume. The production values profile enables this, but still keeps the dashboard at one replica because the current auth/session store is not a shared HA store.
- Mount your actual `config.yaml` via ConfigMap/Secret or a writable volume if you need runtime changes.
- To expose externally, add an Ingress or Service of type LoadBalancer according to your cluster.

Optional Ingress example (Nginx Ingress):

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: semantic-router-dashboard
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
    - host: dashboard.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: semantic-router-dashboard
                port:
                  number: 80
```

## Notes

- The dashboard is a runtime operator/try-it surface, not docs. See repository docs for broader guides.
- Upstream services remain untouched; UX unification happens at the proxy + SPA layer.
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
