# Semantic Router Dashboard

Unified dashboard that brings together Configuration Management, an Interactive Playground, and Real-time Monitoring & Observability. It provides a single entry point across local, Docker Compose, and Kubernetes deployments.

## Goals

- Single landing page for new/existing users
- Embed Observability (Grafana/Prometheus) via iframes behind a single backend proxy for auth and CORS/CSP control
- Read-only configuration viewer powered by the existing Semantic Router router apiserver
- Environment-agnostic: consistent URLs and behavior for local dev, Compose, and K8s

## What’s already in this repo (reused)

- Prometheus + Grafana
  - Docker Compose services in `deploy/docker-compose/docker-compose.yml` (Prometheus 9090, Grafana 3000)
  - Local observability in `tools/observability/docker-compose.obs.yml` (host network)
  - K8s manifests under `deploy/kubernetes/observability/{prometheus,grafana}`
  - Provisioned datasource and dashboard in `tools/observability/`
- Router metrics and API
  - Metrics at `:9190/metrics` (Prometheus format)
  - Router apiserver on `:8080` with endpoints like `GET /api/v1`, `GET /config/router`

These are sufficient to embed and proxy—no need to duplicate core functionality.

## Architecture

### Frontend (React + TypeScript + Vite)

Modern SPA built with:

- **React 18** with TypeScript for type safety
- **Vite 5** for fast development and optimized builds
- **React Router v6** for client-side routing
- **CSS Modules** for scoped styling with theme support (dark/light mode)

Pages:

- **Landing** (`/`): Intro landing with animated terminal demo and quick links
- **Monitoring** (`/monitoring`): Grafana dashboard embedding with custom path input
- **Config** (`/config`): Real-time configuration viewer with editable panels and save support
- **Topology** (`/topology`): Visual topology of request flow and model selection using React Flow
- **Playground** (`/playground`): Built-in chat playground for testing
- **ML Setup** (`/ml-setup`): 3-step wizard for ML model selection — benchmark, train, and generate deployment config
- **Security Policy** (`/security`): RBAC-to-router integration — map roles/groups to models and rate-limit tiers, preview generated router config fragments

Features:

- 🌓 Dark/Light theme toggle with localStorage persistence (default: light)
- � Collapsible sidebar with quick section navigation (Models, Prompt Guard, Similarity Cache, Intelligent Routing, Topology, Tools Selection, Observability, Router Apiserver)
- �📱 Responsive design
- ⚡ Fast navigation with React Router
- 🎨 Modern UI inspired by vLLM website design
- 🗺️ Topology visualization powered by React Flow

Config editing:

- The Config page includes edit/add modals for multiple sections (Models, Endpoints, Prompt Guard, Similarity Cache, Categories, Reasoning Families, Tools, Observability, Batch Classification Settings).
- Backend supports read/write operations:
  - `GET /api/router/config/all` returns the current config (YAML parsed and served as JSON).
  - `POST /api/router/config/update` updates the config file on disk (writes YAML). Requires the process to have write permission to the specified config path.
- Tools DB panel loads `/api/tools-db`, which serves `tools_db.json` from the same directory as your config file.
- Note for containers/Kubernetes: if the config is mounted from a read-only ConfigMap, updates won’t persist. Mount a writable volume or manage config externally if you need persistence.

ML Model Selection Setup (`/ml-setup`):

- A 3-step guided wizard for configuring ML-based intelligent request routing:
  - **Step 1 — Benchmark**: Upload a models YAML and queries JSONL file, then run benchmarks against your LLMs to collect performance data. Real-time progress via SSE with per-query granularity.
  - **Step 2 — Train**: Select one or more ML algorithms (KNN, K-Means, SVM, MLP) and train classifiers on the benchmark data. `ml-train/` remains the mutable latest-deployment output, while each completed job records independent private snapshots under its server-owned job directory. The Device selector (CPU/CUDA) is shown only when MLP is selected.
  - **Step 3 — Configure**: Define routing decisions (name, priority, algorithm, domains, model names) and generate a deployment-ready `ml-model-selection-values.yaml`. The generated YAML follows the semantic-router config schema and can be merged into your `config.yaml` for online inference.
- The ML pipeline data directory (`data/ml-pipeline/`) is created automatically at server startup. Upload and job directories are random/private; output paths returned by a sidecar are accepted only after regular-file, root, filename, and byte-budget validation. The service admits at most three active jobs, including at most two benchmarks and one training job.
- Supports two execution modes:
  - **Subprocess mode** (default): Runs Python scripts directly via `python3` — no additional services needed.
  - **HTTP mode**: Connects to a Python ML service sidecar (set `ML_SERVICE_URL=http://ml-service:8686`), with SSE-based progress streaming.

Read-only dashboard mode:

- Enable via CLI: `vllm-sr serve --readonly`
- Or set env: `DASHBOARD_READONLY=true`
- Effects:
  - Frontend hides add/edit/delete actions and shows a read-only banner
  - Backend rejects write APIs with `403 Forbidden` for:
    - `POST /api/router/config/update`
    - `POST /api/router/config/global/update`

### Backend (Go HTTP Server)

- Serves static frontend (Vite production build)
- Reverse proxy with auth/cors/csp controls:
  - `GET /embedded/grafana/*` → Grafana
  - `GET /embedded/prometheus/*` → Prometheus (optional link-outs)
  - `GET /api/router/*` → Router apiserver (`:8080`)
  - `GET /metrics/router` → Router `/metrics` (optional aggregation later)
  - `GET /api/router/config/all` → Returns your `config.yaml` as JSON (parsed from YAML)
  - `POST /api/router/config/update` → Updates your `config.yaml` (writes YAML)
  - `GET /api/tools-db` → Returns `tools_db.json` next to your config
  - `GET /healthz` → Health check endpoint
  - `POST /api/ml-pipeline/benchmark` → Start a benchmark job (multipart: models YAML + queries JSONL)
  - `POST /api/ml-pipeline/train` → Start a training job on benchmark data
  - `POST /api/ml-pipeline/config` → Generate deployment-ready YAML config
  - `GET /api/ml-pipeline/jobs` → List all ML pipeline jobs
  - `GET /api/ml-pipeline/jobs/{id}` → Get job status and output files
  - `GET /api/ml-pipeline/stream/{id}` → SSE stream for real-time job progress
  - `GET /api/ml-pipeline/download/{id}/{filename}` → Download job output files
  - `GET /api/security/policy` → Get current security policy config
  - `PUT /api/security/policy` → Update security policy, generate router config fragment, and auto-apply to router config
  - `POST /api/security/policy/preview` → Preview generated fragment without saving
- Normalizes headers for iframe embedding: strips/overrides `X-Frame-Options` and `Content-Security-Policy` frame-ancestors as needed
- SPA routing support: serves `index.html` for all non-asset routes
- Central point for JWT/OIDC in the future (forward or exchange tokens to upstreams)

Smart API routing:

- Requests to `/api/router/*` go to the Router API with Authorization forwarded.
- Other `/api/*` requests (e.g., Grafana’s API) are proxied to Grafana when configured.

## Directory Layout

```
dashboard/
├── frontend/                        # React + TypeScript SPA
│   ├── src/
│   │   ├── components/             # Reusable components
│   │   │   ├── Layout.tsx          # Main layout with header/nav
│   │   │   └── Layout.module.css
│   │   ├── pages/                  # Page components
│   │   │   ├── LandingPage.tsx     # Welcome page with terminal demo
│   │   │   ├── MonitoringPage.tsx  # Grafana iframe with path control
│   │   │   ├── ConfigPage.tsx      # Config viewer with API fetch
│   │   │   ├── PlaygroundPage.tsx  # Built-in chat playground
│   │   │   ├── MLSetupPage.tsx     # ML model selection 3-step wizard
│   │   │   ├── SecurityPolicyPage.tsx # RBAC role-to-model & rate-limit management
│   │   │   └── *.module.css        # Scoped styles per page
│   │   ├── hooks/
│   │   │   └── useMLPipeline.ts    # ML pipeline state management & API hooks
│   │   ├── App.tsx                 # Root component with routing
│   │   ├── main.tsx                # Entry point
│   │   └── index.css               # Global styles & CSS variables
│   ├── public/                     # Static assets (vllm.png)
│   ├── package.json                # Node dependencies
│   ├── tsconfig.json               # TypeScript configuration
│   ├── vite.config.ts              # Vite build configuration
│   └── index.html                  # SPA shell
├── backend/                         # Go reverse proxy server
│   ├── main.go                     # Proxy routes & static file server
│   ├── handlers/mlpipeline.go      # ML pipeline HTTP handlers & SSE streaming
│   ├── handlers/security_policy.go # Security policy API & config fragment generation
│   ├── mlpipeline/runner.go        # ML job orchestration (benchmark, train, config gen)
│   ├── go.mod                      # Go module (minimal dependencies)
│   └── Dockerfile                  # Multi-stage build (Node + Go + Alpine)
├── README.md                        # This file
└── (K8s/Compose manifests live under the repository-level `deploy/` folder)
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

- Dashboard auth accepts the exact-HS256 session JWT from `Authorization: Bearer <token>` for non-browser API clients or the HttpOnly `vsr_session` cookie for protected `/api/*` and `/embedded/*` requests. Every token must have an expiration and a non-empty server-owned session ID. Login, bootstrap, and password change are cookie-only by default, and the maintained browser also sends `X-VSR-Auth-Mode: cookie`; response JSON omits the JWT, the browser never stores it or adds it to URLs/headers, and startup removes the legacy local-storage value. A metadata-free non-browser client must explicitly send `X-VSR-Auth-Mode: bearer` to receive a token in JSON; browser-originated bearer-mode requests are rejected.
- Protected control-plane responses, including authentication failures, emit `Cache-Control: no-store`; public fingerprinted static assets keep their immutable cache policy. Each user is limited transactionally to 16 active and 64 retained recent sessions so repeated successful login cannot grow the SQLite session table without bound.
- Login, bootstrap, and password change set an HttpOnly, `SameSite=Lax` session cookie. Logout revokes its server-side session and clears the cookie; every browser logout, including one where `SameSite` withholds the cookie, requires same-origin evidence, while an explicit bearer API client may omit browser metadata. Access tokens in query parameters and ambiguous cookie-plus-Authorization requests are rejected before routing or proxy logs. Unsafe cookie-authenticated HTTP requests and every protected WebSocket upgrade must be strict same-origin before a handler or embedded proxy runs. Browser requests must provide one valid HTTP(S) `Origin` (or same-origin Fetch Metadata for an unsafe HTTP fallback) matching the canonical request scheme and `Host`; missing cookie-origin evidence, `null`, sibling-domain, path-bearing, scheme-mismatched, or ambiguous proxy origins are rejected. Credentialed CORS is never reflected to a sibling origin. A TLS reverse proxy must preserve the public `Host` and overwrite untrusted input with a canonical first-hop `X-Forwarded-Proto: https` value.
- `/api/auth/me` rewrites a validated legacy JavaScript-created cookie with the same JWT, HttpOnly/SameSite/Secure policy, and remaining token expiry without creating a new session. Embedded proxies strip dashboard Authorization, session cookies, query tokens, and Referer before contacting upstreams, filter upstream attempts to set `vsr_session` or widen Service Worker scope, preserve every independent CSP policy with `frame-ancestors 'self'` and `worker-src 'none'`, and bound raw WebSocket upgrade headers and time. Auth-disabled Router API forwarding remains an explicit exception for an upstream API key. Embedded HTML still needs separate-origin or equivalent capability isolation before treating an upstream application as untrusted; [#2465](https://github.com/vllm-project/semantic-router/issues/2465) owns that remaining boundary and any scoped OpenClaw browser-ticket migration.
- Every authenticated user can open `/account/security` to replace their password. The sign-in, bootstrap, and change forms use browser-standard `username`, `current-password`, and `new-password` metadata plus accessible reveal controls; `/.well-known/change-password` redirects there. A successful change atomically revokes every prior session and issues one replacement for the current browser. Administrator password resets revoke all target sessions, and a monotonic auth generation plus transactional revocation prevents deactivation/reactivation from restoring an old token or allowing already-verified login/change work to create a new one.
- New passwords follow the shared NIST-aligned policy: a 15-character minimum, Unicode NFC normalization, no composition rules, exact and normalized-variant checks against common/compromised and account-specific blocklist values, and versioned bcrypt-SHA-256 hashing. The `production` profile requires an external blocklist with at least 10,000 unique NFC entries and an exact SHA-256 configured through `DASHBOARD_PASSWORD_BLOCKLIST_SHA256` or `dashboard.passwordBlocklist.sha256`; provide a stable CSPRNG-generated signing key through `DASHBOARD_JWT_SECRET` or `dashboard.jwtSecret.existingSecret`. Before rolling back to a release without the versioned hash reader, follow the database query and backup procedure in [security-hardening.md](../docs/architecture/security-hardening.md#password-hash-rollback-check).
- Frame embedding: backend strips/overrides `X-Frame-Options` and `Content-Security-Policy` headers from upstreams to permit `frame-ancestors 'self'` only.
- **Security Policy page** (`/security`, accessible via Manager dropdown): allows admins to define role-to-model RBAC mappings and per-role rate-limit tiers. On save, the dashboard translates these into canonical router config (`routing.signals.role_bindings`, `routing.decisions`, and `global.services.ratelimit`), merges them into the running `config.yaml`, and triggers a hot-reload so the router enforces the new policy immediately. Requires the `security.manage` permission for writes; `config.read` is sufficient for viewing. See [security-hardening.md](../docs/architecture/security-hardening.md) for full details.
- **Dashboard RBAC permissions**: `feedback.submit`, `replay.read`, and `security.manage` extend the built-in role/permission matrix. Only admin-role users receive `security.manage` by default.
- **OpenClaw permissions and isolation**: `openclaw.read`, `openclaw.use`, and `openclaw.manage` separately govern observation, room/tool use, and administration. Live HTTP/SSE/WebSocket authorization is cancelled immediately on session or permission changes and is bounded globally, per user, and per session. Production returns `403` for same-origin embedded OpenClaw active content, disables stdio MCP, omits MCP/worker secrets from API DTOs, requires digest-pinned worker images, and requires `OPENCLAW_DEFAULT_NETWORK_MODE` to name an already-created user-defined network. Provision requests may omit `networkMode`, use the legacy generic `host`/`bridge` UI value (which is normalized to the configured network), or repeat that exact configured value; production never creates or accepts any other caller-selected network. It also rotates the 192-bit gateway credential on every reprovision and controls only containers/volumes bearing this Dashboard instance's ownership labels. See [security-hardening.md](../docs/architecture/security-hardening.md#openclaw-and-mcp-production-boundary) for the remaining Docker-socket/root-worker boundary.
- **Control-plane input and job isolation**: maintained JSON/YAML/multipart handlers apply strict byte, field, cardinality, and work budgets before side effects. Internal and fixed reverse-proxy clients do not inherit ambient proxies or redirects; transformed Jaeger/OpenClaw/Replay bodies are capped and fail closed. Evaluation and ML live admission/SSE state is bounded, storage is private and symlink-safe, and ML downloads resolve only job-owned snapshots. Aggregate Evaluation/ML retention and restart recovery remain tracked by [TD048](../docs/agent/tech-debt/td-048-dashboard-job-lifecycle-ownership-gap.md).
- Auth users, roles, permissions, audit logs, workflow state, and session IDs use SQLite under `./data` by default. The auth database and its sidecars are restricted to mode `0600`, and service-created leaf directories use `0700`. In containers or Kubernetes, mount `/app/data` or set `DASHBOARD_AUTH_DB_PATH` and `DASHBOARD_WORKFLOW_DB_PATH` to persistent paths if you need state to survive restarts.
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
