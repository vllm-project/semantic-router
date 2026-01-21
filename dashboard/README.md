# Semantic Router Dashboard

Unified dashboard that brings together Configuration Management, an Interactive Playground, and Real-time Monitoring & Observability. It provides a single entry point across local, Docker Compose, and Kubernetes deployments.

## Goals

- Single landing page for new/existing users
- Embed Observability (Grafana/Prometheus) via iframes behind a single backend proxy for auth and CORS/CSP control
- Read-only configuration viewer powered by the existing Semantic Router Classification API
- Environment-agnostic: consistent URLs and behavior for local dev, Compose, and K8s

## What’s already in this repo (reused)

- Prometheus + Grafana
  - Docker Compose services in `deploy/docker-compose/docker-compose.yml` (Prometheus 9090, Grafana 3000)
  - Local observability in `tools/observability/docker-compose.obs.yml` (host network)
  - K8s manifests under `deploy/kubernetes/observability/{prometheus,grafana}`
  - Provisioned datasource and dashboard in `tools/observability/`
- Router metrics and API
  - Metrics at `:9190/metrics` (Prometheus format)
  - Classification API on `:8080` with endpoints like `GET /api/v1`, `GET /config/classification`

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

Features:

- 🌓 Dark/Light theme toggle with localStorage persistence (default: light)
- � Collapsible sidebar with quick section navigation (Models, Prompt Guard, Similarity Cache, Intelligent Routing, Topology, Tools Selection, Observability, Classification API)
- �📱 Responsive design
- ⚡ Fast navigation with React Router
- 🎨 Modern UI inspired by vLLM website design
- 🗺️ Topology visualization powered by React Flow

Config editing:

- The Config page includes edit/add modals for multiple sections (Models, Endpoints, Prompt Guard, Similarity Cache, Categories, Reasoning Families, Tools, Observability, Batch Classification API).
- Backend supports read/write operations:
  - `GET /api/router/config/all` returns the current config (YAML parsed and served as JSON).
  - `POST /api/router/config/update` updates the config file on disk (writes YAML). Requires the process to have write permission to the specified config path.
- Tools DB panel loads `/api/tools-db`, which serves `tools_db.json` from the same directory as your config file.
- Note for containers/Kubernetes: if the config is mounted from a read-only ConfigMap, updates won’t persist. Mount a writable volume or manage config externally if you need persistence.

Read-only dashboard mode:

- Enable via CLI: `vllm-sr serve --readonly`
- Or set env: `DASHBOARD_READONLY=true`
- Effects:
  - Frontend hides add/edit/delete actions and shows a read-only banner
  - Backend rejects write APIs with `403 Forbidden` for:
    - `POST /api/router/config/update`
    - `POST /api/router/config/defaults/update`

### Backend (Go HTTP Server)

- Serves static frontend (Vite production build)
- Reverse proxy with auth/cors/csp controls:
  - `GET /embedded/grafana/*` → Grafana
  - `GET /embedded/prometheus/*` → Prometheus (optional link-outs)
  - `GET /api/router/*` → Router Classification API (`:8080`)
  - `GET /metrics/router` → Router `/metrics` (optional aggregation later)
  - `GET /api/router/config/all` → Returns your `config.yaml` as JSON (parsed from YAML)
  - `POST /api/router/config/update` → Updates your `config.yaml` (writes YAML)
  - `GET /api/tools-db` → Returns `tools_db.json` next to your config
  - `GET /healthz` → Health check endpoint
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
│   │   │   └── *.module.css        # Scoped styles per page
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

- MVP: bearer token/JWT support via `Authorization: Bearer <token>` in requests to `/api/router/*` (forwarded to router API)
- Frame embedding: backend strips/overrides `X-Frame-Options` and `Content-Security-Policy` headers from upstreams to permit `frame-ancestors 'self'` only
- Future: OIDC login on dashboard, session cookie, and per-route RBAC; signed proxy sessions to embedded services

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
