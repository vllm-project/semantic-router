# Semantic Router Dashboard

Unified dashboard that brings together Configuration Management, an Interactive Playground, and Real-time Monitoring & Observability. It provides a single entry point across local, Docker Compose, and Kubernetes deployments.

## Goals

- Single landing page for new/existing users
- Embed Observability (Grafana/Prometheus) and Playground (Open WebUI) via iframes behind a single backend proxy for auth and CORS/CSP control
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
- Open WebUI integration
  - Pipe in `tools/openwebui-pipe/vllm_semantic_router_pipe.py`

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
- **Config** (`/config`): Real-time configuration viewer with non-persistent edit demo (see note)
- **Playground** (`/playground`): Open WebUI interface for testing

Features:

- 🌓 Dark/Light theme toggle with localStorage persistence
- 📱 Responsive design
- ⚡ Fast navigation with React Router
- 🎨 Modern UI inspired by vLLM website design

Config edit demo (frontend only):

- The Config page includes edit/add modals to showcase how configuration could be managed.
- Current backend is read-only for config: it exposes `GET /api/router/config/all` only.
- Demo save targets `POST /api/router/config/update` (not implemented by default). You can wire this endpoint in the backend to persist changes, or keep the edits as a UI mock.
- Tools DB panel attempts to load `/api/tools-db` for `tools_db.json`. Add a backend route or static file handler to serve this if you want it live.

### Backend (Go HTTP Server)

- Serves static frontend (Vite production build)
- Reverse proxy with auth/cors/csp controls:
  - `GET /embedded/grafana/*` → Grafana
  - `GET /embedded/prometheus/*` → Prometheus (optional link-outs)
  - `GET /embedded/openwebui/*` → Open WebUI (optional)
  - `GET /api/router/*` → Router Classification API (`:8080`)
  - `GET /metrics/router` → Router `/metrics` (optional aggregation later)
  - `GET /api/router/config/all` → Returns your `config.yaml` as JSON (read-only)
  - `GET /healthz` → Health check endpoint
- Normalizes headers for iframe embedding: strips/overrides `X-Frame-Options` and `Content-Security-Policy` frame-ancestors as needed
- SPA routing support: serves `index.html` for all non-asset routes
- Central point for JWT/OIDC in the future (forward or exchange tokens to upstreams)

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
│   │   │   ├── PlaygroundPage.tsx  # Open WebUI iframe
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
├── deploy/
│   └── kubernetes/                  # K8s manifests (Deployment/Service/ConfigMap)
├── README.md                        # This file
└── (no RISKS.md)                    # Security considerations are documented inline for now
```

## Environment-agnostic configuration

The backend exposes a single port (default 8700) and proxies to targets defined via environment variables. This keeps frontend URLs stable and avoids CORS by same-origining everything under the dashboard host.

Required env vars (with sensible defaults per environment):

- `DASHBOARD_PORT` (default: 8700)
- `TARGET_GRAFANA_URL`
- `TARGET_PROMETHEUS_URL`
- `TARGET_ROUTER_API_URL` (router `:8080`)
- `TARGET_ROUTER_METRICS_URL` (router `:9190/metrics`)
- `TARGET_OPENWEBUI_URL` (optional; enable playground tab only if present)
Note: The backend already adjusts frame-busting headers (X-Frame-Options/CSP) to allow embedding from the dashboard origin; no extra env flag is required.

Recommended upstream settings for embedding:

- Grafana: set `GF_SECURITY_ALLOW_EMBEDDING=true` and prefer `access: proxy` datasource (already configured)
- Open WebUI: ensure CSP/frame-ancestors allows embedding, or rely on dashboard proxy to strip/override; configure Open WebUI auth/session to work under proxied path

## URL strategy (stable, user-facing)

- Dashboard Home (Landing): `http://<host>:8700/`
- Monitoring tab: iframe `src="/embedded/grafana/d/<dashboard-uid>?kiosk&theme=light"`
- Config tab: frontend fetch `GET /api/router/config/all` (demo edit modals; see note above)
- Playground tab: iframe `src="/embedded/openwebui/"` (rendered only if `TARGET_OPENWEBUI_URL` is set)

## Deployment matrix

1) Local dev (router and observability on host)

- Use `tools/observability/docker-compose.obs.yml` to start Prometheus (9090) and Grafana (3000) on host network
- Start dashboard backend locally (port 8700)
- Env examples:
  - `TARGET_GRAFANA_URL=http://localhost:3000`
  - `TARGET_PROMETHEUS_URL=http://localhost:9090`
  - `TARGET_ROUTER_API_URL=http://localhost:8080`
  - `TARGET_ROUTER_METRICS_URL=http://localhost:9190/metrics`
  - `TARGET_OPENWEBUI_URL=http://localhost:3001` (if running)

2) Docker Compose (all-in-one)

- Reuse services defined in `deploy/docker-compose/docker-compose.yml` (Dashboard included by default)
- Env examples (inside compose network):
  - `TARGET_GRAFANA_URL=http://grafana:3000`
  - `TARGET_PROMETHEUS_URL=http://prometheus:9090`
  - `TARGET_ROUTER_API_URL=http://semantic-router:8080`
  - `TARGET_ROUTER_METRICS_URL=http://semantic-router:9190/metrics`
  - `TARGET_OPENWEBUI_URL=http://openwebui:8080` (if included)

3) Kubernetes

- Install/confirm Prometheus and Grafana via existing manifests in `deploy/kubernetes/observability`
- Deploy dashboard in `dashboard/deploy/kubernetes/`
- Configure the dashboard Deployment with in-cluster URLs:
  - `TARGET_GRAFANA_URL=http://grafana.<ns>.svc.cluster.local:3000`
  - `TARGET_PROMETHEUS_URL=http://prometheus.<ns>.svc.cluster.local:9090`
  - `TARGET_ROUTER_API_URL=http://semantic-router.<ns>.svc.cluster.local:8080`
  - `TARGET_ROUTER_METRICS_URL=http://semantic-router.<ns>.svc.cluster.local:9190/metrics`
  - `TARGET_OPENWEBUI_URL=http://openwebui.<ns>.svc.cluster.local:8080` (if installed)
- Expose the dashboard via Ingress/Gateway to the outside; upstreams remain ClusterIP

## Security & access control

- MVP: bearer token/JWT support via `Authorization: Bearer <token>` in requests to `/api/router/*` (forwarded to router API)
- Frame embedding: backend strips/overrides `X-Frame-Options` and `Content-Security-Policy` headers from upstreams to permit `frame-ancestors 'self'` only
- Future: OIDC login on dashboard, session cookie, and per-route RBAC; signed proxy sessions to Grafana/Open WebUI

## Extensibility

- New panels: add tabs/components to `frontend/`
- New integrations: add target env vars and a new `/embedded/<service>` route in backend proxy
- Metrics aggregation: add `/api/metrics` in backend to produce derived KPIs from Prometheus

## Implementation notes

— Backend: Go server with reverse proxies for `/embedded/*` and `/api/router/*`, plus `/api/router/config/all`
— Frontend: SPA with three tabs and iframes + structured config viewer
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
# Vite runs at http://localhost:3001 and proxies /api and /embedded to http://localhost:8700

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

The manifest at `dashboard/deploy/kubernetes/deployment.yaml` includes:

- Deployment with args `-port=8700 -static=/app/frontend -config=/app/config/config.yaml`
- Service (ClusterIP) exposing port 80 → container port 8700
- ConfigMap `semantic-router-dashboard-config` for upstream targets (`TARGET_*` env)
- ConfigMap `semantic-router-config` to provide a minimal `config.yaml` (replace with your real one)

Quick start:

```bash
# Set your namespace and apply
kubectl create ns vllm-semantic-router-system --dry-run=client -o yaml | kubectl apply -f -
kubectl -n vllm-semantic-router-system apply -f dashboard/deploy/kubernetes/deployment.yaml

# Port-forward for local testing
kubectl -n vllm-semantic-router-system port-forward svc/semantic-router-dashboard 8700:80
# Open http://localhost:8700
```

Notes:

- Edit `semantic-router-dashboard-config` in the YAML to match your in-cluster service DNS names and namespace.
- Replace `semantic-router-config` content with your actual `config.yaml` or mount a Secret/ConfigMap you already manage.
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
