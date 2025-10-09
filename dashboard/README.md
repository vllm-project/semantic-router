# Semantic Router Modern Dashboard

Unified dashboard that brings together Configuration Management, an Interactive Playground, and Real-time Monitoring & Observability. It provides a single entry point across local, Docker Compose, and Kubernetes deployments.

## Goals

- Single landing page for new/existing users
- Embed Observability (Grafana/Prometheus) and Playground (Open WebUI) via iframes behind a single backend proxy for auth and CORS/CSP control
- Read-only configuration viewer powered by the existing Semantic Router Classification API
- Environment-agnostic: consistent URLs and behavior for local dev, Compose, and K8s

## What’s already in this repo (reused)

- Prometheus + Grafana
  - Docker Compose services in `docker-compose.yml` (ports: Prometheus 9090, Grafana 3000)
  - Local observability in `docker-compose.obs.yml` (host network)
  - K8s manifests under `deploy/kubernetes/observability/{prometheus,grafana}`
  - Provisioned datasource and dashboard in `tools/observability/`
- Router metrics and API
  - Metrics at `:9190/metrics` (Prometheus format)
  - Classification API on `:8080` with endpoints like `GET /api/v1`, `GET /config/classification`
- Open WebUI integration
  - Pipe in `tools/openwebui-pipe/vllm_semantic_router_pipe.py`
  - Doc in `website/docs/tutorials/observability/open-webui-integration.md`

These are sufficient to embed and proxy—no need to duplicate core functionality.

## Architecture (MVP)

- frontend/ (SPA)
  - Tabs: Monitoring, Config Viewer, Playground
  - Iframes for Grafana dashboards and Open WebUI
  - Simple viewer for router config JSON
- backend/ (Go HTTP server)
  - Serves static frontend
  - Reverse proxy with auth/cors/csp controls:
    - `GET /embedded/grafana/*` → Grafana
    - `GET /embedded/prometheus/*` → Prometheus (optional link-outs)
    - `GET /embedded/openwebui/*` → Open WebUI (optional)
    - `GET /api/router/*` → Router Classification API (`:8080`)
    - `GET /metrics/router` → Router `/metrics` (optional aggregation later)
  - Normalizes headers for iframe embedding: strips/overrides `X-Frame-Options` and `Content-Security-Policy` frame-ancestors as needed
  - Central point for JWT/OIDC in the future (forward or exchange tokens to upstreams)

## Directory layout

```
dashboard/
├── frontend/                        # UI for configuration, playground, monitoring
│   ├─ Monitoring (iframe Grafana)
│   ├─ Config Viewer (fetch /api/router/config/classification)
│   └─ Playground (iframe Open WebUI)
├── backend/                         # Go proxy, auth, thin API
│   ├─ /embedded/grafana → Grafana
│   ├─ /embedded/prometheus → Prometheus
│   ├─ /embedded/openwebui → Open WebUI
│   └─ /api/router/* → Semantic Router API
├── deploy/
│   ├── docker/                      # Docker Compose setup for the dashboard
│   ├── kubernetes/                  # K8s manifests (Service/Ingress/ConfigMap)
│   └── local/                       # Local/dev launcher
└── helm-chart/                      # (optional) Helm chart for dashboard
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
- `ALLOW_IFRAME_EMBED` (default: true; backend will remove/override frame-busting headers)

Recommended upstream settings for embedding:

- Grafana: set `GF_SECURITY_ALLOW_EMBEDDING=true` and prefer `access: proxy` datasource (already configured)
- Open WebUI: ensure CSP/frame-ancestors allows embedding, or rely on dashboard proxy to strip/override; configure Open WebUI auth/session to work under proxied path

## URL strategy (stable, user-facing)

- Dashboard Home: `http://<host>:8700/`
- Monitoring tab: iframe `src="/embedded/grafana/d/<dashboard-uid>?kiosk&theme=light"`
- Config tab: frontend fetch `GET /api/router/config/classification`
- Playground tab: iframe `src="/embedded/openwebui/"` (rendered only if `TARGET_OPENWEBUI_URL` is set)

## Deployment matrix

1) Local dev (router and observability on host)

- Use `docker-compose.obs.yml` to start Prometheus (9090) and Grafana (3000) on host network
- Start dashboard backend locally (port 8700)
- Env examples:
  - `TARGET_GRAFANA_URL=http://localhost:3000`
  - `TARGET_PROMETHEUS_URL=http://localhost:9090`
  - `TARGET_ROUTER_API_URL=http://localhost:8080`
  - `TARGET_ROUTER_METRICS_URL=http://localhost:9190/metrics`
  - `TARGET_OPENWEBUI_URL=http://localhost:3001` (if running)

2) Docker Compose (all-in-one)

- Reuse services defined in root `docker-compose.yml`
- Add dashboard and optional Open WebUI services in `dashboard/deploy/docker/compose.yml`
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

## Implementation milestones

1) MVP (this PR)

- Scaffold `dashboard/` (this README)
- Backend: Go server with reverse proxies for `/embedded/*` and `/api/router/*`
- Frontend: minimal SPA with three tabs and iframes + JSON viewer
- Compose overlay: `dashboard/deploy/docker/compose.yml` to launch dashboard with existing stack

2) K8s manifests

- Deployment + Service + ConfigMap with env vars; optional Ingress
- Document `kubectl port-forward` for dev

3) Auth hardening and polish

- Env toggles for anonymous/off
- OIDC enablement behind a flag
- Metrics summary endpoint

## Try it (proposed)

Local with existing observability:

1. Start Prometheus/Grafana on host network:

   - `docker compose -f docker-compose.obs.yml up -d`

2. Start router natively or with Compose
3. Start dashboard backend (port 8700) with env vars above
4. Open `http://localhost:8700`

### Docker Compose unified run (after adding dashboard overlay)

```
docker compose -f docker-compose.yml -f dashboard/deploy/docker/compose.yml up --build
```

The overlay builds the dashboard image using `dashboard/backend/Dockerfile` and exposes it at `http://localhost:8700`.

### Rebuild only dashboard after code changes

```
docker compose -f docker-compose.yml -f dashboard/deploy/docker/compose.yml build dashboard
docker compose -f docker-compose.yml -f dashboard/deploy/docker/compose.yml up -d dashboard
```

### Notes on Dockerfile

- Multi-stage build (Go → distroless) defined in `dashboard/backend/Dockerfile`.
- Standalone Go module in `dashboard/backend/go.mod` isolates dependencies.
- Frontend static assets baked into the image under `/app/frontend`.

### Grafana embedding

- Root `docker-compose.yml` now sets `GF_SECURITY_ALLOW_EMBEDDING=true` for iframe usage.
- If you need stricter policies, remove the flag and authenticate Grafana separately; the dashboard proxy will still sanitize frame headers but Grafana may block unauthenticated panels.

## Notes

- The website/ (Docusaurus) remains for documentation. The dashboard is a runtime operator/try-it surface, not docs.
- We’ll keep upstream services untouched and do all UX unification at the proxy + SPA layer.
