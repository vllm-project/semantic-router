# Observability

Set up Prometheus + Grafana locally with the existing Docker Compose in this repo. The router already exposes Prometheus metrics and ships a ready-to-use Grafana dashboard, so you mainly need to run the services and ensure Prometheus points at the metrics endpoint.

## What’s included

- Router metrics server: `/metrics` on port `9190` (override with `--metrics-port`).
- Classification API health check: `GET /health` on `8080` (`--api-port`).
- Envoy (optional): admin on `19000`, Prometheus metrics at `/stats/prometheus`.
- Docker Compose services: `semantic-router`, `envoy`, `prometheus`, `grafana` on the same `semantic-network`.
- Grafana dashboard: `deploy/llm-router-dashboard.json` (auto-provisioned).

Code reference: `src/semantic-router/cmd/main.go` uses `promhttp` to expose `/metrics` (default `:9190`).

## Files to know

- Prometheus config: `config/prometheus.yaml`. Ensure the path matches the volume mount in `docker-compose.yml`.
- Grafana provisioning:
  - Datasource: `config/grafana/datasource.yaml`
  - Dashboards: `config/grafana/dashboards.yaml`
- Dashboard JSON: `deploy/llm-router-dashboard.json`

These files are already referenced by `docker-compose.yml` so you typically don’t need to edit them unless you’re changing targets or credentials.

## How it works (local)

- Prometheus runs in the same Docker network and scrapes `semantic-router:9190/metrics`. No host port needs to be published for metrics.
- Grafana connects to Prometheus via the internal URL `http://prometheus:9090` and auto-loads the bundled dashboard.
- Envoy (if enabled) can also be scraped by Prometheus at `envoy-proxy:19000/stats/prometheus`.

## Start and access

1) From the project root, start Compose (Prometheus and Grafana are included in the provided file).

```bash
# try it out with mock-vllm
CONFIG_FILE=/app/config/config.testing.yaml docker compose --profile testing up --build
```

2) Open the UIs:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (default admin/admin — change on first login)
3) In Grafana, the “LLM Router” dashboard is pre-provisioned. If needed, import `deploy/llm-router-dashboard.json` manually.

## Minimal expectations

- Prometheus should list targets for:
  - `semantic-router:9190` (required)
  - `envoy-proxy:19000` (optional)
- Grafana’s datasource should point to `http://prometheus:9090` inside the Docker network.

That’s it—run the stack, and you’ll have Prometheus scraping the router plus a prebuilt Grafana dashboard out of the box.
