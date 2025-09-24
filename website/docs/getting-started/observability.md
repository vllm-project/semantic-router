# Observability

This guide helps you set up Prometheus + Grafana for this project, covering local Docker Compose and Kubernetes deployment, key metrics and PromQL, importing the bundled dashboard, alerting, security hardening, and troubleshooting.

Note: The project already exposes Prometheus metrics and ships a Grafana dashboard. You only need to run Prometheus/Grafana and point Prometheus at the router’s metrics endpoint.

## Endpoints and Ports

- The router process starts a dedicated Prometheus metrics HTTP server:
  - Path: `/metrics`
  - Port: `9190` by default (override with `--metrics-port`)
- The Classification API (optional) listens on `8080` (`--api-port`), with a health check at `GET /health` (health only, no metrics).
- Envoy (optional) admin port defaults to `19000`; Prometheus metrics are typically at `/stats/prometheus`.

Code reference: `src/semantic-router/cmd/main.go` uses `promhttp` to expose `/metrics`; default port is 9190.

## Quickstart (Docker Compose)

The provided `docker-compose.yml` starts `semantic-router` and `envoy`. Add Prometheus and Grafana services on the same network to scrape the router’s metrics.

1) Add Prometheus config at `config/prometheus.yml`:

```yaml
global:
  scrape_interval: 10s
  evaluation_interval: 10s

scrape_configs:
  # Semantic Router
  - job_name: semantic-router
    metrics_path: /metrics
    static_configs:
      - targets: ["semantic-router:9190"]
        labels:
          service: semantic-router
          env: dev

  # Optional: Envoy
  - job_name: envoy
    metrics_path: /stats/prometheus
    static_configs:
      - targets: ["envoy-proxy:19000"]
        labels:
          service: envoy
          env: dev
```

2) In the same folder as `docker-compose.yml`, add/merge the following services (ensure the same `semantic-network`):

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.53.0
    container_name: prometheus
    volumes:
      - ./config/prometheus.yaml:/etc/prometheus/prometheus.yaml:ro
    command:
      - --config.file=/etc/prometheus/prometheus.yaml
      - --storage.tsdb.retention.time=15d
    ports:
      - "9090:9090"
    networks:
      - semantic-network

  grafana:
    image: grafana/grafana:11.5.1
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    ports:
      - "3000:3000"
    volumes:
      # Auto-provision Prometheus datasource and dashboard (see below)
      - ./config/grafana/datasource.yaml:/etc/grafana/provisioning/datasources/datasource.yaml:ro
      - ./config/grafana/dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:ro
      - ./deploy/llm-router-dashboard.json:/etc/grafana/provisioning/dashboards/llm-router-dashboard.json:ro
    networks:
      - semantic-network

networks:
  semantic-network:
    driver: bridge
```

3) Grafana provisioning for datasource and dashboards (optional but recommended):

`config/grafana/datasource.yml`

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

`config/grafana/dashboards.yml`

```yaml
apiVersion: 1
providers:
  - name: LLM Router Dashboards
    orgId: 1
    folder: "LLM Router"
    type: file
    disableDeletion: false
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

After (re)starting Compose you can access:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default admin/admin; change on first login)

Dashboard import: Grafana will auto-load the bundled `deploy/llm-router-dashboard.json`. You can also import it manually via the Grafana UI.
