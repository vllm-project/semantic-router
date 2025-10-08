# Observability Configuration

Prometheus and Grafana configuration files for monitoring semantic-router.

## Files

- `prometheus.yaml` - Prometheus scrape config (uses `$ROUTER_TARGET` env var)
- `grafana-datasource.yaml` - Grafana datasource (uses `$PROMETHEUS_URL` env var)
- `grafana-dashboard.yaml` - Dashboard provisioning config
- `llm-router-dashboard.json` - LLM Router dashboard

## Usage

**Local mode** (router on host, observability in Docker):

```bash
make o11y-local
```

**Compose mode** (all services in Docker):

```bash
make o11y-compose
# or: docker compose up
```

**Access:**

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
