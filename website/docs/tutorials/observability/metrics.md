# Metrics & Monitoring

Metrics collection and visualization for Semantic Router using Prometheus and Grafana.

---

## 1. Metrics & Endpoints

| Component                | Endpoint                  | Notes                                      |
| ------------------------ | ------------------------- | ------------------------------------------ |
| Router metrics           | `:9190/metrics`           | Prometheus format (flag: `--metrics-port`) |
| Router health            | `:8080/health`            | HTTP readiness/liveness                    |
| Envoy metrics (optional) | `:19000/stats/prometheus` | If Envoy is enabled                        |

**Configuration location**: `tools/observability/`  
**Dashboard**: `tools/observability/llm-router-dashboard.json`

---

## 2. Local Mode (Router on Host)

Run router natively on host, observability in Docker.

### Quick Start

```bash
# Start router
make run-router

# Start observability
make o11y-local
```

**Access:**

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

**Verify targets:**

```bash
# Check Prometheus scrapes localhost:9190
open http://localhost:9090/targets
```

**Stop:**

```bash
make stop-observability
```

### Configuration

All configs in `tools/observability/`:

- `prometheus.yaml` - Scrapes the target from the `ROUTER_TARGET` env var (default: `localhost:9190`)
- `grafana-datasource.yaml` - Points to `localhost:9090`
- `grafana-dashboard.yaml` - Dashboard provisioning
- `llm-router-dashboard.json` - Dashboard definition

### Troubleshooting

| Issue         | Fix                                     |
| ------------- | --------------------------------------- |
| Target DOWN   | Start router: `make run-router`         |
| No metrics    | Generate traffic, check `:9190/metrics` |
| Port conflict | Change port or stop conflicting service |

---

## 3. Docker Compose Mode

All services in Docker containers.

### Quick Start

```bash
# Start full stack (includes observability)
docker compose -f deploy/docker-compose/docker-compose.yml up --build

# Or with testing profile
docker compose -f deploy/docker-compose/docker-compose.yml --profile testing up --build
```

**Access:**

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

**Expected targets:**

- `semantic-router:9190`
- `envoy-proxy:19000` (optional)

### Configuration

Same configs as local mode (`tools/observability/`), but:

- `ROUTER_TARGET=semantic-router:9190`
- `PROMETHEUS_URL=prometheus:9090`
- Uses `semantic-network` bridge network

---

## 4. Kubernetes Mode

Production-ready Prometheus + Grafana for K8s clusters.

> **Namespace**: `vllm-semantic-router-system`

### Components

| Component  | Purpose                               | Location                                       |
| ---------- | ------------------------------------- | ---------------------------------------------- |
| Prometheus | Scrapes router metrics, 15d retention | `deploy/kubernetes/observability/prometheus/`  |
| Grafana    | Dashboard visualization               | `deploy/kubernetes/observability/grafana/`     |
| Ingress    | Optional external access              | `deploy/kubernetes/observability/ingress.yaml` |

### Deploy

```bash
# Apply manifests
kubectl apply -k deploy/kubernetes/observability/

# Verify
kubectl get pods -n vllm-semantic-router-system
```

### Access

**Port-forward:**

```bash
kubectl port-forward svc/prometheus 9090:9090 -n vllm-semantic-router-system
kubectl port-forward svc/grafana 3000:3000 -n vllm-semantic-router-system
```

**Ingress:** Customize `ingress.yaml` with your domain and TLS

### Key Configuration

**Prometheus** uses Kubernetes service discovery:

```yaml
scrape_configs:
  - job_name: semantic-router
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names: [vllm-semantic-router-system]
```

**Grafana** credentials (change in production):

```bash
kubectl create secret generic grafana-admin \
  --namespace vllm-semantic-router-system \
  --from-literal=admin-user=admin \
  --from-literal=admin-password='your-password'
```

---

## 5. Key Metrics

| Metric                                  | Type      | Description              |
| --------------------------------------- | --------- | ------------------------ |
| `llm_category_classifications_count`    | counter   | Category classifications |
| `llm_model_completion_tokens_total`     | counter   | Tokens per model         |
| `llm_model_routing_modifications_total` | counter   | Model routing changes    |
| `llm_model_completion_latency_seconds`  | histogram | Completion latency       |

**Example queries:**

```promql
rate(llm_model_completion_tokens_total[5m])
histogram_quantile(0.95, rate(llm_model_completion_latency_seconds_bucket[5m]))
```

---

## 6. Troubleshooting

| Issue           | Check               | Fix                                                   |
| --------------- | ------------------- | ----------------------------------------------------- |
| Target DOWN     | Prometheus /targets | Verify router is running and exposing `:9190/metrics` |
| No metrics      | Generate traffic    | Send requests through router                          |
| Dashboard empty | Grafana datasource  | Check Prometheus URL configuration                    |

---
