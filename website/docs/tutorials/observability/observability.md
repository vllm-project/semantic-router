# Observability

This page focuses solely on collecting and visualizing metrics for Semantic Router using Prometheus and Grafana—deployment method (Docker Compose vs Kubernetes) is covered in `docker-quickstart.md`.

---

## 1. Metrics & Endpoints Summary

| Component                    | Endpoint                  | Notes                                      |
| ---------------------------- | ------------------------- | ------------------------------------------ |
| Router metrics               | `:9190/metrics`           | Prometheus format (flag: `--metrics-port`) |
| Router health (future probe) | `:8080/health`            | HTTP readiness/liveness candidate          |
| Envoy metrics (optional)     | `:19000/stats/prometheus` | If you enable Envoy                        |

Dashboard JSON: `deploy/llm-router-dashboard.json`.

Primary source file exposing metrics: `src/semantic-router/cmd/main.go` (uses `promhttp`).

---

## 2. Docker Compose Observability

Compose bundles: `prometheus`, `grafana`, `semantic-router`, (optional) `envoy`, `mock-vllm`.

Key files:

- `config/prometheus.yaml`
- `config/grafana/datasource.yaml`
- `config/grafana/dashboards.yaml`
- `deploy/llm-router-dashboard.json`

Start (with testing profile example):

```bash
CONFIG_FILE=/app/config/config.testing.yaml docker compose --profile testing up --build
```

Access:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

Expected Prometheus targets:

- `semantic-router:9190`
- `envoy-proxy:19000` (optional)

---

## 3. Kubernetes Observability

After applying `deploy/kubernetes/`, you get services:

- `semantic-router` (gRPC)
- `semantic-router-metrics` (metrics 9190)

### 3.1 Prometheus Operator (ServiceMonitor)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: semantic-router
  namespace: semantic-router
spec:
  selector:
    matchLabels:
      app: semantic-router
      service: metrics
  namespaceSelector:
    matchNames: ["semantic-router"]
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
```

Ensure the metrics Service carries a label like `service: metrics`. (It does in the provided manifests.)

### 3.2 Plain Prometheus Static Scrape

```yaml
scrape_configs:
  - job_name: semantic-router
    kubernetes_sd_configs:
      - role: endpoints
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        regex: semantic-router-metrics
        action: keep
```

### 3.3 Port Forward for Spot Checks

```bash
kubectl -n semantic-router port-forward svc/semantic-router-metrics 9190:9190
curl -s localhost:9190/metrics | head
```

### 3.4 Grafana Dashboard Provision

If using kube-prometheus-stack or a Grafana sidecar:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: semantic-router-dashboard
  namespace: semantic-router
  labels:
    grafana_dashboard: "1"
data:
  llm-router-dashboard.json: |
    # Paste JSON from deploy/llm-router-dashboard.json
```

Otherwise import the JSON manually in Grafana UI.

---

## 4. Key Metrics (Sample)

| Metric                                                        | Type      | Description                                  |
| ------------------------------------------------------------- | --------- | -------------------------------------------- |
| `llm_category_classifications_count`                          | counter   | Number of category classification operations |
| `llm_model_completion_tokens_total`                           | counter   | Tokens emitted per model                     |
| `llm_model_routing_modifications_total`                       | counter   | Model switch / routing adjustments           |
| `llm_model_completion_latency_seconds`                        | histogram | Completion latency distribution              |
| `process_cpu_seconds_total` / `process_resident_memory_bytes` | standard  | Runtime resource usage                       |

Use typical PromQL patterns:

```promql
rate(llm_model_completion_tokens_total[5m])
histogram_quantile(0.95, sum by (le) (rate(llm_model_completion_latency_seconds_bucket[5m])))
```

---

## 5. Troubleshooting

| Symptom               | Likely Cause              | Check                                    | Fix                                                              |
| --------------------- | ------------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| Target DOWN (Docker)  | Service name mismatch     | Prometheus /targets                      | Ensure `semantic-router` container running                       |
| Target DOWN (K8s)     | Label/selectors mismatch  | `kubectl get ep semantic-router-metrics` | Align labels or ServiceMonitor selector                          |
| No new tokens metrics | No traffic                | Generate chat/completions via Envoy      | Send test requests                                               |
| Dashboard empty       | Datasource URL wrong      | Grafana datasource settings              | Point to `http://prometheus:9090` (Docker) or cluster Prometheus |
| Large 5xx spikes      | Backend model unreachable | Router logs                              | Verify vLLM endpoints configuration                              |

---
