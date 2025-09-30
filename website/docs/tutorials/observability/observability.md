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

This guide adds a production-ready Prometheus + Grafana stack to the existing Semantic Router Kubernetes deployment. It includes manifests for collectors, dashboards, data sources, RBAC, and ingress so you can monitor routing performance in any cluster.

> **Namespace** – All manifests default to the `vllm-semantic-router-system` namespace to match the core deployment. Override it with Kustomize if you use a different namespace.

## What Gets Installed

| Component    | Purpose | Key Files |
|--------------|---------|-----------|
| Prometheus   | Scrapes Semantic Router metrics and stores them with persistent retention | `prometheus/` (`rbac.yaml`, `configmap.yaml`, `deployment.yaml`, `pvc.yaml`, `service.yaml`)|
| Grafana      | Visualizes metrics using the bundled LLM Router dashboard and a pre-configured Prometheus datasource | `grafana/` (`secret.yaml`, `configmap-*.yaml`, `deployment.yaml`, `pvc.yaml`, `service.yaml`)|
| Ingress (optional) | Exposes the UIs outside the cluster | `ingress.yaml`|
| Dashboard provisioning | Automatically loads `deploy/llm-router-dashboard.json` into Grafana | `grafana/configmap-dashboard.yaml`|

Prometheus is configured to discover the `semantic-router-metrics` service (port `9190`) automatically. Grafana provisions the same LLM Router dashboard that ships with the Docker Compose stack.

### 1. Prerequisites

- Deployed Semantic Router workload via `deploy/kubernetes/`
- A Kubernetes cluster (managed, on-prem, or kind)
- `kubectl` v1.23+
- Optional: an ingress controller (NGINX, ALB, etc.) if you want external access

### 2. Directory Layout

```
deploy/kubernetes/observability/
├── README.md
├── kustomization.yaml          # (created in the next step)
├── ingress.yaml                # optional HTTPS ingress examples
├── prometheus/
│   ├── configmap.yaml          # Scrape config (Kubernetes SD)
│   ├── deployment.yaml
│   ├── pvc.yaml
│   ├── rbac.yaml               # SA + ClusterRole + binding
│   └── service.yaml
└── grafana/
    ├── configmap-dashboard.yaml    # Bundled LLM router dashboard
    ├── configmap-provisioning.yaml # Datasource + provider config
    ├── deployment.yaml
    ├── pvc.yaml
    ├── secret.yaml                 # Admin credentials (override in prod)
    └── service.yaml
```

### 3. Prometheus Configuration Highlights

- Uses `kubernetes_sd_configs` to enumerate endpoints in `vllm-semantic-router-system`
- Keeps 15 days of metrics by default (`--storage.tsdb.retention.time=15d`)
- Stores metrics in a `PersistentVolumeClaim` named `prometheus-data`
- RBAC rules grant read-only access to Services, Endpoints, Pods, Nodes, and EndpointSlices

#### Scrape configuration snippet

```yaml
scrape_configs:
  - job_name: semantic-router
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
            - vllm-semantic-router-system
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        regex: semantic-router-metrics
        action: keep
      - source_labels: [__meta_kubernetes_endpoint_port_name]
        regex: metrics
        action: keep
```

Modify the namespace or service name if you changed them in your primary deployment.

### 4. Grafana Configuration Highlights

- Stateful deployment backed by the `grafana-storage` PVC
- Datasource provisioned automatically pointing to `http://prometheus:9090`
- Dashboard provider watches `/var/lib/grafana-dashboards`
- Bundled `llm-router-dashboard.json` is identical to `deploy/llm-router-dashboard.json`
- Admin credentials pulled from the `grafana-admin` secret (default `admin/admin` – **change this!)**

#### Updating credentials

```bash
kubectl create secret generic grafana-admin \
  --namespace vllm-semantic-router-system \
  --from-literal=admin-user=monitor \
  --from-literal=admin-password='pick-a-strong-password' \
  --dry-run=client -o yaml | kubectl apply -f -
```

Remove or overwrite the committed `secret.yaml` when you adopt a different secret management approach.

### 5. Deployment Steps

#### 5.1. Create the Kustomization

Create `deploy/kubernetes/observability/kustomization.yaml` (see below) to assemble all manifests. This guide assumes you keep Prometheus & Grafana in the same namespace as the router.

#### 5.2. Apply manifests

```bash
kubectl apply -k deploy/kubernetes/observability/
```

Verify pods:

```bash
kubectl get pods -n vllm-semantic-router-system
```

You should see `prometheus-...` and `grafana-...` pods in `Running` state.

#### 5.3. Integration with the core deployment

1. Deploy or update Semantic Router (`kubectl apply -k deploy/kubernetes/`).
2. Deploy observability stack (`kubectl apply -k deploy/kubernetes/observability/`).
3. Confirm the metrics service (`semantic-router-metrics`) has endpoints:

   ```bash
   kubectl get endpoints semantic-router-metrics -n vllm-semantic-router-system
   ```

4. Prometheus target should transition to **UP** within ~15 seconds.

#### 5.4. Accessing the UIs

> **Optional Ingress** – If you prefer to keep the stack private, delete `ingress.yaml` from `kustomization.yaml` before applying.

- **Port-forward (quick check)**

  ```bash
  kubectl port-forward svc/prometheus 9090:9090 -n vllm-semantic-router-system
  kubectl port-forward svc/grafana 3000:3000 -n vllm-semantic-router-system
  ```

  Prometheus → http://localhost:9090, Grafana → http://localhost:3000

- **Ingress (production)** – Customize `ingress.yaml` with real domains, TLS secrets, and your ingress class before applying. Replace `*.example.com` and configure HTTPS certificates via cert-manager or your provider.

### 6. Verifying Metrics Collection

1. Open Prometheus (port-forward or ingress) → **Status ▸ Targets** → ensure `semantic-router` job is green.
2. Query `rate(llm_model_completion_tokens_total[5m])` – should return data after traffic.
3. Open Grafana, log in with the admin credentials, and confirm the **LLM Router Metrics** dashboard exists under the *Semantic Router* folder.
4. Generate traffic to Semantic Router (classification or routing requests). Key panels should start populating:
   - Prompt Category counts
   - Token usage rate per model
   - Routing modifications between models
   - Latency histograms (TTFT, completion p95)

### 7. Dashboard Customization

- Duplicate the provisioned dashboard inside Grafana to make changes while keeping the original as a template.
- Update Grafana provisioning (`grafana/configmap-provisioning.yaml`) to point to alternate folders or add new providers.
- Add additional dashboards by extending `grafana/configmap-dashboard.yaml` or mounting a different ConfigMap.
- Incorporate Kubernetes cluster metrics (CPU/memory) by adding another datasource or deploying kube-state-metrics + node exporters.

### 8. Best Practices

#### Resource Sizing

- Prometheus: increase CPU/memory with higher scrape cardinality or retention > 15 days.
- Grafana: start with `500m` CPU / `1Gi` RAM; scale replicas horizontally when concurrent viewers exceed a few dozen.

#### Storage

- Use SSD-backed storage classes for Prometheus when retention/window is large.
- Increase `prometheus/pvc.yaml` (default 20Gi) and `grafana/pvc.yaml` (default 10Gi) to match retention requirements.
- Enable volume snapshots or backups for dashboards and alert history.

#### Security

- Replace the demo `grafana-admin` secret with credentials stored in your preferred secret manager.
- Restrict ingress access with network policies, OAuth proxies, or SSO integrations.
- Enable Grafana role-based access control and API keys for automation.
- Scope Prometheus RBAC to only the namespaces you need. If metrics run in multiple namespaces, list them in the scrape config.

#### Maintenance

- Monitor Prometheus disk usage; prune retention or scale PVC before it fills up.
- Back up Grafana dashboards or store them in Git (already done through this ConfigMap).
- Roll upgrades separately: update Prometheus and Grafana images via `kustomization.yaml` patches.
- Consider adopting the Prometheus Operator (`ServiceMonitor` + `PodMonitor`) if you already run kube-prometheus-stack. A sample `ServiceMonitor` is in `website/docs/tutorials/observability/observability.md`.

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
