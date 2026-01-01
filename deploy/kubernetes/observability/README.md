# Semantic Router Observability on Kubernetes

This guide adds a production-ready Prometheus + Grafana stack to the existing Semantic Router Kubernetes deployment. It includes manifests for collectors, dashboards, data sources, RBAC, and ingress so you can monitor routing performance in any cluster.

> **Namespace** – All manifests default to the `vllm-semantic-router-system` namespace to match the core deployment. Override it with Kustomize if you use a different namespace.

## What Gets Installed

| Component              | Purpose                                                                                              | Key Files                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Prometheus             | Scrapes Semantic Router metrics and stores them with persistent retention                            | `prometheus/` (`rbac.yaml`, `configmap.yaml`, `deployment.yaml`, `pvc.yaml`, `service.yaml`)  |
| Grafana                | Visualizes metrics using the bundled LLM Router dashboard and a pre-configured Prometheus datasource | `grafana/` (`secret.yaml`, `configmap-*.yaml`, `deployment.yaml`, `pvc.yaml`, `service.yaml`) |
| Dashboard              | Unified UI that links Router, Prometheus, and embeds Grafana; reads Router config                    | `dashboard/` (`configmap.yaml`, `deployment.yaml`, `service.yaml`)                            |
| Ingress (optional)     | Exposes the UIs outside the cluster                                                                  | `ingress.yaml`                                                                                |
| Dashboard provisioning | Automatically loads `deploy/llm-router-dashboard.json` into Grafana                                  | `grafana/configmap-dashboard.yaml`                                                            |

Prometheus is configured to discover the `semantic-router-metrics` service (port `9190`) automatically. Grafana provisions the same LLM Router dashboard that ships with the Docker Compose stack.

## 1. Prerequisites

- Deployed Semantic Router workload via `deploy/kubernetes/`
- A Kubernetes cluster (managed, on-prem, or kind)
- `kubectl` v1.23+
- Optional: an ingress controller (NGINX, ALB, etc.) if you want external access

## 2. Directory Layout

```
deploy/kubernetes/observability/
├── README.md
├── kustomization.yaml          # Assembles all observability components
├── ingress.yaml                # Optional HTTPS ingress examples
├── prometheus/
│   ├── configmap.yaml          # Scrape config (Kubernetes SD)
│   ├── deployment.yaml
│   ├── pvc.yaml
│   ├── rbac.yaml               # SA + ClusterRole + binding
│   └── service.yaml
├── grafana/
│   ├── configmap-dashboard.yaml    # Bundled LLM router dashboard
│   ├── configmap-provisioning.yaml # Datasource + provider config
│   ├── deployment.yaml
│   ├── pvc.yaml
│   ├── secret.yaml                 # Admin credentials (override in prod)
│   └── service.yaml
├── dashboard/
│   ├── configmap.yaml              # TARGET_* URLs for dashboard backend
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── config.yaml                 # Router config copied locally for CM
│   └── tools_db.json               # Tools DB copied locally for CM
```

## 3. Prometheus Configuration Highlights

- Uses `kubernetes_sd_configs` to enumerate endpoints in `vllm-semantic-router-system`
- Keeps 15 days of metrics by default (`--storage.tsdb.retention.time=15d`)
- Stores metrics in a `PersistentVolumeClaim` named `prometheus-data`
- RBAC rules grant read-only access to Services, Endpoints, Pods, Nodes, and EndpointSlices

### Scrape configuration snippet

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

## 4. Grafana Configuration Highlights

- Stateful deployment backed by the `grafana-storage` PVC
- Datasource provisioned automatically pointing to `http://prometheus:9090`
- Dashboard provider watches `/var/lib/grafana-dashboards`
- Bundled `llm-router-dashboard.json` is identical to `deploy/llm-router-dashboard.json`
- Admin credentials pulled from the `grafana-admin` secret (default `admin/admin` – **change this!)**

### Updating credentials

```bash
kubectl create secret generic grafana-admin \
  --namespace vllm-semantic-router-system \
  --from-literal=admin-user=monitor \
  --from-literal=admin-password='pick-a-strong-password' \
  --dry-run=client -o yaml | kubectl apply -f -
```

Remove or overwrite the committed `secret.yaml` when you adopt a different secret management approach.

## 5. Deployment Steps

### 5.1. Create the Kustomization

Create `deploy/kubernetes/observability/kustomization.yaml` (see below) to assemble all manifests. This guide assumes you keep Prometheus & Grafana in the same namespace as the router.

### 5.2. Apply manifests

```bash
kubectl apply -k deploy/kubernetes/observability/
```

Verify pods:

```bash
kubectl get pods -n vllm-semantic-router-system
```

You should see `prometheus-...`, `grafana-...`, and `semantic-router-dashboard-...` pods in `Running` state.

### 5.3. Integration with the core deployment

1. Deploy or update Semantic Router (`kubectl apply -k deploy/kubernetes/`).
2. Deploy observability stack (`kubectl apply -k deploy/kubernetes/observability/`).
3. Confirm the metrics service (`semantic-router-metrics`) has endpoints:

   ```bash
   kubectl get endpoints semantic-router-metrics -n vllm-semantic-router-system
   ```

4. Prometheus target should transition to **UP** within ~15 seconds.

### 5.4. Accessing the UIs

> **Optional Ingress** – If you prefer to keep the stack private, delete `ingress.yaml` from `kustomization.yaml` before applying.

- **Port-forward (quick check)**

  ```bash
  kubectl port-forward svc/prometheus 9090:9090 -n vllm-semantic-router-system
  kubectl port-forward svc/grafana 3000:3000 -n vllm-semantic-router-system
  kubectl port-forward svc/semantic-router-dashboard 8700:80 -n vllm-semantic-router-system
  ```

  Prometheus → http://localhost:9090, Grafana → http://localhost:3000, Dashboard → http://localhost:8700, Open WebUI → http://localhost:3001, Chat UI → http://localhost:3002

### 5.5. Ingress (production)

Use Ingress to expose the UIs on real domains with TLS.

1. Install an Ingress Controller (example: NGINX)

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm upgrade -i ingress-nginx ingress-nginx/ingress-nginx \
  -n ingress-nginx --create-namespace
```

2. Set your ingress class and hostnames

- Edit `deploy/kubernetes/observability/ingress.yaml` and replace `grafana.example.com`, `prometheus.example.com`, `dashboard.example.com` with your domains.
- Prefer using `spec.ingressClassName: nginx` instead of the deprecated annotation. You can add it via Kustomize for all Ingresses:

```yaml
patches:
  - target:
      kind: Ingress
    patch: |-
      - op: add
        path: /spec/ingressClassName
        value: nginx
```

3. Provide TLS certificates

- Option A (manual secrets):

```bash
kubectl create secret tls grafana-tls --cert=/path/to/grafana.crt --key=/path/to/grafana.key -n vllm-semantic-router-system
kubectl create secret tls prometheus-tls --cert=/path/to/prometheus.crt --key=/path/to/prometheus.key -n vllm-semantic-router-system
kubectl create secret tls dashboard-tls --cert=/path/to/dashboard.crt --key=/path/to/dashboard.key -n vllm-semantic-router-system
```

- Option B (recommended): use cert-manager; reference your `ClusterIssuer` via annotations in `ingress.yaml`.

4. Apply and verify

```bash
kubectl apply -k deploy/kubernetes/observability/
kubectl get ingress -n vllm-semantic-router-system
```

5. Configure DNS

- Point DNS A/AAAA records to the Ingress LoadBalancer address.
- For local testing, you can add temporary entries to `/etc/hosts`.

Dev tip: to run HTTP without TLS, remove the `tls:` blocks and set `nginx.ingress.kubernetes.io/ssl-redirect: "false"` in `ingress.yaml`.

## 6. Verifying Metrics Collection

1. Open Prometheus (port-forward or ingress) → **Status ▸ Targets** → ensure `semantic-router` job is green.
2. Query `rate(llm_model_completion_tokens_total[5m])` – should return data after traffic.
3. Open Grafana, log in with the admin credentials, and confirm the **LLM Router Metrics** dashboard exists under the _Semantic Router_ folder.
4. Generate traffic to Semantic Router (classification or routing requests). Key panels should start populating:
   - Prompt Category counts
   - Token usage rate per model
   - Routing modifications between models
   - Latency histograms (TTFT, completion p95)

## 7. Playground UIs

- Chat UI is configured with `OPENAI_BASE_URL` pointing at Envoy's OpenAI-compatible endpoint and uses Mongo for persistence (development default). For production, switch Mongo to a managed service.

## 8. Dashboard Customization

- Duplicate the provisioned dashboard inside Grafana to make changes while keeping the original as a template.
- Update Grafana provisioning (`grafana/configmap-provisioning.yaml`) to point to alternate folders or add new providers.
- Add additional dashboards by extending `grafana/configmap-dashboard.yaml` or mounting a different ConfigMap.
- Incorporate Kubernetes cluster metrics (CPU/memory) by adding another datasource or deploying kube-state-metrics + node exporters.

## 9. Best Practices

### Resource Sizing

- Prometheus: increase CPU/memory with higher scrape cardinality or retention > 15 days.
- Grafana: start with `500m` CPU / `1Gi` RAM; scale replicas horizontally when concurrent viewers exceed a few dozen.

### Storage

- Use SSD-backed storage classes for Prometheus when retention/window is large.
- Increase `prometheus/pvc.yaml` (default 20Gi) and `grafana/pvc.yaml` (default 10Gi) to match retention requirements.
- Enable volume snapshots or backups for dashboards and alert history.

### Security

- Replace the demo `grafana-admin` secret with credentials stored in your preferred secret manager.
- Restrict ingress access with network policies, OAuth proxies, or SSO integrations.
- Enable Grafana role-based access control and API keys for automation.
- Scope Prometheus RBAC to only the namespaces you need. If metrics run in multiple namespaces, list them in the scrape config.

### Maintenance

- Monitor Prometheus disk usage; prune retention or scale PVC before it fills up.
- Back up Grafana dashboards or store them in Git (already done through this ConfigMap).
- Roll upgrades separately: update Prometheus and Grafana images via `kustomization.yaml` patches.
- Consider adopting the Prometheus Operator (`ServiceMonitor` + `PodMonitor`) if you already run kube-prometheus-stack. A sample `ServiceMonitor` is in `website/docs/tutorials/observability/observability.md`.

## 10. Troubleshooting

| Symptom                    | Checks                                                                         | Fix                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Prometheus target **DOWN** | `kubectl get endpoints semantic-router-metrics -n vllm-semantic-router-system` | Ensure the Semantic Router deployment is running and the service labels match `app=semantic-router`, `service=metrics` |
| Grafana dashboard empty    | **Configuration → Data Sources**                                               | Confirm Prometheus datasource URL resolves and the Prometheus service is reachable                                     |
| Login fails                | `kubectl get secret grafana-admin -o yaml`                                     | Update the secret to match the credentials you expect                                                                  |
| PVC Pending                | `kubectl describe pvc prometheus-data`                                         | Provide a storage class via `storageClassName`, or provision storage manually                                          |
| Ingress 404                | `kubectl describe ingress grafana`                                             | Update hostnames, TLS secrets, and ensure ingress controller is installed                                              |

## 10. Next Steps

- Configure alerts for critical metrics (Prometheus alerting rules + Alertmanager)
- Add log aggregation (Loki, Elasticsearch, or Cloud-native logging)
- Automate stack deployment through CI/CD pipelines using `kubectl apply -k`

With this observability stack in place, you can track Semantic Router health, routing accuracy, latency distributions, and usage trends across any Kubernetes environment.
