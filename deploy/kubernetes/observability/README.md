# Kubernetes observability example

This Kustomize package runs Prometheus, Grafana, and the Semantic Router
Dashboard in `vllm-semantic-router-system`. It is a self-contained example for
development clusters. It is not a substitute for a platform-managed monitoring
stack, identity provider, backup policy, or alert-routing service.

| Component | What the manifests configure |
| --- | --- |
| Prometheus | Kubernetes endpoint discovery for `semantic-router-metrics`, 15-day retention, a PVC, and Router alert rules. |
| Grafana | A Prometheus datasource, the bundled Router dashboard, a PVC, and local admin credentials. |
| Dashboard | Router config and tools data plus links to the monitoring services. |
| Ingress | Example hosts and TLS Secret names for all three UIs. |

## Review before applying

The checked-in values are intentionally easy to inspect, not secure defaults
for a shared cluster:

- [`grafana/secret.yaml`](grafana/secret.yaml) contains `admin` / `admin`.
- [`ingress.yaml`](ingress.yaml) contains example hostnames and TLS Secret
  names.
- Prometheus has cluster-scoped discovery RBAC and persistent local retention.
- Dashboard state and config mutation require explicit storage and access
  decisions.

Replace the Grafana Secret through your secret manager, patch or remove the
Ingress resources, choose storage classes, and review resource limits before
deployment. See the
[security hardening guide](../../../website/docs/installation/security-hardening.md)
for the Dashboard and inference boundary.

Render the final resources first:

```bash
kubectl kustomize deploy/kubernetes/observability/
```

## Deploy

The Router metrics Service must exist in the same namespace and expose a port
named `metrics`:

```bash
kubectl get service semantic-router-metrics \
  --namespace vllm-semantic-router-system
```

After applying your security and storage overlays:

```bash
kubectl apply -k deploy/kubernetes/observability/
kubectl rollout status deployment/prometheus \
  --namespace vllm-semantic-router-system
kubectl rollout status deployment/grafana \
  --namespace vllm-semantic-router-system
kubectl rollout status deployment/semantic-router-dashboard \
  --namespace vllm-semantic-router-system
```

For a disposable cluster, you can replace the demo Grafana credential before
use:

```bash
kubectl create secret generic grafana-admin \
  --namespace vllm-semantic-router-system \
  --from-literal=admin-user=monitor \
  --from-literal=admin-password="$GRAFANA_ADMIN_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/grafana \
  --namespace vllm-semantic-router-system
```

Avoid putting the password directly in a committed overlay.

## Access without public ingress

Use separate terminals for the services you need:

```bash
kubectl port-forward --namespace vllm-semantic-router-system \
  service/prometheus 9090:9090
```

```bash
kubectl port-forward --namespace vllm-semantic-router-system \
  service/grafana 3000:3000
```

```bash
kubectl port-forward --namespace vllm-semantic-router-system \
  service/semantic-router-dashboard 8700:80
```

Prometheus is at `http://127.0.0.1:9090`, Grafana at
`http://127.0.0.1:3000`, and the Dashboard at `http://127.0.0.1:8700`.

## Verify data

1. Open Prometheus **Status > Targets** and find the `semantic-router` job.
2. Send inference traffic through Envoy.
3. Query a metric that the current Router emits, such as
   `llm_model_requests_total`.
4. Open the provisioned Grafana Router dashboard and confirm its time range
   includes the traffic.

If the target is missing, check service discovery before changing dashboards:

```bash
kubectl get endpoints semantic-router-metrics \
  --namespace vllm-semantic-router-system
kubectl logs deployment/prometheus \
  --namespace vllm-semantic-router-system
```

## Alert rules

[`prometheus/rules.yaml`](prometheus/rules.yaml) defines thresholds for request
errors, completion latency, TTFT, TPOT, routing latency, in-flight requests, and
cache hit rate. Treat these as starting points. Tune them from an observed
baseline and route them through your existing Alertmanager or incident system;
this package does not install notification delivery.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Prometheus target is down | Service endpoints, port name `metrics`, namespace, and Prometheus RBAC. |
| Grafana panels are empty | Prometheus datasource health, query metric names, dashboard time range, and actual traffic. |
| PVC is pending | `kubectl describe pvc` and the cluster storage class. |
| Ingress returns 404 or TLS errors | Ingress class, patched hostnames, DNS, and referenced TLS Secrets. |
| Dashboard cannot update config | The config mount, writable flags, and persistent state described in [`dashboard/README.md`](../../../dashboard/README.md). |

To remove the example:

```bash
kubectl delete -k deploy/kubernetes/observability/
```

PVC deletion and retention behavior depends on the cluster and storage class;
confirm it before removing data you may need.
