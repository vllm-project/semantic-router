# OpenShift Observability Stack for Semantic Router

This directory contains observability stack (Prometheus + Grafana) for monitoring the vLLM Semantic Router deployment on OpenShift.

## Overview

The observability stack provides comprehensive monitoring including:

- **Model Selection Tracking**: See which model is selected when using "auto" routing
- **PII Protection Monitoring**: Track PII violations and policy denials by type (SSN, email, phone, etc.)
- **Jailbreak Detection**: Monitor jailbreak attempts and blocks in real-time
- **Performance Metrics**: Latency (TTFT, TPOT), token usage, and request rates per model
- **Cost Tracking**: Monitor costs by model and currency

## Components

| Component   | Purpose                                         | Storage |
|-------------|-------------------------------------------------|---------|
| Prometheus  | Metrics collection and storage                  | 20Gi    |
| Grafana     | Visualization with pre-configured LLM dashboard | 10Gi    |

## Quick Deployment

### Prerequisites

- Existing semantic-router deployment in `vllm-semantic-router-system` namespace
- OpenShift CLI (`oc`) configured and logged in
- Sufficient cluster resources (1.5 vCPU, 3Gi RAM)

### Deploy Observability Stack

```bash
# Using the deployment script (recommended)
cd deploy/openshift
./deploy-to-openshift.sh --observability-only

# Or using kustomize directly
oc apply -k deploy/openshift/observability/
```

### Access the Dashboards

```bash
# Get Grafana URL
oc get route grafana -n vllm-semantic-router-system -o jsonpath='{.spec.host}'

# Get Prometheus URL
oc get route prometheus -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
```

**Default Grafana credentials**: `admin` / `admin`

**⚠️ IMPORTANT**: Change the default password after first login!

## Key Metrics

### Model Routing Metrics

Track which model handles requests when using "auto" selection:

```promql
# Model routing rate (auto → Model-A or Model-B)
sum(rate(llm_model_routing_modifications_total[5m])) by (source_model, target_model)

# Prompt category distribution
sum by(category) (llm_category_classifications_count)

# Token usage by model
sum(rate(llm_model_completion_tokens_total[5m])) by (model)
```

### PII Protection Metrics

Monitor PII detection and blocking:

```promql
# PII policy denials by model
sum(rate(llm_request_errors_total{reason="pii_policy_denied"}[5m])) by (model)

# Detailed PII violations by type (SSN, email, phone, etc.)
sum(rate(llm_pii_violations_total[5m])) by (model, pii_type)

# PII refusal rate percentage
sum(rate(llm_request_errors_total{reason="pii_policy_denied"}[5m])) by (model) /
sum(rate(llm_model_requests_total[5m])) by (model)
```

### Jailbreak Protection Metrics

Monitor jailbreak attempts:

```promql
# Jailbreak blocks by model
sum(rate(llm_request_errors_total{reason="jailbreak_block"}[5m])) by (model)

# Combined security refusal rate (PII + Jailbreak)
sum(rate(llm_request_errors_total{reason=~"pii_policy_denied|jailbreak_block"}[5m])) by (model) /
sum(rate(llm_model_requests_total[5m])) by (model)
```

## Dashboard Panels

The pre-configured **LLM Router Metrics** dashboard includes:

| Panel                            | Metric                                       | Description                           |
|----------------------------------|----------------------------------------------|---------------------------------------|
| Prompt Category                  | `llm_category_classifications_count`         | Bar gauge of prompt categories        |
| Token Usage Rate by Model        | `llm_model_completion_tokens_total`          | Time series of tokens/sec by model    |
| **Model Routing Rate**           | `llm_model_routing_modifications_total`      | Shows auto → Model-A/B routing        |
| **Refusal Rates by Model**       | `llm_request_errors_total`                   | PII + Jailbreak blocks (time series)  |
| **Refusal Rate Percentage**      | Combined PII/Jailbreak %                     | Color-coded security effectiveness    |
| Model Completion Latency (p95)   | `llm_model_completion_latency_seconds`       | Response time percentiles             |
| TTFT (p95) by Model              | `llm_model_ttft_seconds`                     | Time to first token                   |
| TPOT (p95) by Model              | `llm_model_tpot_seconds`                     | Time per output token                 |
| Model Cost Rate                  | `llm_model_cost_total`                       | USD/sec by model                      |
| Total Cost by Model              | `llm_model_cost_total`                       | Cumulative costs                      |

**Bold panels** = Key for tracking model selection, PII, and jailbreak protection

## Verification

### 1. Check Prometheus Targets

```bash
# Open Prometheus and navigate to Status → Targets
PROM_URL=$(oc get route prometheus -n vllm-semantic-router-system -o jsonpath='{.spec.host}')
echo "Prometheus: http://$PROM_URL/targets"

# Expected: semantic-router job should be "UP"
```

### 2. Verify Metrics Collection

```bash
# Query Prometheus for routing metrics
curl "http://$PROM_URL/api/v1/query?query=llm_model_routing_modifications_total"

# Check for PII metrics
curl "http://$PROM_URL/api/v1/query?query=llm_pii_violations_total"
```

### 3. Test Dashboard

1. Open Grafana: `http://<grafana-route>`
2. Login with `admin` / `admin`
3. Navigate to **Dashboards** → **LLM Router Metrics**
4. Generate traffic via OpenWebUI with model="auto"
5. Watch panels update:
   - **Model Routing Rate** shows which model is selected
   - **Refusal Rates** shows PII/jailbreak blocks
   - **Token Usage** shows active models

## Cleanup

### Remove Only Observability Stack

```bash
# Using deployment script (recommended)
./deploy-to-openshift.sh --cleanup-observability

# Or using kustomize
oc delete -k deploy/openshift/observability/
```

This removes Prometheus and Grafana while keeping the semantic-router deployment intact.

### Verify Cleanup

```bash
# Should return no resources
oc get all -n vllm-semantic-router-system -l app.kubernetes.io/component=observability
```

## Troubleshooting

### Prometheus Not Scraping Metrics

**Symptom**: Prometheus targets show "DOWN" for semantic-router

**Checks**:

```bash
# Verify semantic-router-metrics service exists
oc get service semantic-router-metrics -n vllm-semantic-router-system

# Check service endpoints
oc get endpoints semantic-router-metrics -n vllm-semantic-router-system

# View Prometheus logs
oc logs deployment/prometheus -n vllm-semantic-router-system | grep semantic-router
```

**Fix**: Ensure semantic-router deployment is running and metrics port (9190) is exposed.

### Grafana Dashboard Empty

**Symptom**: Dashboard loads but shows no data

**Checks**:

```bash
# Test Prometheus datasource from within Grafana pod
oc exec deployment/grafana -n vllm-semantic-router-system -- \
  curl -s http://prometheus:9090/api/v1/query?query=up

# Check Grafana logs
oc logs deployment/grafana -n vllm-semantic-router-system
```

**Fix**: Verify Prometheus service is reachable and datasource is configured correctly.

### PVC Pending

**Symptom**: Prometheus or Grafana pods stuck in Pending state

**Checks**:

```bash
# Check PVC status
oc get pvc -n vllm-semantic-router-system

# Describe PVC for details
oc describe pvc prometheus-data -n vllm-semantic-router-system
oc describe pvc grafana-storage -n vllm-semantic-router-system
```

**Fix**: Ensure storage class `gp3-csi` exists or update PVC with available storage class.

### Grafana Login Fails

**Symptom**: Cannot login with admin/admin

**Checks**:

```bash
# Verify secret exists
oc get secret grafana-admin -n vllm-semantic-router-system

# Check secret contents (base64 encoded)
oc get secret grafana-admin -o yaml -n vllm-semantic-router-system
```

**Fix**: Update the secret with correct credentials:

```bash
oc create secret generic grafana-admin \
  --namespace vllm-semantic-router-system \
  --from-literal=admin-user=admin \
  --from-literal=admin-password=newpassword \
  --dry-run=client -o yaml | oc apply -f -

# Restart Grafana
oc rollout restart deployment/grafana -n vllm-semantic-router-system
```

## Resource Requirements

| Component  | CPU Request | CPU Limit | Memory Request | Memory Limit | Storage |
|------------|-------------|-----------|----------------|--------------|---------|
| Prometheus | 500m        | 1         | 1Gi            | 2Gi          | 20Gi    |
| Grafana    | 250m        | 500m      | 512Mi          | 1Gi          | 10Gi    |
| **Total**  | **750m**    | **1.5**   | **1.5Gi**      | **3Gi**      | **30Gi**|

## Security Considerations

1. **Change Default Password**: Update Grafana admin password immediately after deployment
2. **Network Policies**: Consider adding network policies to restrict access
3. **Route Security**: Enable TLS for Routes in production:

   ```yaml
   spec:
     tls:
       termination: edge
       insecureEdgeTerminationPolicy: Redirect
   ```

4. **RBAC**: Prometheus uses minimal RBAC (read-only access to endpoints/services)

## Advanced Configuration

### Increase Metrics Retention

Edit Prometheus deployment to increase retention from 15 days:

```yaml
# prometheus/deployment.yaml
args:
  - '--storage.tsdb.retention.time=30d'  # Increase to 30 days
```

Don't forget to increase PVC size accordingly (20Gi → 40Gi recommended).

### Add Custom Dashboards

1. Create dashboard in Grafana UI
2. Export dashboard JSON
3. Add to `grafana/configmap-dashboard.yaml`
4. Reapply: `oc apply -k deploy/openshift/observability/`

### Monitor Additional Services

Edit `prometheus/configmap.yaml` to add more scrape targets:

```yaml
scrape_configs:
  - job_name: my-service
    static_configs:
      - targets:
          - my-service:9090
```

## Example Queries

### Model Selection Analysis

```promql
# Most frequently selected model (from auto)
topk(1, sum by (target_model) (rate(llm_model_routing_modifications_total{source_model="auto"}[5m])))

# Model selection ratio
sum by (target_model) (llm_model_routing_modifications_total) /
sum(llm_model_routing_modifications_total)
```

### Security Monitoring

```promql
# PII violations by type
topk(5, sum by (pii_type) (rate(llm_pii_violations_total[5m])))

# Combined security blocks per minute
sum(rate(llm_request_errors_total{reason=~"pii_policy_denied|jailbreak_block"}[1m])) * 60

# Security effectiveness (% of requests blocked)
(sum(rate(llm_request_errors_total{reason=~"pii_policy_denied|jailbreak_block"}[5m])) /
sum(rate(llm_model_requests_total[5m]))) * 100
```

## Support

For issues or questions:

1. Check logs: `oc logs deployment/prometheus` or `oc logs deployment/grafana`
2. Review events: `oc get events -n vllm-semantic-router-system --sort-by='.lastTimestamp'`
3. File an issue at https://github.com/vllm-project/semantic-router/issues

## Next Steps

After deploying observability:

1. Generate traffic via OpenWebUI
2. Monitor model selection in real-time
3. Verify PII and jailbreak protection is working
4. Set up alerting rules (optional)
5. Export dashboards for backup
