# OpenShift Deployment for Semantic Router

This directory contains OpenShift-specific deployment manifests for the vLLM Semantic Router.

## Quick Deployment

### Prerequisites

- OpenShift cluster access
- `oc` CLI tool configured and logged in
- Cluster admin privileges (or permissions to create namespaces and routes)

### One-Command Deployment

```bash
oc apply -k deploy/openshift/
```

### Step-by-Step Deployment

1. **Create namespace:**

   ```bash
   oc apply -f deploy/openshift/namespace.yaml
   ```

2. **Deploy core resources:**

   ```bash
   oc apply -f deploy/openshift/pvc.yaml
   oc apply -f deploy/openshift/deployment.yaml
   oc apply -f deploy/openshift/service.yaml
   ```

3. **Create external routes:**

   ```bash
   oc apply -f deploy/openshift/routes.yaml
   ```

## Accessing Services

After deployment, the services will be accessible via OpenShift Routes:

### Get Route URLs

```bash
# Classification API (HTTP REST)
oc get route semantic-router-api -n vllm-semantic-router-system -o jsonpath='{.spec.host}'

# gRPC API
oc get route semantic-router-grpc -n vllm-semantic-router-system -o jsonpath='{.spec.host}'

# Metrics
oc get route semantic-router-metrics -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
```

### Example Usage

```bash
# Get the API route
API_ROUTE=$(oc get route semantic-router-api -n vllm-semantic-router-system -o jsonpath='{.spec.host}')

# Test health endpoint
curl https://$API_ROUTE/health

# Test classification
curl -X POST https://$API_ROUTE/api/v1/classify/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?"}'
```

## Architecture Differences from Kubernetes

### Security Context

- Removed `runAsNonRoot: false` for OpenShift compatibility
- Enhanced security context with `capabilities.drop: ALL` and `seccompProfile`
- OpenShift automatically enforces non-root containers

### Networking

- Uses OpenShift Routes instead of port-forwarding for external access
- TLS termination handled by OpenShift router
- Automatic HTTPS certificates via OpenShift

### Storage

- Uses OpenShift's default storage class
- PVC automatically bound to available storage

## Monitoring

### Check Deployment Status

```bash
# Check pods
oc get pods -n vllm-semantic-router-system

# Check services
oc get services -n vllm-semantic-router-system

# Check routes
oc get routes -n vllm-semantic-router-system

# Check logs
oc logs -f deployment/semantic-router -n vllm-semantic-router-system
```

### Metrics

Access Prometheus metrics via the metrics route:

```bash
METRICS_ROUTE=$(oc get route semantic-router-metrics -n vllm-semantic-router-system -o jsonpath='{.spec.host}')
curl https://$METRICS_ROUTE/metrics
```

## Cleanup

Remove all resources:

```bash
oc delete -k deploy/openshift/
```

Or remove individual components:

```bash
oc delete -f deploy/openshift/routes.yaml
oc delete -f deploy/openshift/service.yaml
oc delete -f deploy/openshift/deployment.yaml
oc delete -f deploy/openshift/pvc.yaml
oc delete -f deploy/openshift/namespace.yaml
```

## Troubleshooting

### Common Issues

**1. Pod fails to start due to security context:**

```bash
oc describe pod -l app=semantic-router -n vllm-semantic-router-system
```

**2. Storage issues:**

```bash
oc get pvc -n vllm-semantic-router-system
oc describe pvc semantic-router-models -n vllm-semantic-router-system
```

**3. Route not accessible:**

```bash
oc get routes -n vllm-semantic-router-system
oc describe route semantic-router-api -n vllm-semantic-router-system
```

### Resource Requirements

The deployment requires:

- **Memory**: 3Gi request, 6Gi limit
- **CPU**: 1 core request, 2 cores limit
- **Storage**: 10Gi for model storage

Adjust resource limits in `deployment.yaml` if needed for your cluster capacity.

## Files Overview

- `namespace.yaml` - Namespace with OpenShift-specific annotations
- `pvc.yaml` - Persistent volume claim for model storage
- `deployment.yaml` - Main application deployment with OpenShift security contexts
- `service.yaml` - Services for gRPC, HTTP API, and metrics
- `routes.yaml` - OpenShift routes for external access
- `config.yaml` - Application configuration
- `tools_db.json` - Tools database for semantic routing
- `kustomization.yaml` - Kustomize configuration for easy deployment
