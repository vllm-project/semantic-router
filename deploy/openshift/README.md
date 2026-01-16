# OpenShift Deployment for Semantic Router

This directory contains OpenShift-specific deployment manifests for the vLLM Semantic Router with **dynamic IP configuration** for cross-cluster portability.

## Quick Deployment

### Prerequisites

- OpenShift cluster access
- `oc` CLI tool configured and logged in
- Cluster admin privileges (or permissions to create namespaces and routes)
- Local source code (for dashboard build)

### One-Click Full Deployment (Recommended)

Deploy the complete stack including semantic-router, vLLM models, and all observability components:

```bash
cd deploy/openshift
./deploy-to-openshift.sh
```

This script will deploy:

**Core Components:**

- ✅ Build the llm-katan image from Dockerfile
- ✅ Create namespace and PVCs
- ✅ Deploy vLLM model services (Model-A and Model-B)
- ✅ Auto-discover Kubernetes service ClusterIPs
- ✅ Generate configuration with actual IPs (portable across clusters)
- ✅ Deploy semantic-router with Envoy proxy sidecar
- ✅ Create OpenShift routes for external access

**Observability Stack:**

- ✅ Dashboard (built from local source with PlaygroundPage fix)
- ✅ OpenWebUI playground for testing models
- ✅ Grafana for metrics visualization
- ✅ Prometheus for metrics collection

### Minimal Deployment (Core Only)

If you only want the core semantic-router and vLLM models without observability:

```bash
cd deploy/openshift
./deploy-to-openshift.sh --no-observability
```

This deploys only the core components without Dashboard, OpenWebUI, Grafana, and Prometheus.

### Command Line Options

| Flag | Description |
|------|-------------|
| `--no-observability` | Skip deploying Dashboard, OpenWebUI, Grafana, and Prometheus |
| `--help`, `-h` | Show help message |

### Manual Deployment (Advanced)

If you prefer manual deployment or need to customize:

1. **Create namespace:**

   ```bash
   oc create namespace vllm-semantic-router-system
   ```

2. **Build llm-katan image:**

   ```bash
   oc new-build --dockerfile - --name llm-katan -n vllm-semantic-router-system < Dockerfile.llm-katan
   ```

3. **Deploy resources:**

   ```bash
   oc apply -f deployment.yaml -n vllm-semantic-router-system
   ```

4. **Note:** You'll need to manually configure ClusterIPs in `config-openshift.yaml`

## How Dashboard Build Works

The deployment script uses OpenShift's **binary build** approach for the dashboard:

1. Creates a BuildConfig with Docker strategy
2. Uploads the local `dashboard/` directory as build source
3. Builds the image inside OpenShift (no local Docker required)
4. Pushes to OpenShift internal registry
5. Deploys using the built image

### Why Binary Build?

- ✅ No local Docker daemon required
- ✅ Works on any machine with `oc` CLI
- ✅ Builds with your local code changes (including PlaygroundPage fix)
- ✅ Automatically integrated with OpenShift registry
- ✅ Works across different OpenShift clusters

### Updating Dashboard

If you make changes to the dashboard code, rebuild and redeploy:

```bash
# Rebuild dashboard image from local source
cd dashboard
oc start-build dashboard-custom --from-dir=. --follow -n vllm-semantic-router-system

# Restart deployment to use new image
oc rollout restart deployment/dashboard -n vllm-semantic-router-system
```

## Accessing Services

After deployment, the script will display URLs for all services. Routes are automatically generated with cluster-appropriate hostnames.

### Get Route URLs

```bash
# Core Services
oc get route semantic-router-api -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
oc get route semantic-router-grpc -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
oc get route semantic-router-metrics -n vllm-semantic-router-system -o jsonpath='{.spec.host}'

# Observability (if deployed)
oc get route dashboard -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
oc get route grafana -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
oc get route prometheus -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
```

### Example Usage

```bash
# Get the API route
API_ROUTE=$(oc get route semantic-router-api -n vllm-semantic-router-system -o jsonpath='{.spec.host}')

# Test health endpoint
curl -k https://$API_ROUTE/health

# Test classification
curl -k -X POST https://$API_ROUTE/api/v1/classify/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?"}'

# Get the Envoy route (for chat completions endpoint)
ENVOY_ROUTE=$(oc get route envoy-http -n vllm-semantic-router-system -o jsonpath='{.spec.host}')

# Test auto routing (hits a model backend via Envoy)
curl -k -X POST https://$ENVOY_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}'
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

### Quick Cleanup

Remove the entire namespace and all resources (recommended):

```bash
cd deploy/openshift
./cleanup-openshift.sh
```

If not already logged in to OpenShift:

```bash
oc login <your-cluster-url>
./cleanup-openshift.sh
```

### Cleanup Options

The cleanup script supports different cleanup levels:

| Level | What Gets Deleted | What's Preserved |
|-------|------------------|------------------|
| `deployment` | Deployments, services, routes, configmaps, buildconfigs | Namespace, PVCs |
| `namespace` (default) | Entire namespace and all resources | Nothing |
| `all` | Namespace + cluster-wide resources | Nothing |

**Examples:**

```bash
# Remove everything (default)
./cleanup-openshift.sh

# Keep namespace and PVCs, remove only deployments
./cleanup-openshift.sh --level deployment

# Dry run to see what would be deleted
./cleanup-openshift.sh --dry-run

# Force cleanup without confirmation
./cleanup-openshift.sh --force
```

### What Gets Cleaned Up

The cleanup script removes:

**Core Components:**

- semantic-router deployment
- vLLM model deployments (Model-A, Model-B)
- All services and routes
- ConfigMaps (router config, envoy config)
- BuildConfigs and ImageStreams (llm-katan, dashboard-custom)

**Observability Stack:**

- Dashboard deployment
- OpenWebUI deployment
- Grafana deployment
- Prometheus deployment
- All related services, routes, and configmaps

**Storage (namespace level only):**

- PVCs for models and cache

### Manual Cleanup

If you prefer manual cleanup:

```bash
# Delete entire namespace (removes everything)
oc delete namespace vllm-semantic-router-system

# Or delete specific components
oc delete deployment,service,route,configmap,buildconfig,imagestream --all -n vllm-semantic-router-system
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
