# vLLM Semantic Router - Single Namespace Deployment

This directory contains the OpenShift deployment configuration for vLLM Semantic Router using a **single-namespace architecture** where all components run in separate pods within the same namespace.

## Architecture Overview

```
Namespace: vllm-semantic-router
├── Pod: semantic-router (router + envoy containers)
├── Pod: model-a (llm-katan container)
└── Pod: model-b (llm-katan container)
```

### Components

1. **Router Pod** (`semantic-router` deployment)
   - Container: `semantic-router` - ExtProc server for routing logic
   - Container: `envoy-proxy` - Envoy proxy for HTTP handling
   - No GPU required
   - Handles: Classification, PII detection, jailbreak guard, semantic caching

2. **Model-A Pod** (`model-a` deployment)
   - Container: `model-a` - llm-katan serving Qwen/Qwen3-0.6B
   - Requires: 1 GPU (NVIDIA T4)
   - Service: `model-a` (ClusterIP)

3. **Model-B Pod** (`model-b` deployment)
   - Container: `model-b` - llm-katan serving Qwen/Qwen2.5-1.5B
   - Requires: 1 GPU (NVIDIA T4)
   - Service: `model-b` (ClusterIP)

## Files Structure

```
single-namespace/
├── README.md                    # This file
├── namespace.yaml               # Namespace definition
├── pvcs.yaml                    # PersistentVolumeClaims (router models/cache, model caches)
├── services.yaml                # Services (envoy-proxy, semantic-router, model-a, model-b)
├── imagestreams.yaml            # ImageStreams (semantic-router, llm-katan)
├── buildconfig-router.yaml      # BuildConfig for semantic-router
├── buildconfig-llm-katan.yaml   # BuildConfig for llm-katan
├── configmap-router.yaml        # Router configuration
├── configmap-envoy.yaml         # Envoy configuration
├── deployment-router.yaml       # Router + Envoy deployment
├── deployment-model-a.yaml      # Model-A deployment
├── deployment-model-b.yaml      # Model-B deployment
└── scripts/
    └── deploy-all.sh            # Automated deployment script
```

## Prerequisites

1. **OpenShift Cluster Access**
   - Logged in with `oc login`
   - Sufficient permissions to create namespaces, deployments, services, etc.

2. **GPU Nodes**
   - At least 2 GPU nodes with NVIDIA T4 GPUs
   - NVIDIA GPU Operator installed
   - Node Feature Discovery (NFD) installed

3. **Storage**
   - StorageClass `gp3-csi` available (or modify `pvcs.yaml`)
   - At least 35Gi total storage available

## Deployment

### Quick Start

```bash
cd /home/szedan/semantic-router/deploy/openshift/single-namespace
./scripts/deploy-all.sh
```

The script will:
1. Create namespace `vllm-semantic-router`
2. Create PVCs for storage
3. Create ConfigMaps for router and Envoy
4. Create Services
5. Create ImageStreams and BuildConfigs
6. Start container image builds
7. Deploy all components
8. Monitor deployment progress

### Manual Deployment

If you prefer step-by-step deployment:

```bash
# 1. Create namespace
oc apply -f namespace.yaml

# 2. Create PVCs
oc apply -f pvcs.yaml

# 3. Create ConfigMaps
oc apply -f configmap-router.yaml
oc apply -f configmap-envoy.yaml

# 4. Create Services
oc apply -f services.yaml

# 5. Create ImageStreams
oc apply -f imagestreams.yaml

# 6. Create BuildConfigs
oc apply -f buildconfig-router.yaml
oc apply -f buildconfig-llm-katan.yaml

# 7. Start builds
cd /home/szedan/semantic-router
oc start-build semantic-router --from-dir=. --follow -n vllm-semantic-router
oc start-build llm-katan -n vllm-semantic-router

# 8. Deploy applications (after builds complete)
oc apply -f deployment-router.yaml
oc apply -f deployment-model-a.yaml
oc apply -f deployment-model-b.yaml
```

## Monitoring

### Check Deployment Status

```bash
# All resources
oc get all -n vllm-semantic-router

# Pods
oc get pods -n vllm-semantic-router -o wide

# Services
oc get svc -n vllm-semantic-router

# Builds
oc get builds -n vllm-semantic-router
```

### Check Logs

```bash
# Router logs
oc logs -f deployment/semantic-router -c semantic-router -n vllm-semantic-router

# Envoy logs
oc logs -f deployment/semantic-router -c envoy-proxy -n vllm-semantic-router

# Model-A logs
oc logs -f deployment/model-a -n vllm-semantic-router

# Model-B logs
oc logs -f deployment/model-b -n vllm-semantic-router

# Build logs
oc logs -f build/semantic-router-1 -n vllm-semantic-router
oc logs -f build/llm-katan-1 -n vllm-semantic-router
```

### Check Resource Usage

```bash
# Pod resource usage
oc adm top pods -n vllm-semantic-router

# Node resource usage
oc adm top nodes
```

## Testing

### Expose Service (if needed)

```bash
# Create route to access from outside cluster
oc expose svc/envoy-proxy -n vllm-semantic-router

# Get route URL
oc get route -n vllm-semantic-router
```

### Send Test Request

```bash
# Get route hostname
ROUTE_HOST=$(oc get route envoy-proxy -n vllm-semantic-router -o jsonpath='{.spec.host}')

# Test request
curl -X POST "http://${ROUTE_HOST}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

### Port Forward (for local testing)

```bash
# Forward Envoy port to localhost
oc port-forward svc/envoy-proxy 8801:8801 -n vllm-semantic-router

# In another terminal, test locally
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

## Configuration

### Router Configuration

Edit `configmap-router.yaml` to modify:
- **vLLM endpoints**: Model service addresses (currently `model-a` and `model-b`)
- **Categories**: Classification categories and system prompts
- **Classifiers**: Category, PII, and jailbreak classifier settings
- **Semantic cache**: Cache backend, thresholds, and TTL
- **Tools**: Tool selection settings

After editing, apply changes:
```bash
oc apply -f configmap-router.yaml
oc rollout restart deployment/semantic-router -n vllm-semantic-router
```

### Envoy Configuration

Edit `configmap-envoy.yaml` to modify:
- **Timeouts**: Request and connection timeouts
- **Routes**: HTTP routing rules
- **Clusters**: Upstream service definitions

After editing, apply changes:
```bash
oc apply -f configmap-envoy.yaml
oc rollout restart deployment/semantic-router -n vllm-semantic-router
```

### Model Configuration

To change the models served:

1. Edit `deployment-model-a.yaml` or `deployment-model-b.yaml`
2. Modify the `--model` argument to use a different model
3. Apply changes:
   ```bash
   oc apply -f deployment-model-a.yaml
   # or
   oc apply -f deployment-model-b.yaml
   ```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
oc describe pod <pod-name> -n vllm-semantic-router

# Check resource availability
oc describe nodes | grep -A 5 "Allocated resources"
```

### Image Pull Errors

```bash
# Check build status
oc get builds -n vllm-semantic-router

# Rebuild if needed
oc start-build semantic-router --from-dir=/home/szedan/semantic-router -n vllm-semantic-router
oc start-build llm-katan -n vllm-semantic-router
```

### Router CrashLoopBackOff

Common causes:
1. **Models not downloaded**: Check init container logs
2. **Invalid config**: Check router container logs
3. **PVC not mounted**: Check PVC status

```bash
# Check init container logs
oc logs deployment/semantic-router -c model-downloader -n vllm-semantic-router

# Check router logs
oc logs deployment/semantic-router -c semantic-router -n vllm-semantic-router --tail=100

# Check PVC status
oc get pvc -n vllm-semantic-router
```

### Model Pods Pending

Check GPU availability:
```bash
# Check GPU node labels
oc get nodes --show-labels | grep gpu

# Check GPU allocation
oc describe nodes | grep -A 10 "nvidia.com/gpu"
```

### Service Communication Issues

Test service connectivity:
```bash
# From within router pod
oc exec -it deployment/semantic-router -c semantic-router -n vllm-semantic-router -- curl http://model-a:8000/
oc exec -it deployment/semantic-router -c semantic-router -n vllm-semantic-router -- curl http://model-b:8000/
```

## Cleanup

To remove the entire deployment:

```bash
# Delete namespace (removes everything)
oc delete namespace vllm-semantic-router

# Or delete resources individually
oc delete -f deployment-router.yaml
oc delete -f deployment-model-a.yaml
oc delete -f deployment-model-b.yaml
oc delete -f services.yaml
oc delete -f pvcs.yaml
oc delete -f configmap-router.yaml
oc delete -f configmap-envoy.yaml
oc delete -f imagestreams.yaml
oc delete -f buildconfig-router.yaml
oc delete -f buildconfig-llm-katan.yaml
oc delete -f namespace.yaml
```

## Resource Requirements

### Router Pod
- CPU: 1-2 cores
- Memory: 3-6 GB
- Storage: 15 Gi (models + cache)

### Model-A Pod
- CPU: 0.5-1 core
- Memory: 2-4 GB
- GPU: 1x NVIDIA T4
- Storage: 10 Gi (cache)

### Model-B Pod
- CPU: 0.5-1 core
- Memory: 2-4 GB
- GPU: 1x NVIDIA T4
- Storage: 10 Gi (cache)

### Total
- CPU: 2-4 cores
- Memory: 7-14 GB
- GPU: 2x NVIDIA T4
- Storage: 35 Gi

## Comparison with Multi-Namespace Deployment

| Aspect | Single-Namespace | Multi-Namespace |
|--------|-----------------|-----------------|
| **Namespace** | 1 namespace | 3 namespaces |
| **Service Discovery** | Short names (`model-a`) | FQDN (`model-a.vllm-model-a.svc.cluster.local`) |
| **Network Policies** | Optional | Required for cross-namespace |
| **RBAC** | Simpler | More complex |
| **Isolation** | Pod-level | Namespace-level |
| **Complexity** | Lower | Higher |
| **Use Case** | Development, testing, simple deployments | Production, multi-tenant, strict isolation |

## Next Steps

1. **Configure monitoring**: Set up Prometheus and Grafana dashboards
2. **Enable tracing**: Configure OpenTelemetry for distributed tracing
3. **Add autoscaling**: Configure HPA for router and model pods
4. **Setup alerts**: Configure alerting for pod failures and resource usage
5. **Optimize resources**: Adjust resource requests/limits based on actual usage

## Support

For issues and questions:
- GitHub Issues: https://github.com/sdan/semantic-router/issues
- Documentation: https://vllm-semantic-router.com
