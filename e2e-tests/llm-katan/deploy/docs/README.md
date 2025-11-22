# LLM Katan - Kubernetes Deployment

Comprehensive Kubernetes support for deploying LLM Katan in cloud-native environments.

## Overview

This directory provides production-ready Kubernetes manifests using Kustomize for deploying LLM Katan - a lightweight LLM server designed for testing and development workflows.

## Architecture

### Pod Structure
Each deployment consists of two containers:
- **initContainer (model-downloader)**: Downloads models from HuggingFace to PVC
  - Image: `python:3.11-slim` (~45MB)
  - Checks if model exists before downloading
  - Runs once before main container starts
  
- **main container (llm-katan)**: Serves the LLM API
  - Image: `llm-katan:latest` (~1.35GB)
  - Loads model from PVC cache
  - Exposes OpenAI-compatible API on port 8000

### Storage
- **PersistentVolumeClaim**: 5Gi for model caching
- **Mount Path**: `/cache/models/`
- **Access Mode**: ReadWriteOnce (single Pod write)
- Models persist across Pod restarts

### Namespace
All resources deploy to the `llm-katan-system` namespace. Each overlay creates isolated instances within this namespace:
- **gpt35**: Simulates GPT-3.5-turbo
- **claude**: Simulates Claude-3-Haiku

### Resource Naming
Kustomize applies `nameSuffix` to avoid conflicts:
- Base: `llm-katan`
- gpt35 overlay: `llm-katan-gpt35` (via `nameSuffix: -gpt35`)
- claude overlay: `llm-katan-claude` (via `nameSuffix: -claude`)

**How it works:**
```yaml
# overlays/gpt35/kustomization.yaml
nameSuffix: -gpt35  # Automatically appends to all resource names
```

This creates unique resource names for each overlay without manual patches, allowing multiple instances to coexist in the same namespace.

### Networking
- **Service Type**: ClusterIP (internal only)
- **Port**: 8000 (HTTP)
- **Endpoints**: `/health`, `/v1/models`, `/v1/chat/completions`

### Health Checks
- **Startup Probe**: 30s initial delay, 60 failures (15 min max startup)
- **Liveness Probe**: 15s delay, checks every 20s
- **Readiness Probe**: 5s delay, checks every 10s

## Directory Structure

kubernetes/
├── base/                          # Base Kubernetes manifests
│   ├── namespace.yaml            # llm-katan-system namespace
│   ├── deployment.yaml           # Main deployment with health checks
│   ├── service.yaml              # ClusterIP service (port 8000)
│   ├── pvc.yaml                  # Model cache storage (5Gi)
│   └── kustomization.yaml        # Base kustomization
│
├── components/                    # Reusable Kustomize components
│   └── common/                   # Common labels for all resources
│       └── kustomization.yaml    # Shared label definitions
│
└── overlays/                      # Environment-specific configurations
    ├── gpt35/                    # GPT-3.5-turbo simulation
    │   └── kustomization.yaml    # Overlay with patches for gpt35
    │
    └── claude/                   # Claude-3-Haiku simulation
        └── kustomization.yaml    # Overlay with patches for claude

## Prerequisites

Before starting, ensure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) - Container runtime
- [minikube](https://minikube.sigs.k8s.io/docs/start/) - Local Kubernetes 
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- `kustomize` (built into kubectl 1.14+)


## Configuration

### Environment Variables

Configure via `config.env` or overlay ConfigMaps:

| Variable | Default | Description |
|----------|---------|-------------|
| `YLLM_MODEL` | `Qwen/Qwen3-0.6B` | HuggingFace model to load |
| `YLLM_SERVED_MODEL_NAME` | (empty) | Model name for API (defaults to YLLM_MODEL) |
| `YLLM_BACKEND` | `transformers` | Backend: `transformers` or `vllm` |
| `YLLM_HOST` | `0.0.0.0` | Server bind address |
| `YLLM_PORT` | `8000` | Server port |

### Resource Limits

Default per instance:

```yaml
resources:
  requests:
    cpu: "1"
    memory: "3Gi"
  limits:
    cpu: "2"
    memory: "6Gi"
```

### Storage

- **PVC Size**: 5Gi (adjust in overlays if needed)
- **Access Mode**: ReadWriteOnce
- **Mount Path**: `/cache/models/`
- **Purpose**: Cache downloaded models between restarts

### Deploy Single Instance (Base)

```bash
# From repository root
cd e2e-tests/llm-katan/deploy/kubernetes

# Deploy with default settings
kubectl apply -k base/

# Check status
kubectl get pods -n llm-katan-system
kubectl logs -n llm-katan-system -l app=llm-katan -f

# Test the deployment
kubectl port-forward -n llm-katan-system svc/llm-katan 8000:8000
curl http://localhost:8000/health
```

### Deploy Multi-Instance (Overlays)

```bash
# Deploy GPT-3.5-turbo simulation
kubectl apply -k overlays/gpt35/

# Deploy Claude-3-Haiku simulation
kubectl apply -k overlays/claude/

# Or deploy both simultaneously
kubectl apply -k overlays/gpt35/ && kubectl apply -k overlays/claude/

# Verify both are running
kubectl get pods -n llm-katan-system
kubectl get svc -n llm-katan-system


## Testing & Verification

### Health Check

```bash
kubectl port-forward -n llm-katan-system svc/llm-katan 8000:8000
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","model":"Qwen/Qwen3-0.6B","backend":"transformers"}
```

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Models Endpoint

```bash
curl http://localhost:8000/v1/models
```

### Metrics (Prometheus)

```bash
dont forget -> kubectl port-forward -n llm-katan-system svc/llm-katan 8000:8000
curl http://localhost:8000/metrics

# Metrics exposed:
# - llm_katan_requests_total
# - llm_katan_tokens_generated_total
# - llm_katan_response_time_seconds
# - llm_katan_uptime_seconds
```

## Troubleshooting

### Common Issues

**Common pod error:**

  - OOMKilled: Increase memory limits (current: 6Gi)
  - ImagePullBackOff: Load image into Kind with kind load docker-image llm-katan:latest
  - Init:CrashLoopBackOff: Check initContainer logs for download issues

**Pod not starting:**

```bash
# Check pod status
kubectl get pods -n llm-katan-system

# Describe pod for events
kubectl describe pod -n llm-katan-system -l app.kubernetes.io/name=llm-katan

# Check initContainer logs (model download)
kubectl logs -n llm-katan-system -l app.kubernetes.io/name=llm-katan -c model-downloader

# Check main container logs
kubectl logs -n llm-katan-system -l app.kubernetes.io/name=llm-katan -c llm-katan -f
```

**LLM Katan not responding::**

```bash
# Check deployment status
kubectl get deployment -n llm-katan-system

# Check service
kubectl get svc -n llm-katan-system

# Check if port-forward is active
ps aux | grep "port-forward" | grep llm-katan

# Test health endpoint
kubectl port-forward -n llm-katan-system svc/llm-katan-gpt35 8000:8000 &
curl http://localhost:8000/health
```

**PVC issues:**

```bash
# Check PVC status
kubectl get pvc -n llm-katan-system

# Check PVC details
kubectl describe pvc -n llm-katan-system

# Check volume contents (if pod is running)
kubectl exec -n llm-katan-system <pod-name> -- ls -lah /cache/models/
```

## Cleanup

**Remove Specific Overlay:**

```bash
# Remove gpt35 instance
kubectl delete -k e2e-tests/llm-katan/deploy/kubernetes/overlays/gpt35/

# Remove claude instance
kubectl delete -k e2e-tests/llm-katan/deploy/kubernetes/overlays/claude/
```

**Remove All llm-katan Resources:**

```bash
# Delete entire namespace (removes everything)
kubectl delete namespace llm-katan-system

# Or delete base deployment
kubectl delete -k e2e-tests/llm-katan/deploy/kubernetes/base/
```

**Cleanup Kind Cluster:**

```bash
# Stop Kind cluster
kind delete cluster --name llm-katan-test

# Or if using default cluster name
kind delete cluster
```
