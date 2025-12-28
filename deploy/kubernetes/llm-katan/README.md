# LLM Katan - Kubernetes Deployment

Comprehensive Kubernetes support for deploying LLM Katan in cloud-native environments.

## Overview

This directory provides production-ready Kubernetes manifests using Kustomize for deploying LLM Katan - a lightweight LLM server designed for testing and development workflows.

**Local Development:** This guide includes complete setup examples for both **kind** and **minikube** clusters, making it easy to run LLM Katan locally for development and testing.

## Architecture

### Pod Structure

Each deployment consists of two containers:

- **initContainer (model-downloader)**: Downloads models from HuggingFace to PVC
  - Image: `python:3.11-slim` (~45MB) with huggingface_hub installed on-the-fly
  - Checks if model exists before downloading
  - Runs once before main container starts

- **main container (llm-katan)**: Serves the LLM API
  - Image: `ghcr.io/vllm-project/semantic-router/llm-katan:latest` (~1.35GB)
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
- **Endpoints**: `/health`, `/v1/models`, `/v1/chat/completions`, `/metrics`

### Health Checks

- **Startup Probe**: 30s initial delay, 60 failures (15 min max startup)
- **Liveness Probe**: 15s delay, checks every 20s
- **Readiness Probe**: 5s delay, checks every 10s

## Directory Structure

```
deploy/kubernetes/llm-katan/
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
├── overlays/                      # Environment-specific configurations
│   ├── gpt35/                    # GPT-3.5-turbo simulation
│   │   └── kustomization.yaml    # Overlay with patches for gpt35
│   └── claude/                   # Claude-3-Haiku simulation
│       └── kustomization.yaml    # Overlay with patches for claude
│
├── README.md                      # This file - comprehensive deployment guide

└── verify-deployment.sh          # Automated verification script
```

## Prerequisites

Before starting, ensure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) - Container runtime
- **Local Kubernetes cluster** (choose one):
  - [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker (recommended for CI/CD)
  - [minikube](https://minikube.sigs.k8s.io/docs/start/) - Local Kubernetes (recommended for development)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- `kustomize` (built into kubectl 1.14+)

## Local Cluster Setup

This guide provides examples for both **kind** and **minikube** clusters. Choose the one that best fits your needs.

### Option 1: kind (Kubernetes in Docker)

**Installation:**

```bash
# Install kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Verify installation
kind version
```

**Create Cluster:**

```bash
# Create a basic cluster
kind create cluster --name llm-katan-test

# Verify cluster is running
kubectl cluster-info --context kind-llm-katan-test
kind get clusters
```

**Load Docker Image (Required):**

```bash
# Build the image first (if not already built)
docker build -t ghcr.io/vllm-project/semantic-router/llm-katan:latest -f Dockerfile .

# Load image into kind cluster
kind load docker-image ghcr.io/vllm-project/semantic-router/llm-katan:latest --name llm-katan-test
```

### Option 2: minikube

**Installation:**

```bash
# Download minikube
cd /tmp && curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64

# Install minikube
sudo install /tmp/minikube-linux-amd64 /usr/local/bin/minikube

# Verify installation
minikube version
```

**Start Cluster:**

```bash
# Start with recommended resources (16GB for running multiple instances)
minikube start --driver=docker --memory=16384 --cpus=4

# Verify cluster is running
minikube status
kubectl cluster-info
```

**Load Docker Image (Required):**

```bash
# Build the image first (if not already built)
docker build -t ghcr.io/vllm-project/semantic-router/llm-katan:latest -f Dockerfile .

# Load image into minikube
minikube image load ghcr.io/vllm-project/semantic-router/llm-katan:latest

# Verify image is loaded
minikube image ls | grep llm-katan
```

### Switching Between Clusters

If you have multiple clusters (kind, minikube, etc.), you need to select which one kubectl should use:

```bash
# List all contexts
kubectl config get-contexts

# Switch to kind
kubectl config use-context kind-llm-katan-test

# Switch to minikube
kubectl config use-context minikube

# Check current context
kubectl config current-context
```

The `*` symbol indicates the active context. All `kubectl` commands will target this cluster.

### Configuration

Environment variables are defined directly in `deployment.yaml`:

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

**GPU Support:**

LLM Katan is optimized for CPU workloads with tiny models. For GPU testing scenarios:

```yaml
# Add to deployment.yaml resources section
limits:
  nvidia.com/gpu: 1
```

**Note:** For production GPU deployments with larger models, use the main Semantic Router instead of LLM Katan.

### Storage

- **PVC Size**: 5Gi (adjust in overlays if needed)
- **Access Mode**: ReadWriteOnce
- **Mount Path**: `/cache/models/`
- **Purpose**: Cache downloaded models between restarts

## Complete Workflows

### Quick Start (Using Make)

Complete setup from scratch using make targets:

```bash
# 1. Create kind cluster (if using kind)
make create-cluster KIND_CLUSTER_NAME=llm-katan-test

# 2. Build and load Docker image
make docker-build-llm-katan
make kube-load-llm-katan-image KIND_CLUSTER_NAME=llm-katan-test

# 3. Deploy both models
make kube-deploy-llm-katan-multi

# 4. Check status
make kube-status-llm-katan

# 5. Test deployment
make kube-test-llm-katan

# 6. Access the service (in another terminal)
make kube-port-forward-llm-katan
# Then: curl http://localhost:8000/health
```

### Development Workflow

For iterative development and testing:

```bash
# Build and deploy
make docker-build-llm-katan
make kube-load-llm-katan-image
make kube-deploy-llm-katan-gpt35

# Make changes, rebuild, and redeploy
make docker-build-llm-katan
make kube-load-llm-katan-image
kubectl rollout restart deployment/llm-katan-gpt35 -n llm-katan-system

# View logs during testing
make kube-logs-llm-katan
```

### Testing Multiple Models

For testing routing between different LLM models:

```bash
# Deploy both models
make kube-deploy-llm-katan-multi

# Port-forward both (in separate terminals)
# Terminal 1:
make kube-port-forward-llm-katan LLM_KATAN_OVERLAY=gpt35 PORT=8000

# Terminal 2:
make kube-port-forward-llm-katan LLM_KATAN_OVERLAY=claude PORT=8001

# Test both endpoints
curl http://localhost:8000/v1/models  # GPT-3.5
curl http://localhost:8001/v1/models  # Claude
```

## Deployment Options

You have two main ways to deploy LLM Katan:

### Option A: Using Make Targets (Recommended)

**Best for:** Daily use, automation, simplified commands

See the [Complete Workflows](#complete-workflows) section above for step-by-step guides.

```bash
# Quick deployment
make kube-deploy-llm-katan-multi     # Deploy both models
make kube-status-llm-katan           # Check status
make kube-test-llm-katan             # Verify deployment
```

### Option B: Using kubectl Directly

**Best for:** Custom configurations, troubleshooting, learning Kubernetes

**Deploy from repository root:**

```bash
# Single model
kubectl apply -k deploy/kubernetes/llm-katan/overlays/gpt35

# Both models
kubectl apply -k deploy/kubernetes/llm-katan/overlays/gpt35
kubectl apply -k deploy/kubernetes/llm-katan/overlays/claude

# Verify
kubectl get all -n llm-katan-system
```

## Make Targets

All commands should be run from the repository root.

### Configuration Variables

The following environment variables can be used to customize the make targets:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_KATAN_OVERLAY` | `gpt35` | Overlay to deploy: `gpt35`, `claude`, or `all` (for undeploy) |
| `LLM_KATAN_NAMESPACE` | `llm-katan-system` | Kubernetes namespace for deployments |
| `LLM_KATAN_BASE_PATH` | `deploy/kubernetes/llm-katan` | Base path to Kubernetes manifests |
| `PORT` | `8000` | Local port for port-forwarding |
| `KIND_CLUSTER_NAME` | `semantic-router-cluster` | Kind cluster name |

### Deployment

```bash
# Deploy single overlay
make kube-deploy-llm-katan                         # Deploy with default overlay (gpt35)
make kube-deploy-llm-katan LLM_KATAN_OVERLAY=claude # Deploy with custom overlay

# Deploy specific overlays
make kube-deploy-llm-katan-gpt35                   # Deploy GPT-3.5 simulation
make kube-deploy-llm-katan-claude                  # Deploy Claude simulation

# Deploy multiple overlays
make kube-deploy-llm-katan-multi                   # Deploy both gpt35 and claude
```

### Status & Monitoring

```bash
# Show deployment status
make kube-status-llm-katan                         # Show all llm-katan resources

# View logs
make kube-logs-llm-katan                           # View logs (default: gpt35)
make kube-logs-llm-katan LLM_KATAN_OVERLAY=claude  # View Claude logs
```

### Testing & Debugging

```bash
# Test deployment
make kube-test-llm-katan                           # Test deployment (default: gpt35)
make kube-test-llm-katan LLM_KATAN_OVERLAY=claude  # Test Claude deployment

# Port forward for local access
make kube-port-forward-llm-katan                   # Port forward to localhost:8000 (gpt35)
make kube-port-forward-llm-katan LLM_KATAN_OVERLAY=claude # Port forward Claude
make kube-port-forward-llm-katan LLM_KATAN_OVERLAY=claude PORT=8001 # Custom port
```

### Image Management

```bash
# Build and load Docker images
make docker-build-llm-katan                        # Build llm-katan Docker image
make kube-load-llm-katan-image                     # Load image into kind cluster
```

### Cleanup

```bash
# Remove specific deployment
make kube-undeploy-llm-katan                       # Remove default overlay (gpt35)
make kube-undeploy-llm-katan LLM_KATAN_OVERLAY=gpt35   # Remove gpt35 deployment
make kube-undeploy-llm-katan LLM_KATAN_OVERLAY=claude  # Remove claude deployment

# Remove all deployments
make kube-undeploy-llm-katan LLM_KATAN_OVERLAY=all     # Remove all llm-katan deployments
```

### Help

```bash
# Show Kubernetes makefile help
make help-kube                                     # Display all available Kubernetes targets
```

## Direct kubectl Commands

### Deploy

```bash
# Deploy using kustomize overlays
kubectl apply -k deploy/kubernetes/llm-katan/overlays/gpt35
kubectl apply -k deploy/kubernetes/llm-katan/overlays/claude

# Deploy both
kubectl apply -k deploy/kubernetes/llm-katan/overlays/gpt35 && \
kubectl apply -k deploy/kubernetes/llm-katan/overlays/claude
```

### Status

```bash
# Get all resources
kubectl get all -n llm-katan-system

# Get pods
kubectl get pods -n llm-katan-system -o wide

# Get services
kubectl get svc -n llm-katan-system

# Get PVCs
kubectl get pvc -n llm-katan-system
```

### Logs

```bash
# View logs
kubectl logs -n llm-katan-system -l app=llm-katan-gpt35 -f
kubectl logs -n llm-katan-system -l app=llm-katan-claude -f

# View init container logs (model download)
kubectl logs -n llm-katan-system -l app=llm-katan-gpt35 -c model-downloader
```

### Port Forward

```bash
# Forward to localhost
kubectl port-forward -n llm-katan-system svc/llm-katan-gpt35 8000:8000
kubectl port-forward -n llm-katan-system svc/llm-katan-claude 8001:8000
```

### Testing

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Metrics
curl http://localhost:8000/metrics
```

### Cleanup

```bash
# Remove specific deployment
kubectl delete -k deploy/kubernetes/llm-katan/overlays/gpt35
kubectl delete -k deploy/kubernetes/llm-katan/overlays/claude

# Remove entire namespace
kubectl delete namespace llm-katan-system
```

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
# Don't forget to port-forward first
kubectl port-forward -n llm-katan-system svc/llm-katan 8000:8000

# Get metrics
curl http://localhost:8000/metrics
```

## Creating Custom Overlays

Want to add another model simulation Follow this pattern:

### Step 1: Create Overlay Directory

```bash
mkdir -p deploy/kubernetes/llm-katan/overlays/your-model
cd deploy/kubernetes/llm-katan/overlays/your-model
```

### Step 2: Create kustomization.yaml

**⚠️ Important**: Always add unique `app` label to prevent service selector conflicts!

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

metadata:
  name: llm-katan-your-model

resources:
  - ../../base

components:
  - ../../components/common

nameSuffix: -your-model

patches:
  - target:
      kind: Deployment
      name: llm-katan
    patch: |-
      # CRITICAL: Add unique app label for service selector
      - op: add
        path: /spec/selector/matchLabels/app
        value: "llm-katan-your-model"
      - op: add
        path: /spec/template/metadata/labels/app
        value: "llm-katan-your-model"
      # Set the served model name (what API returns)
      - op: add
        path: /spec/template/spec/containers/0/env/-
        value:
          name: YLLM_SERVED_MODEL_NAME
          value: "your-model-name"
      # Optional: Add descriptive label
      - op: add
        path: /spec/template/metadata/labels/model-alias
        value: "your-model"
  
  - target:
      kind: Service
      name: llm-katan
    patch: |-
      # CRITICAL: Match the unique app label from deployment
      - op: add
        path: /spec/selector/app
        value: "llm-katan-your-model"
      - op: add
        path: /metadata/labels/model-alias
        value: "your-model"
  
  # Update PVC reference to use suffixed name
  - target:
      kind: Deployment
      name: llm-katan
    patch: |-
      - op: replace
        path: /spec/template/spec/volumes/0/persistentVolumeClaim/claimName
        value: llm-katan-models-your-model
```

### Step 3: Deploy Your Custom Overlay

```bash
kubectl apply -k deploy/kubernetes/llm-katan/overlays/your-model/

# Verify
kubectl get pods,svc -n llm-katan-system
kubectl port-forward -n llm-katan-system svc/llm-katan-your-model 8002:8000
curl http://localhost:8002/v1/models
```

### Using Different Models

To use a different underlying model (not just different served name):

```yaml
patches:
  - target:
      kind: Deployment
      name: llm-katan
    patch: |-
      # ... existing patches ...
      # Change the actual model
      - op: replace
        path: /spec/template/spec/initContainers/0/env/0/value
        value: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
      - op: replace
        path: /spec/template/spec/containers/0/env/0/value
        value: "/cache/models/TinyLlama-1.1B-Chat-v1.0"
```

### Example: GPT-4 Simulator

```yaml
# overlays/gpt4/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

nameSuffix: -gpt4

patches:
  - target:
      kind: Deployment
      name: llm-katan
    patch: |-
      - op: add
        path: /spec/selector/matchLabels/app
        value: "llm-katan-gpt4"
      - op: add
        path: /spec/template/metadata/labels/app
        value: "llm-katan-gpt4"
      - op: add
        path: /spec/template/spec/containers/0/env/-
        value:
          name: YLLM_SERVED_MODEL_NAME
          value: "gpt-4"
  
  - target:
      kind: Service
      name: llm-katan
    patch: |-
      - op: add
        path: /spec/selector/app
        value: "llm-katan-gpt4"
```

## Best Practices

1. **Memory Allocation**: Allocate minimum 8Gi RAM for single instance, 16Gi for multi-model deployments
2. **Model Caching**: Keep PVCs to avoid re-downloading models (first deploy: 5-15 min, cached: 1-3 min)
3. **Cluster Selection**: Use `kind` for CI/CD and automated testing, `minikube` for local development with dashboard
4. **Iterative Testing**: Use `kubectl rollout restart` instead of redeploy for faster iterations (1-3 min vs 5-15 min)
5. **Tool Choice**: Use Make targets for simplified workflows, kubectl for fine-grained control and troubleshooting
6. **Debugging**: Watch pods with `-w` flag, check init container logs for download issues, use `describe pod` for events
7. **Production**: LLM Katan is for testing only - for production use `/deploy/helm/`, `/deploy/kubernetes/`, `/deploy/kserve/`, or `/deploy/openshift/`
8. **Security**: Deployments use non-root containers and enforce resource limits for secure operation
9. **Custom Overlays**: Always add unique `app` labels to prevent service selector conflicts in multi-instance deployments

## Advanced Integration

### Service Mesh Compatibility

LLM Katan deployments work with service mesh solutions like Istio and Linkerd:

**Automatic Features:**

- mTLS encryption between pods
- Traffic metrics and observability
- Automatic retries and circuit breakers
- Advanced load balancing

**Enable sidecar injection:**

```bash
# Label namespace for automatic injection
kubectl label namespace llm-katan-system istio-injection=enabled

# Redeploy to inject sidecars
kubectl rollout restart deployment -n llm-katan-system
```

**Note:** For production Semantic Router with service mesh, see `/deploy/kubernetes/istio/`

### Testing Semantic Router with LLM Katan

LLM Katan simulates LLM APIs (GPT, Claude) locally, enabling you to test Semantic Router **without API costs**.

**Use Case:** Test intelligent routing logic before deploying to production with real LLM APIs.

#### Step 1: Deploy LLM Katan

```bash
# Deploy both GPT-3.5 and Claude simulators
make kube-deploy-llm-katan-multi

# Verify services are running
kubectl get svc -n llm-katan-system
# NAME               TYPE        CLUSTER-IP      PORT(S)
# llm-katan-gpt35    ClusterIP   10.96.186.147   8000/TCP
# llm-katan-claude   ClusterIP   10.96.119.98    8000/TCP
```

#### Step 2: Configure Semantic Router

Update `config/config.yaml` to point to LLM Katan endpoints:

```yaml
# config/config.yaml

vllm_endpoints:
  - name: "gpt35-katan"
    address: "llm-katan-gpt35.llm-katan-system"  # Kubernetes DNS
    port: 8000
    weight: 1

  - name: "claude-katan"
    address: "llm-katan-claude.llm-katan-system"
    port: 8000
    weight: 1

model_config:
  "gpt-3.5-turbo":
    preferred_endpoints: ["gpt35-katan"]
  
  "claude-3-haiku-20240307":
    preferred_endpoints: ["claude-katan"]

categories:
  - name: coding
    utterances:
      - "write code"
      - "debug"
    model_scores:
      "gpt-3.5-turbo": 0.9
```

#### Step 3: Deploy and Test

```bash
# Deploy Semantic Router (using Helm)
helm install semantic-router deploy/helm/semantic-router \
  -f config/config.yaml

# Or run locally
make run-router

# Test routing (Semantic Router port-forward to 8080)
curl -X POST http://localhost:8080/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Write a Python function to sort a list",
    "stream": false
  }'

```

### Deployment Verification

Use the automated verification script:

```bash
# Run comprehensive deployment checks (default: llm-katan-system namespace)
./deploy/kubernetes/llm-katan/verify-deployment.sh

# Or specify namespace and service name
./deploy/kubernetes/llm-katan/verify-deployment.sh llm-katan-system llm-katan-gpt35
./deploy/kubernetes/llm-katan/verify-deployment.sh llm-katan-system llm-katan-claude
```

## Troubleshooting

### Common Issues

**Common pod error:**

- **OOMKilled (Exit Code 137)**: Pod exceeded memory limit during model loading
  - Solution for Minikube: Restart with more RAM: `minikube delete && minikube start --memory=16384 --cpus=4`
  - Solution for manifests: Increase memory in `deployment.yaml` (current: 6Gi)
- **ImagePullBackOff**: Image not available in cluster
  - For kind: `kind load docker-image ghcr.io/vllm-project/semantic-router/llm-katan:latest --name llm-katan-test`
  - For minikube: `minikube image load ghcr.io/vllm-project/semantic-router/llm-katan:latest`
- **Init:CrashLoopBackOff**: Model download failed
  - Check initContainer logs: `kubectl logs -n llm-katan-system <pod-name> -c model-downloader`

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

**LLM Katan not responding:**

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
kubectl delete -k deploy/kubernetes/llm-katan/overlays/gpt35/

# Remove claude instance
kubectl delete -k deploy/kubernetes/llm-katan/overlays/claude/
```

**Remove All llm-katan Resources:**

```bash
# Delete entire namespace (removes everything)
kubectl delete namespace llm-katan-system

# Or delete base deployment
kubectl delete -k deploy/kubernetes/llm-katan/base/```

**Cleanup Local Cluster:**

```bash
# For kind
kind delete cluster --name llm-katan-test
# Or if using default cluster name
kind delete cluster

# For minikube
minikube stop    # Stop the cluster (preserves state)
minikube delete  # Delete the cluster entirely
```

## CI/CD Integration

### GitHub Actions Example

Complete workflow with e2e tests:

```yaml
name: LLM Katan E2E Tests

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test-deployment:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install test dependencies
        run: pip install pytest requests
      
      - name: Create kind cluster
        run: make create-cluster KIND_CLUSTER_NAME=ci-test
      
      - name: Build and load Docker image
        run: |
          make docker-build-llm-katan
          make kube-load-llm-katan-image KIND_CLUSTER_NAME=ci-test
      
      - name: Deploy LLM Katan (both models)
        run: make kube-deploy-llm-katan-multi
      
      - name: Wait for deployments
        run: |
          make kube-test-llm-katan LLM_KATAN_OVERLAY=gpt35
          make kube-test-llm-katan LLM_KATAN_OVERLAY=claude
      
      - name: Run integration tests
        run: |
          # Port-forward in background
          kubectl port-forward -n llm-katan-system svc/llm-katan-gpt35 8000:8000 &
          kubectl port-forward -n llm-katan-system svc/llm-katan-claude 8001:8000 &
          sleep 5
          
          # Run e2e tests (if available)
          # pytest e2e/testing/ -v
          
          # Or simple health check
          curl -f http://localhost:8000/health
          curl -f http://localhost:8001/health
      
      - name: Show logs on failure
        if: failure()
        run: |
          kubectl get all -n llm-katan-system
          kubectl logs -n llm-katan-system -l app=llm-katan-gpt35 --tail=100
          kubectl logs -n llm-katan-system -l app=llm-katan-claude --tail=100
      
      - name: Cleanup
        if: always()
        run: |
          make kube-undeploy-llm-katan LLM_KATAN_OVERLAY=all
          make delete-cluster KIND_CLUSTER_NAME=ci-test
```

### GitLab CI Example

```yaml
test-llm-katan:
  stage: test
  script:
    - make create-cluster
    - make docker-build-llm-katan
    - make kube-load-llm-katan-image
    - make kube-deploy-llm-katan-multi
    - make kube-test-llm-katan
  after_script:
    - make delete-cluster

```

## Quick Reference

### Essential Make Commands (Recommended)

**From repository root:**

```bash
# Deployment
make kube-deploy-llm-katan-multi              # Deploy both models
make kube-deploy-llm-katan-gpt35              # Deploy GPT-3.5 only
make kube-deploy-llm-katan-claude             # Deploy Claude only

# Status & Logs
make kube-status-llm-katan                    # Show all resources
make kube-logs-llm-katan                      # View logs (gpt35)
make kube-logs-llm-katan LLM_KATAN_OVERLAY=claude

# Testing
make kube-test-llm-katan                      # Test gpt35
make kube-port-forward-llm-katan              # Access at localhost:8000

# Cleanup
make kube-undeploy-llm-katan LLM_KATAN_OVERLAY=gpt35
make kube-undeploy-llm-katan LLM_KATAN_OVERLAY=all
```

### Direct kubectl Commands (For Advanced Use)

**When you need more control:**

```bash
# Deploy
kubectl apply -k deploy/kubernetes/llm-katan/overlays/gpt35
kubectl apply -k deploy/kubernetes/llm-katan/overlays/claude

# Status
kubectl get all,pvc -n llm-katan-system
kubectl get pods -n llm-katan-system -o wide
kubectl describe pod -n llm-katan-system -l app=llm-katan-gpt35

# Logs
kubectl logs -n llm-katan-system -l app=llm-katan-gpt35 -f
kubectl logs -n llm-katan-system <pod-name> -c model-downloader  # Init container

# Port-forward
kubectl port-forward -n llm-katan-system svc/llm-katan-gpt35 8000:8000
kubectl port-forward -n llm-katan-system svc/llm-katan-claude 8001:8000

# Testing
kubectl exec -n llm-katan-system deployment/llm-katan-gpt35 -- curl localhost:8000/health

# Cleanup
kubectl delete -k deploy/kubernetes/llm-katan/overlays/gpt35
kubectl delete namespace llm-katan-system
```

### Resource Specifications

| Component | Value |
|-----------|-------|
| **Namespace** | `llm-katan-system` |
| **Service Port** | `8000` |
| **PVC Size** | `5Gi` |
| **CPU Request** | `1 core` |
| **CPU Limit** | `2 cores` |
| **Memory Request** | `3Gi` |
| **Memory Limit** | `6Gi` |
| **Startup Timeout** | `15 minutes` |

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/v1/models` | List available models |
| `/v1/chat/completions` | Chat completion (OpenAI compatible) |
| `/metrics` | Prometheus metrics |
