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
| `--kserve` | Deploy semantic-router with a KServe backend (add `--simulator` for KServe simulator) |
| *(auto)* | If the KServe CRD is missing, the script installs upstream KServe and cert-manager |
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
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain the elements of a contract under common law and give a simple example."}]}'

curl -k -X POST https://$ENVOY_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

### KServe Mode Routes

When using `--kserve`, routes are created with different names:

```bash
# Classification API route
API_ROUTE=$(oc get route semantic-router-kserve-api -n vllm-semantic-router-system -o jsonpath='{.spec.host}')

# Envoy route for chat completions
ENVOY_ROUTE=$(oc get route semantic-router-kserve -n vllm-semantic-router-system -o jsonpath='{.spec.host}')

curl -k https://$API_ROUTE/health

curl -k -X POST https://$API_ROUTE/api/v1/classify/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?"}'

curl -k -X POST https://$ENVOY_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain the elements of a contract under common law and give a simple example."}]}'

curl -k -X POST https://$ENVOY_ROUTE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

### KServe Deployment Steps (Summary)

```bash
# 1) Install KServe + LLMInferenceService CRDs (one-time)
./deploy/kserve/install-kserve.sh

# 2) (GPU only) Install NVIDIA GPU Operator
./deploy/kserve/install-gpu-operator.sh

# 3) Deploy KServe simulator (CPU) or GPU model (run from repo root)
./deploy/openshift/deploy-to-openshift.sh --kserve --simulator --no-observability

# GPU (Qwen 0.6B)
./deploy/openshift/deploy-to-openshift.sh --kserve --no-observability

# 4) Test via Envoy route
ENVOY_ROUTE=$(oc get route semantic-router-kserve -n vllm-semantic-router-system -o jsonpath='{.spec.host}')
curl -sk "https://${ENVOY_ROUTE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain the elements of a contract under common law and give a simple example."}]}'

curl -sk "https://${ENVOY_ROUTE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

To run the KServe simulator with two models (Model-A and Model-B). Simulator mode uses LLMInferenceServices and downloads a small HF model (opt-125m) to satisfy the LLMISVC `model.uri`:

```bash
./deploy/openshift/deploy-to-openshift.sh --kserve --simulator --no-observability
```

### KServe with GPU (Real Model Deployment)

To deploy a real LLM (Qwen 0.6B) on GPU with KServe:

**Step 1: Install NVIDIA GPU Operator (one-time setup)**

```bash
# Automated installation
./deploy/kserve/install-gpu-operator.sh

# Verify GPU resources are available (wait 5-10 minutes after install)
oc get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"
```

**Step 2: Deploy with GPU**

```bash
./deploy/openshift/deploy-to-openshift.sh --kserve --no-observability
```

This deploys:

- Qwen 0.6B model via LLMInferenceService on GPU node
- vLLM serving backend on port 8000
- Semantic-router with Envoy proxy

**Step 3: Test the deployment**

```bash
# Get route URL
ENVOY_ROUTE=$(oc get route semantic-router-kserve -n vllm-semantic-router-system -o jsonpath='{.spec.host}')

# Test auto-routing through vSR (single-model setups will route to Qwen)
curl -sk "https://${ENVOY_ROUTE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain the elements of a contract under common law and give a simple example."}]}'

# (Optional) Force the backend model directly
curl -sk "https://${ENVOY_ROUTE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"What is 2+2?"}]}'

# Confirm which backend was selected in vSR logs
oc logs -l app=semantic-router -c semantic-router -n vllm-semantic-router-system --tail=100 | rg -i "selected|route|model"
```

**Notes:**

- The deploy script grants the `privileged` SCC to the `default` service account in `vllm-semantic-router-system` to allow the storage-initializer to download models.
- If a restart hangs with `Multi-Attach` on PVCs, scale down the router first:
  `oc scale deployment/semantic-router-kserve -n vllm-semantic-router-system --replicas=0`

**Supported GPU Instance Types:**

| Instance | GPU | Memory | Recommended Models |
|----------|-----|--------|-------------------|
| g4dn.xlarge | T4 | 16GB | Models up to 7B |
| g5.xlarge | A10G | 24GB | Models up to 13B |
| g6.xlarge | L4 | 24GB | Models up to 13B |

For custom models, create your own LLMInferenceService manifest based on `deploy/kserve/inference-examples/inferenceservice-qwen-0.6b-gpu.yaml`.

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

## GPU Support

### Prerequisites for GPU Workloads

To run GPU-accelerated LLM inference (e.g., LLM-Katan with vLLM backend), you need:

1. **GPU-enabled nodes** (e.g., AWS g4dn, g5, g6 instance types)
2. **Node Feature Discovery (NFD) Operator** - Detects hardware features including GPUs
3. **NVIDIA GPU Operator** - Installs drivers, device plugin, and container toolkit

### Installing NVIDIA GPU Operator (OpenShift Console)

#### Step 1: Install Node Feature Discovery (NFD) Operator

1. Navigate to **Operators → OperatorHub**
2. Search for **"Node Feature Discovery"**
3. Select the **Red Hat** version (not community)
4. Click **Install** with defaults:
   - Update channel: `stable`
   - Installation mode: `All namespaces`
   - Installed Namespace: `openshift-nfd`
5. Wait for operator to install (Status: Succeeded)

#### Step 2: Create NFD Instance

1. Navigate to **Operators → Installed Operators → Node Feature Discovery**
2. Click on **NodeFeatureDiscovery** tab
3. Click **Create NodeFeatureDiscovery**
4. Accept defaults and click **Create**
5. Wait for NFD pods to be running:

   ```bash
   oc get pods -n openshift-nfd
   ```

#### Step 3: Install NVIDIA GPU Operator

1. Navigate to **Operators → OperatorHub**
2. Search for **"gpu"** and select **"NVIDIA GPU Operator"** (Certified)
3. Click **Install** with defaults:
   - Update channel: `stable`
   - Installation mode: `All namespaces`
   - Installed Namespace: `nvidia-gpu-operator`
4. Wait for operator to install

#### Step 4: Create ClusterPolicy

1. Navigate to **Operators → Installed Operators → NVIDIA GPU Operator**
2. Click on **ClusterPolicy** tab
3. Click **Create ClusterPolicy**
4. Accept defaults (or customize driver version if needed)
5. Click **Create**

#### Step 5: Verify GPU Resources

Wait for all GPU operator pods to be ready (this can take 5-10 minutes as drivers are compiled):

```bash
# Check GPU operator pods
oc get pods -n nvidia-gpu-operator

# Verify GPU resources are exposed on nodes
oc get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"

# Should show:
# NAME                          GPU
# ip-10-0-1-84.ec2.internal     1
```

### Installing via CLI (Recommended)

Use the provided installer script for automated installation:

```bash
# From the repository root
./deploy/kserve/install-gpu-operator.sh
```

This script will:

1. Install the NFD Operator and create an NFD instance
2. Install the NVIDIA GPU Operator
3. Create a ClusterPolicy to enable GPU support
4. Wait for all components to be ready
5. Verify GPU resources are available on nodes

**Manual CLI installation** (if you prefer step-by-step):

```bash
# 1. Install NFD Operator
cat <<EOF | oc apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: openshift-nfd
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: openshift-nfd
  namespace: openshift-nfd
spec:
  targetNamespaces:
  - openshift-nfd
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: nfd
  namespace: openshift-nfd
spec:
  channel: stable
  name: nfd
  source: redhat-operators
  sourceNamespace: openshift-marketplace
EOF

# Wait for NFD operator
oc wait --for=condition=Available deployment -l app.kubernetes.io/name=node-feature-discovery-operator -n openshift-nfd --timeout=5m

# 2. Create NFD instance
cat <<EOF | oc apply -f -
apiVersion: nfd.openshift.io/v1
kind: NodeFeatureDiscovery
metadata:
  name: nfd-instance
  namespace: openshift-nfd
spec:
  operand:
    image: registry.redhat.io/openshift4/ose-node-feature-discovery-rhel9:v4.16
    servicePort: 12000
  workerConfig:
    configData: |
      sources:
        pci:
          deviceClassWhitelist:
            - "0200"
            - "03"
            - "12"
          deviceLabelFields:
            - vendor
EOF

# 3. Install NVIDIA GPU Operator
cat <<EOF | oc apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: nvidia-gpu-operator
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: nvidia-gpu-operator
  namespace: nvidia-gpu-operator
spec:
  targetNamespaces:
  - nvidia-gpu-operator
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: gpu-operator-certified
  namespace: nvidia-gpu-operator
spec:
  channel: stable
  name: gpu-operator-certified
  source: certified-operators
  sourceNamespace: openshift-marketplace
EOF

# Wait for GPU operator
oc wait --for=condition=Available deployment -l app.kubernetes.io/component=gpu-operator -n nvidia-gpu-operator --timeout=5m

# 4. Create ClusterPolicy (after operator is ready)
cat <<EOF | oc apply -f -
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: gpu-cluster-policy
spec:
  operator:
    defaultRuntime: crio
  driver:
    enabled: true
  toolkit:
    enabled: true
  devicePlugin:
    enabled: true
  dcgm:
    enabled: true
  dcgmExporter:
    enabled: true
  gfd:
    enabled: true
  migManager:
    enabled: true
  nodeStatusExporter:
    enabled: true
  mig:
    strategy: single
EOF
```

### Deploying LLM-Katan with GPU

Once the GPU operator is installed and GPU resources are visible:

```bash
# 1. Grant anyuid SCC (required for LLM-Katan's init container)
oc adm policy add-scc-to-user anyuid -z default -n llm-katan-system

# 2. Deploy LLM-Katan base
kubectl apply -k deploy/kubernetes/llm-katan/base

# 3. Add GPU resource limit
oc patch deployment llm-katan -n llm-katan-system --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/resources/limits/nvidia.com~1gpu", "value": "1"}
]'

# 4. Verify pod is scheduled on GPU node
oc get pods -n llm-katan-system -o wide
```

**Note:** The default LLM-Katan image uses the `transformers` backend which runs on CPU.
For GPU acceleration with vLLM, ensure the image includes vLLM dependencies:

```bash
# Switch to vLLM backend (requires vLLM-enabled image)
oc set env deployment/llm-katan -n llm-katan-system YLLM_BACKEND=vllm
```

### GPU Instance Types (AWS)

| Instance Type | GPU | GPU Memory | Use Case |
|--------------|-----|------------|----------|
| g4dn.xlarge | 1x T4 | 16 GB | Small models (< 7B) |
| g4dn.2xlarge | 1x T4 | 16 GB | Small models with more CPU |
| g5.xlarge | 1x A10G | 24 GB | Medium models (7B-13B) |
| g5.2xlarge | 1x A10G | 24 GB | Medium models with more CPU |
| g6.xlarge | 1x L4 | 24 GB | Medium models, latest gen |
| g6.2xlarge | 1x L4 | 24 GB | Medium models with more CPU |
| p4d.24xlarge | 8x A100 | 320 GB | Large models (70B+) |

### Troubleshooting GPU Issues

**GPU not showing in node allocatable:**

```bash
# Check if NFD detected GPU
oc get node <node-name> -o yaml | grep -i nvidia

# Check GPU operator pods
oc get pods -n nvidia-gpu-operator

# Check driver pod logs
oc logs -n nvidia-gpu-operator -l app=nvidia-driver-daemonset
```

**Pod pending due to GPU:**

```bash
# Check if GPU resources are available
oc describe node <gpu-node> | grep -A5 "Allocated resources"

# Check pod events
oc describe pod <pod-name> -n <namespace>
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
