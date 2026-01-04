# Install with NVIDIA Dynamo

This guide provides step-by-step instructions for integrating vLLM Semantic Router with NVIDIA Dynamo.

## About NVIDIA Dynamo

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) is a high-performance distributed inference platform designed for large language model serving. Dynamo provides advanced features for optimizing GPU utilization and reducing inference latency through intelligent routing and caching mechanisms.

### Key Features

- **Disaggregated Serving**: Separate Prefill and Decode workers for optimal GPU utilization
- **KV-Aware Routing**: Routes requests to workers with relevant KV cache for prefix cache optimization
- **Dynamic Scaling**: Planner component handles auto-scaling based on workload
- **Multi-Tier KV Cache**: GPU HBM → System Memory → NVMe for efficient cache management
- **Worker Coordination**: etcd and NATS for distributed worker registration and message queuing
- **Backend Agnostic**: Supports vLLM, SGLang, and TensorRT-LLM backends

### Integration Benefits

Integrating vLLM Semantic Router with NVIDIA Dynamo provides several advantages:

1. **Dual-Layer Intelligence**: Semantic Router provides request-level intelligence (model selection, classification) while Dynamo optimizes infrastructure-level efficiency (worker selection, KV cache reuse)

2. **Intelligent Model Selection**: Semantic Router analyzes incoming requests and routes them to the most appropriate model based on content understanding, while Dynamo's KV-aware router efficiently selects optimal workers

3. **Dual-Layer Caching**: Semantic cache (request-level, Milvus-backed) combined with KV cache (token-level, Dynamo-managed) for maximum latency reduction

4. **Enhanced Security**: PII detection and jailbreak prevention filter requests before reaching inference workers

5. **Disaggregated Architecture**: Separate prefill and decode workers with KV-aware routing for reduced latency and better throughput

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT                                  │
│  curl -X POST http://localhost:8080/v1/chat/completions         │
│       -d '{"model": "MoM", "messages": [...]}'                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ENVOY GATEWAY                                  │
│  • Routes traffic, applies ExtProc filter                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              SEMANTIC ROUTER (ExtProc Filter)                    │
│  • Classifies query → selects category (e.g., "math")           │
│  • Selects model → rewrites request                             │
│  • Injects domain-specific system prompt                        │
│  • PII/Jailbreak detection                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              DYNAMO FRONTEND (KV-Aware Routing)                  │
│  • Receives enriched request with selected model                │
│  • Routes to optimal worker based on KV cache state             │
│  • Coordinates workers via etcd/NATS                            │
└─────────────────────────────────────────────────────────────────┘
                     │                          │
                     ▼                          ▼
     ┌───────────────────────────┐  ┌───────────────────────────┐
     │   PREFILL WORKER (GPU 1)  │  │   DECODE WORKER (GPU 2)   │
     │   Processes input tokens  │──▶  Generates output tokens  │
     │   --is-prefill-worker     │  │                           │
     └───────────────────────────┘  └───────────────────────────┘
```

## Prerequisites

### GPU Requirements

**This deployment requires a machine with at least 3 GPUs:**

| Component | GPU | Description |
|-----------|-----|-------------|
| Frontend | GPU 0 | Dynamo Frontend with KV-aware routing (`--router-mode kv`) |
| VLLMPrefillWorker | GPU 1 | Handles prefill phase of inference (`--is-prefill-worker`) |
| VLLMDecodeWorker | GPU 2 | Handles decode phase of inference |

### Required Tools

Before starting, ensure you have the following tools installed:

- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [Helm](https://helm.sh/docs/intro/install/) - Package manager for Kubernetes

### NVIDIA Runtime Configuration (One-Time Setup)

Configure Docker to use the NVIDIA runtime as the default:

```bash
# Configure NVIDIA runtime as default
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default

# Restart Docker
sudo systemctl restart docker

# Verify configuration
docker info | grep -i "default runtime"
# Expected output: Default Runtime: nvidia
```

## Step 1: Create Kind Cluster with GPU Support

Create a local Kubernetes cluster with GPU support. Choose one of the following options:

### Option 1: Quick Setup (External Documentation)

For a quick setup, follow the official Kind GPU documentation:

```bash
kind create cluster --name semantic-router-dynamo

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

For GPU support, see the [Kind GPU documentation](https://kind.sigs.k8s.io/docs/user/configuration/#extra-mounts) for details on configuring extra mounts and deploying the NVIDIA device plugin.

### Option 2: Full GPU Setup (E2E Procedure)

This is the procedure used in our E2E tests. It includes all the steps needed to set up GPU support in Kind.

#### 2.1 Create Kind Cluster with GPU Configuration

Create a Kind config file with GPU mount support:

```bash
# Create Kind config for GPU support
cat > kind-gpu-config.yaml << 'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: semantic-router-dynamo
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: /mnt
        containerPath: /mnt
  - role: worker
    extraMounts:
      - hostPath: /mnt
        containerPath: /mnt
      - hostPath: /dev/null
        containerPath: /var/run/nvidia-container-devices/all
EOF

# Create cluster with GPU config
kind create cluster --name semantic-router-dynamo --config kind-gpu-config.yaml --wait 5m

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

#### 2.2 Set Up NVIDIA Libraries in Kind Worker

Copy NVIDIA libraries from the host to the Kind worker node:

```bash
# Set worker name
WORKER_NAME="semantic-router-dynamo-worker"

# Detect NVIDIA driver version
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Detected NVIDIA driver version: $DRIVER_VERSION"

# Verify GPU devices exist in the Kind worker
docker exec $WORKER_NAME ls /dev/nvidia0
echo "✅ GPU devices found in Kind worker"

# Create directory for NVIDIA libraries
docker exec $WORKER_NAME mkdir -p /nvidia-driver-libs

# Copy nvidia-smi binary
tar -cf - -C /usr/bin nvidia-smi | docker exec -i $WORKER_NAME tar -xf - -C /nvidia-driver-libs/

# Copy NVIDIA libraries from host
tar -cf - -C /usr/lib64 libnvidia-ml.so.$DRIVER_VERSION libcuda.so.$DRIVER_VERSION | \
  docker exec -i $WORKER_NAME tar -xf - -C /nvidia-driver-libs/

# Create symlinks
docker exec $WORKER_NAME bash -c "cd /nvidia-driver-libs && \
  ln -sf libnvidia-ml.so.$DRIVER_VERSION libnvidia-ml.so.1 && \
  ln -sf libcuda.so.$DRIVER_VERSION libcuda.so.1 && \
  chmod +x nvidia-smi"

# Verify nvidia-smi works inside the Kind worker
docker exec $WORKER_NAME bash -c "LD_LIBRARY_PATH=/nvidia-driver-libs /nvidia-driver-libs/nvidia-smi"
echo "✅ nvidia-smi verified in Kind worker"
```

#### 2.3 Deploy NVIDIA Device Plugin

Deploy the NVIDIA device plugin to make GPUs allocatable in Kubernetes:

```bash
# Create device plugin manifest
cat > nvidia-device-plugin.yaml << 'EOF'
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.14.1
        name: nvidia-device-plugin-ctr
        env:
        - name: LD_LIBRARY_PATH
          value: "/nvidia-driver-libs"
        securityContext:
          privileged: true
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
        - name: dev
          mountPath: /dev
        - name: nvidia-driver-libs
          mountPath: /nvidia-driver-libs
          readOnly: true
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
      - name: dev
        hostPath:
          path: /dev
      - name: nvidia-driver-libs
        hostPath:
          path: /nvidia-driver-libs
EOF

# Apply device plugin
kubectl apply -f nvidia-device-plugin.yaml

# Wait for device plugin to be ready
sleep 20

# Verify GPUs are allocatable
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
echo "✅ GPU setup complete"
```

:::tip E2E Testing
The Semantic Router project includes automated E2E tests that handle all of this GPU setup automatically. You can run:

```bash
make e2e-test E2E_PROFILE=dynamo E2E_VERBOSE=true
```

This will create a Kind cluster with GPU support, deploy all components, and run the test suite.
:::

## Step 2: Install Dynamo Platform

Deploy the Dynamo platform components (etcd, NATS, Dynamo Operator):

```bash
# Add the Dynamo Helm repository
helm repo add dynamo https://nvidia.github.io/dynamo
helm repo update

# Install Dynamo CRDs
helm install dynamo-crds dynamo/dynamo-crds \
  --namespace dynamo-system \
  --create-namespace

# Install Dynamo Platform (etcd, NATS, Operator)
helm install dynamo-platform dynamo/dynamo-platform \
  --namespace dynamo-system \
  --wait

# Wait for platform components to be ready
kubectl wait --for=condition=Available deployment -l app.kubernetes.io/instance=dynamo-platform -n dynamo-system --timeout=300s
```

## Step 3: Install Envoy Gateway

Deploy Envoy Gateway with ExtensionAPIs enabled for Semantic Router integration:

```bash
# Install Envoy Gateway with custom values
helm install envoy-gateway oci://docker.io/envoyproxy/gateway-helm \
  --version v1.3.0 \
  --namespace envoy-gateway-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/envoy-gateway-values.yaml

# Wait for Envoy Gateway to be ready
kubectl wait --for=condition=Available deployment/envoy-gateway -n envoy-gateway-system --timeout=300s
```

**Important:** The values file enables `extensionApis.enableEnvoyPatchPolicy: true`, which is required for the Semantic Router ExtProc integration.

## Step 4: Deploy vLLM Semantic Router

Deploy the Semantic Router with Dynamo-specific configuration:

```bash
# Install Semantic Router from GHCR OCI registry
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/semantic-router-values/values.yaml

# Wait for deployment to be ready
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# Verify deployment status
kubectl get pods -n vllm-semantic-router-system
```

**Note:** The values file configures Semantic Router to route to the TinyLlama model served by Dynamo workers.

## Step 5: Deploy RBAC Resources

Apply RBAC permissions for Semantic Router to access Dynamo CRDs:

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/rbac.yaml
```

## Step 6: Deploy DynamoGraphDeployment

Deploy the Dynamo workers using the DynamoGraphDeployment CRD:

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/dynamo-graph-deployment.yaml

# Wait for the Dynamo operator to create the deployments
kubectl wait --for=condition=Available deployment/vllm-frontend -n dynamo-system --timeout=600s
kubectl wait --for=condition=Available deployment/vllm-vllmprefillworker -n dynamo-system --timeout=600s
kubectl wait --for=condition=Available deployment/vllm-vllmdecodeworker -n dynamo-system --timeout=600s
```

The DynamoGraphDeployment creates:

- **Frontend**: HTTP API server with KV-aware routing
- **VLLMPrefillWorker**: Specialized worker for prefill phase
- **VLLMDecodeWorker**: Specialized worker for decode phase

## Step 7: Create Gateway API Resources

Deploy the Gateway API resources to connect everything:

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/gwapi-resources.yaml

# Verify EnvoyPatchPolicy is accepted
kubectl get envoypatchpolicy -n default
```

**Important:** The EnvoyPatchPolicy status must show `Accepted: True`. If it shows `Accepted: False`, verify that Envoy Gateway was installed with the correct values file.

## Testing the Deployment

### Method 1: Port Forwarding (Recommended for Local Testing)

Set up port forwarding to access the gateway locally:

```bash
# Get the Envoy service name
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
  --selector=gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router \
  -o jsonpath='{.items[0].metadata.name}')

kubectl port-forward -n envoy-gateway-system svc/$ENVOY_SERVICE 8080:80
```

### Send Test Requests

Test the inference endpoint with a math query:

```bash
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

### Expected Response

```bash
HTTP/1.1 200 OK
server: fasthttp
date: Thu, 06 Nov 2025 06:38:08 GMT
content-type: application/json
x-vsr-selected-category: math
x-vsr-selected-reasoning: on
x-vsr-selected-model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
x-vsr-injected-system-prompt: true
transfer-encoding: chunked

{"id":"chatcmpl-...","model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"message":{"role":"assistant","content":"..."}}],"usage":{"prompt_tokens":15,"completion_tokens":54,"total_tokens":69}}
```

**Success indicators:**

- ✅ Request sent with `model="MoM"`
- ✅ Response shows `model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"` (rewritten by Semantic Router)
- ✅ Headers show category classification and system prompt injection

### Check Semantic Router Logs

```bash
# View classification and routing decisions
kubectl logs -n vllm-semantic-router-system deployment/semantic-router -f | grep -E "category|routing_decision"
```

Expected output:

```text
Classified as category: math (confidence=0.933)
Selected model TinyLlama/TinyLlama-1.1B-Chat-v1.0 for category math with score 1.0000
routing_decision: original_model="MoM", selected_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Verify EnvoyPatchPolicy Status

```bash
kubectl get envoypatchpolicy -n default -o yaml | grep -A 5 "status:"
```

Expected status:

```yaml
status:
  conditions:
  - type: Accepted
    status: "True"
  - type: Programmed
    status: "True"
```

## Cleanup

To remove the entire deployment:

```bash
# Remove Gateway API resources
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/gwapi-resources.yaml

# Remove DynamoGraphDeployment
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/dynamo-graph-deployment.yaml

# Remove RBAC
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/rbac.yaml

# Remove Semantic Router
helm uninstall semantic-router -n vllm-semantic-router-system

# Remove Envoy Gateway
helm uninstall envoy-gateway -n envoy-gateway-system

# Remove Dynamo Platform
helm uninstall dynamo-platform -n dynamo-system
helm uninstall dynamo-crds -n dynamo-system

# Delete namespaces
kubectl delete namespace vllm-semantic-router-system
kubectl delete namespace envoy-gateway-system
kubectl delete namespace dynamo-system

# Delete Kind cluster (optional)
kind delete cluster --name semantic-router-dynamo
```

## Production Configuration

For production deployments with larger models, modify the DynamoGraphDeployment:

```yaml
# Example: Using Llama-3-8B instead of TinyLlama
args:
  - "python3 -m dynamo.vllm --model meta-llama/Llama-3-8b-hf --tensor-parallel-size 2 --enforce-eager"
resources:
  requests:
    nvidia.com/gpu: 2  # Increase for tensor parallelism
```

**Considerations for production:**

- Use larger models appropriate for your use case
- Configure tensor parallelism for multi-GPU inference
- Enable distributed KV cache for multi-node deployments
- Set up monitoring and observability
- Configure autoscaling based on GPU utilization

## Next Steps

- Review the [NVIDIA Dynamo Integration Proposal](../../proposals/nvidia-dynamo-integration.md) for detailed architecture
- Set up [monitoring and observability](../../tutorials/observability/metrics.md)
- Configure [semantic caching with Milvus](../../tutorials/semantic-cache/milvus-cache.md) for production
- Scale the deployment for production workloads

## References

- [NVIDIA Dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- [Dynamo Documentation](https://docs.nvidia.com/dynamo/latest/)
- [Demo Video: Semantic Router + Dynamo E2E](https://www.youtube.com/watch?v=rRULSR9gTds&list=PLmrddZ45wYcuPrXisC-yl7bMI39PLo4LO&index=2)
