import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install with Gateway API Inference Extension

This guide provides step-by-step instructions for integrating vLLM Semantic Router (vSR) with a Gateway API Inference Extension (GIE) conformant inference gateway. GIE lets you manage self-hosted, OpenAI-compatible models with Kubernetes-native APIs such as `InferencePool`, while vSR adds prompt-aware model routing through the gateway's ExtProc integration.

## Architecture Overview

The deployment consists of three main components:

- **vLLM Semantic Router**: Classifies incoming requests and selects the target model.
- **GIE-conformant inference gateway**: Provides the Kubernetes Gateway API data plane. This guide includes tabs for Istio and agentgateway.
- **Gateway API Inference Extension (GIE)**: Provides Kubernetes-native inference APIs such as `InferencePool` for load-aware backend selection.

## Benefits of Integration

Integrating vSR with GIE provides a robust, Kubernetes-native solution for serving LLMs with several key benefits:

### 1. **Kubernetes-Native LLM Management**

Manage your models, routing, and scaling policies directly through `kubectl` using familiar Custom Resource Definitions (CRDs).

### 2. **Intelligent Model and Replica Routing**

Combine vSR's prompt-based model routing with GIE's smart, load-aware replica selection. This ensures requests are sent not only to the right model but also to the healthiest replica, all in a single, efficient hop.

### 3. **Protect Your Models from Overload**

The built-in scheduler tracks backend load and request queues, automatically shedding traffic to prevent your model servers from crashing under high demand.

### 4. **Deep Observability**

Gain insights from both high-level Gateway metrics and detailed vSR performance data (like token usage and classification accuracy) to monitor and troubleshoot your entire AI stack.

### 5. **Secure Multi-Tenancy**

Isolate tenant workloads using standard Kubernetes namespaces and `HTTPRoutes`. Apply rate limits and other policies while sharing a common, secure gateway infrastructure.

## Supported Backend Models and APIs

The demo models in this guide use the llm-d inference simulator to emulate Llama3 and Phi-4 through an **OpenAI-compatible API**. The simulator keeps the walkthrough runnable on a local kind cluster without GPUs or model downloads. You can replace it with your own model servers as long as the Semantic Router configuration, gateway routing resources, and GIE backend configuration agree on the request format and backend target.

OpenAI-compatible APIs are not the only supported option. agentgateway custom providers can declare provider-native formats such as OpenAI chat completions, Anthropic messages, responses, embeddings, token counting, and realtime APIs, and can route those providers to a host, Kubernetes `Service`, or `InferencePool`. GIE endpoint picker implementations can also support additional request formats through parser configuration; for example, the llm-d EPP parser framework includes OpenAI, Anthropic, vLLM HTTP, vLLM gRPC, Vertex AI, and passthrough parsers.

For details, see the agentgateway [custom providers](https://agentgateway.dev/docs/kubernetes/main/llm/providers/custom/) guide and the llm-d EPP [request parser documentation](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/requesthandling/parsers/README.md).

## Prerequisites

Before starting, ensure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) or another container runtime.
- [kind](https://kind.sigs.k8s.io/) v0.22+ or any Kubernetes 1.29+ cluster.
- [kubectl](https://kubernetes.io/docs/tasks/tools/) v1.30+.
- [Helm](https://helm.sh/) v3.14+.
- [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/) v1.28+ if you choose the Istio tab.

You can validate your toolchain versions with the following commands:

```bash
kind version
kubectl version --client --short
helm version --short
istioctl version --remote=false
```

## Step 1: Create a Kind Cluster (Optional)

If you don't have a Kubernetes cluster, create a local one for testing:

```bash
kind create cluster --name vsr-gie

# Verify the cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## Step 2: Install Gateway API and GIE CRDs

Install the shared CRDs for Gateway API and GIE before installing a gateway implementation:

```bash
export GATEWAY_API_VERSION=v1.5.0
export GIE_VERSION=v1.5.0

# Install Gateway API CRDs
kubectl apply --server-side --force-conflicts \
  -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

# Install Gateway API Inference Extension CRDs
kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GIE_VERSION}/manifests.yaml"

# Verify CRDs are installed
kubectl get crd | grep 'gateway.networking.k8s.io'
kubectl get crd | grep 'inference.networking.k8s.io'
```

## Step 3: Install an Inference Gateway

Choose the inference gateway you want to use. Each tab only contains gateway-specific installation steps.

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

Install Istio with support for Gateway API and external processing:

```bash
# Download and install Istio
export ISTIO_VERSION=1.29.0
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=$ISTIO_VERSION sh -
export PATH="$PWD/istio-$ISTIO_VERSION/bin:$PATH"
istioctl install -y --set profile=minimal --set values.pilot.env.ENABLE_GATEWAY_API=true

# Verify Istio is ready
kubectl wait --for=condition=Available deployment/istiod \
  -n istio-system \
  --timeout=300s
```

</TabItem>
<TabItem value="agentgateway">

Install agentgateway with inference extension support enabled:

```bash
export AGENTGATEWAY_VERSION=v1.3.0-alpha.1

# Install agentgateway CRDs
helm upgrade -i --create-namespace \
  --namespace agentgateway-system \
  --version "${AGENTGATEWAY_VERSION}" \
  agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds

# Install agentgateway with inference extension support
helm upgrade -i -n agentgateway-system \
  agentgateway oci://cr.agentgateway.dev/charts/agentgateway \
  --version "${AGENTGATEWAY_VERSION}" \
  --set inferenceExtension.enabled=true

# Verify agentgateway is ready
kubectl get pods -n agentgateway-system
```

This guide uses agentgateway `v1.3.0-alpha.1` or newer so the ExtProc policy can set `processingOptions` and `allowModeOverride`.

</TabItem>
</Tabs>

## Step 4: Deploy Demo LLM Servers

Deploy two lightweight inference simulator instances, Llama3 and Phi-4, to act as backends. These simulator deployments do not require GPUs or a Hugging Face token, and their labels match the `InferencePool` selectors used later in this guide.

```bash
# Deploy the simulator model servers
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inference-sim.yaml

# Wait for simulator models to be ready
kubectl wait --for=condition=Available deployment/vllm-llama3-8b-instruct --timeout=300s
kubectl wait --for=condition=Available deployment/phi4-mini --timeout=300s
```

The demo manifest runs in the `default` namespace and exposes services named `vllm-llama3-8b-instruct` and `phi4-mini`.

## Step 5: Deploy vLLM Semantic Router

Deploy the vLLM Semantic Router using its official Helm chart. This component runs as an ExtProc server that the selected gateway calls for routing decisions.

```bash
helm upgrade -i semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/semantic-router-values/values.yaml

# Wait for the router to be ready
kubectl -n vllm-semantic-router-system wait \
  --for=condition=Available deploy/semantic-router \
  --timeout=10m
```

The values file configures vSR to select `llama3-8b` for general prompts and `phi4-mini` for math prompts.

## Step 6: Deploy Gateway and Routing Logic

Apply the gateway-specific resources that create the public-facing `Gateway`, attach vSR as an ExtProc service, and route selected models to GIE `InferencePools`.

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

Apply the existing Istio gateway, `InferencePool`, `HTTPRoute`, `DestinationRule`, and `EnvoyFilter` resources:

```bash
# Apply all routing and gateway resources
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/gateway.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/envoyfilter.yaml

# Verify the Gateway is programmed by Istio
kubectl wait --for=condition=Programmed gateway/inference-gateway --timeout=120s
kubectl wait --for=condition=Available deployment/llm-d-inference-scheduler-llama3-8b --timeout=300s
kubectl wait --for=condition=Available deployment/llm-d-inference-scheduler-phi4-mini --timeout=300s
```

The Istio-specific `EnvoyFilter` inserts the ExtProc filter that calls the `semantic-router` service.

</TabItem>
<TabItem value="agentgateway">

Create an agentgateway-backed `Gateway`, apply the shared GIE `InferencePool` and `HTTPRoute` resources, and attach vSR with an `AgentgatewayPolicy`:

```bash
# Create an agentgateway inference Gateway
kubectl apply -f- <<'EOF'
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
  namespace: default
spec:
  gatewayClassName: agentgateway
  listeners:
  - name: http
    protocol: HTTP
    port: 80
    allowedRoutes:
      namespaces:
        from: All
EOF

# Apply shared GIE routing resources
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml

# Attach Semantic Router as the gateway ExtProc service
kubectl apply -f- <<'EOF'
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayPolicy
metadata:
  name: semantic-router-extproc
  namespace: default
spec:
  targetRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
  traffic:
    phase: PreRouting
    extProc:
      backendRef:
        name: semantic-router
        namespace: vllm-semantic-router-system
        port: 50051
      processingOptions:
        requestHeaderMode: Send
        requestBodyMode: Buffered
        responseHeaderMode: Send
        responseBodyMode: Buffered
        allowModeOverride: true
EOF

# Verify the Gateway is programmed by agentgateway
kubectl wait --for=condition=Programmed gateway/inference-gateway --timeout=120s
kubectl wait --for=condition=Available deployment/llm-d-inference-scheduler-llama3-8b --timeout=300s
kubectl wait --for=condition=Available deployment/llm-d-inference-scheduler-phi4-mini --timeout=300s
```

agentgateway uses `AgentgatewayPolicy` for ExtProc configuration, so it does not need Istio `DestinationRule` or `EnvoyFilter` resources. The policy uses `phase: PreRouting` so vSR can add `x-selected-model` before agentgateway evaluates the `HTTPRoute` header matches.

</TabItem>
</Tabs>

## Testing the Deployment

### Port Forwarding

Set up port forwarding to access the selected gateway from your local machine.

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

```bash
# The Gateway service is named inference-gateway-istio
kubectl port-forward svc/inference-gateway-istio 8080:80
```

</TabItem>
<TabItem value="agentgateway">

```bash
# agentgateway creates a service for the Gateway
kubectl port-forward svc/inference-gateway 8080:80
```

</TabItem>
</Tabs>

### Send Test Requests

Once port forwarding is active, send OpenAI-compatible requests to `localhost:8080`.

**Test 1: Explicitly request a model**

This request should be served by the Llama simulator. Add `-i` to the command if you want to inspect response headers such as `x-inference-pod` and `x-vsr-selected-model`.

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Summarize the Kubernetes Gateway API in three sentences."}]
  }'
```

**Test 2: Let the Semantic Router choose the model**

By setting `"model": "auto"`, you ask vSR to classify the prompt. It will identify this as a math query and add the `x-selected-model: phi4-mini` header, which `HTTPRoute` uses to route the request to the Phi-4 `InferencePool`. In the agentgateway tab, the `AgentgatewayPolicy` runs in the `PreRouting` phase so the header is present before route matching.

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2 * (5-1)?"}],
    "max_tokens": 64
  }'
```

## Troubleshooting

**Problem: CRDs are missing**

If you see errors like `no matches for kind "InferencePool"`, check that the CRDs are installed.

```bash
# Check for GIE CRDs
kubectl get crd | grep inference.networking.k8s.io
```

**Problem: Gateway is not ready**

If `kubectl port-forward` fails or requests time out, check the Gateway status.

```bash
# The Programmed condition should be True
kubectl get gateway inference-gateway -o yaml
```

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

**Problem: vSR is not being called**

If requests work but routing seems incorrect, check the Istio proxy logs for `ext_proc` errors.

```bash
# Get the Istio gateway pod name
export ISTIO_GW_POD=$(kubectl get pod -l istio=ingressgateway -o jsonpath='{.items[0].metadata.name}')

# Check its logs
kubectl logs $ISTIO_GW_POD -c istio-proxy | grep ext_proc
```

</TabItem>
<TabItem value="agentgateway">

**Problem: agentgateway or the ExtProc policy is not ready**

Check the agentgateway controller, generated Gateway workload, and `AgentgatewayPolicy`.

```bash
kubectl get pods -n agentgateway-system
kubectl get deployment inference-gateway
kubectl logs -n agentgateway-system deployment/agentgateway
kubectl describe agentgatewaypolicy semantic-router-extproc
```

</TabItem>
</Tabs>

**Problem: Requests are failing**

Check the logs for vSR and the backend models.

```bash
# Check vSR logs
kubectl logs deploy/semantic-router -n vllm-semantic-router-system

# Check backend logs
kubectl logs deployment/vllm-llama3-8b-instruct
kubectl logs deployment/phi4-mini
```

## Cleanup

To remove all resources created in this guide, run the cleanup commands for the gateway tab you used.

<Tabs groupId="gie-gateway" defaultValue="istio" values={[
  {label: 'Istio', value: 'istio'},
  {label: 'agentgateway', value: 'agentgateway'},
]}>
<TabItem value="istio">

```bash
# Delete routing and gateway resources
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/envoyfilter.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/destinationrule.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/istio/gateway.yaml

# Delete demo models
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inference-sim.yaml

# Uninstall Helm releases and Istio
helm uninstall semantic-router -n vllm-semantic-router-system
istioctl uninstall -y --purge

# Delete the kind cluster, if you created it
kind delete cluster --name vsr-gie
```

</TabItem>
<TabItem value="agentgateway">

```bash
# Delete gateway-specific resources
kubectl delete agentgatewaypolicy semantic-router-extproc --ignore-not-found
kubectl delete gateway inference-gateway --ignore-not-found

# Delete shared GIE routing resources
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml

# Delete demo models
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/kubernetes/llmd-base/inference-sim.yaml

# Uninstall Helm releases
helm uninstall semantic-router -n vllm-semantic-router-system
helm uninstall agentgateway -n agentgateway-system
helm uninstall agentgateway-crds -n agentgateway-system

# Delete the kind cluster, if you created it
kind delete cluster --name vsr-gie
```

</TabItem>
</Tabs>

## Next Steps

- **Customize Routing**: Modify the `values.yaml` file for the `semantic-router` Helm chart to define your own routing categories and rules.
- **Add Your Own Models**: Replace the demo Llama3 and Phi-4 deployments with your own OpenAI-compatible model servers.
- **Explore Advanced GIE Features**: Look into using `InferenceObjective` for more advanced autoscaling and scheduling policies.
- **Monitor Performance**: Integrate your Gateway and vSR with Prometheus and Grafana to build monitoring dashboards.
