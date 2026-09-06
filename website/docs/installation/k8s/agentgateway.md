---
title: Deploy with agentgateway
description: Attach Semantic Router as an ExtProc service to an agentgateway Kubernetes data plane.
---

# Deploy with agentgateway

Use this topology when [agentgateway](https://agentgateway.dev/) owns the
Kubernetes Gateway API data plane. Semantic Router runs as an Envoy ExtProc
service: it evaluates the selected recipe and writes the chosen model into the
request before agentgateway forwards it to an OpenAI-compatible backend.

## Responsibility split

The deployment consists of:

- **Semantic Router** owns semantic policy, model selection, and recipe-scoped
  request or response processing.
- **agentgateway** owns the Gateway, `HTTPRoute`, backend, and ExtProc policy.
- **The model server** owns inference capacity. The simulator below is for
  validating the integration, not for production inference.

## Prerequisites

You need:

- Kubernetes `1.31`–`1.36`; [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
  is sufficient for the demo;
- Gateway API `1.4`–`1.6` CRDs (this guide installs `1.6.0`);
- [kubectl](https://kubernetes.io/docs/tasks/tools/) within the supported
  version skew for the cluster; and
- [Helm](https://helm.sh/docs/intro/install/) `3.12` or later.

This guide pins the agentgateway 1.4 release line, including ExtProc
`processingOptions` and `allowModeOverride`. Review the upstream
[ExtProc reference](https://agentgateway.dev/docs/kubernetes/latest/traffic-management/extproc/)
before upgrading either side of the integration.

## Step 1: Create Kind Cluster (Optional)

Create a local Kubernetes cluster for testing:

```bash
kind create cluster --name semantic-router-agentgateway

# Verify cluster is ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## Step 2: Install agentgateway

Install the Kubernetes Gateway API CRDs and the agentgateway control plane:

```bash
export AGENTGATEWAY_VERSION=v1.4.1

kubectl apply --server-side --force-conflicts \
  -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.6.0/standard-install.yaml

helm upgrade -i agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds \
  --create-namespace \
  --namespace agentgateway-system \
  --version "${AGENTGATEWAY_VERSION}"

helm upgrade -i agentgateway oci://cr.agentgateway.dev/charts/agentgateway \
  --namespace agentgateway-system \
  --version "${AGENTGATEWAY_VERSION}" \
  --wait

kubectl get pods -n agentgateway-system
```

## Step 3: Create an agentgateway proxy

Create a Gateway that uses the agentgateway GatewayClass:

```bash
kubectl apply -f- <<'EOF'
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: agentgateway-proxy
  namespace: agentgateway-system
spec:
  gatewayClassName: agentgateway
  listeners:
  - protocol: HTTP
    port: 80
    name: http
    allowedRoutes:
      namespaces:
        from: All
EOF

kubectl wait --for=condition=Available deployment/agentgateway-proxy \
  -n agentgateway-system \
  --timeout=300s
```

## Step 4: Deploy Demo LLM

Deploy a lightweight OpenAI-compatible simulator that serves `base-model` plus the LoRA adapter names selected by the Semantic Router demo configuration:

```bash
kubectl apply -f- <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b-instruct
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-llama3-8b-instruct
  template:
    metadata:
      labels:
        app: vllm-llama3-8b-instruct
    spec:
      containers:
      - name: vllm-sim
        image: ghcr.io/llm-d/llm-d-inference-sim:v0.6.1
        imagePullPolicy: IfNotPresent
        args:
        - --model
        - base-model
        - --port
        - "8000"
        - --max-loras
        - "6"
        - --lora-modules
        - '{"name": "math-expert"}'
        - '{"name": "science-expert"}'
        - '{"name": "social-expert"}'
        - '{"name": "humanities-expert"}'
        - '{"name": "law-expert"}'
        - '{"name": "general-expert"}'
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        readinessProbe:
          httpGet:
            path: /health
            port: http
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama3-8b-instruct
  namespace: default
  labels:
    app: vllm-llama3-8b-instruct
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
  selector:
    app: vllm-llama3-8b-instruct
EOF

kubectl wait --for=condition=Available deployment/vllm-llama3-8b-instruct \
  -n default \
  --timeout=300s
```

## Step 5: Deploy vLLM Semantic Router

Install the Semantic Router in the `agentgateway-system` namespace so the agentgateway ExtProc policy can reference the `semantic-router` service directly:

```bash
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace agentgateway-system \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/agentgateway/semantic-router-values/values.yaml \
  --set config.global.router.streamed_body.enabled=true \
  --set config.global.router.streamed_body.max_bytes=10485760 \
  --set config.global.router.streamed_body.timeout_sec=30

kubectl wait --for=condition=Available deployment/semantic-router \
  -n agentgateway-system \
  --timeout=600s
```

The values file configures Semantic Router to send traffic to `vllm-llama3-8b-instruct.default.svc.cluster.local:8000` and to select adapter names such as `math-expert`, `science-expert`, and `general-expert`.

## Step 6: Create agentgateway routing resources

Create an `AgentgatewayBackend` for the vLLM-compatible backend and route OpenAI-compatible requests to it:

```bash
kubectl apply -f- <<'EOF'
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayBackend
metadata:
  name: semantic-router-vllm
  namespace: agentgateway-system
spec:
  ai:
    provider:
      openai: {}
      host: vllm-llama3-8b-instruct.default.svc.cluster.local
      port: 8000
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: semantic-router-vllm
  namespace: agentgateway-system
spec:
  parentRefs:
  - name: agentgateway-proxy
    namespace: agentgateway-system
  rules:
  - backendRefs:
    - name: semantic-router-vllm
      namespace: agentgateway-system
      group: agentgateway.dev
      kind: AgentgatewayBackend
EOF
```

The `openai.model` field is intentionally omitted so agentgateway uses the model name from the request body after Semantic Router selects the target model or LoRA adapter.

## Step 7: Attach Semantic Router as ExtProc

Create an `AgentgatewayPolicy` that sends request and response processing phases to the Semantic Router ExtProc service:

```bash
kubectl apply -f- <<'EOF'
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayPolicy
metadata:
  name: semantic-router-extproc
  namespace: agentgateway-system
spec:
  targetRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: agentgateway-proxy
  traffic:
    extProc:
      backendRef:
        name: semantic-router
        namespace: agentgateway-system
        port: 50051
      processingOptions:
        requestHeaderMode: Send
        requestBodyMode: FullDuplexStreamed
        responseHeaderMode: Send
        responseBodyMode: Buffered
        requestTrailerMode: Send
        responseTrailerMode: Send
        allowModeOverride: true
EOF
```

The bundled agentgateway example explicitly opts into full-duplex streamed request
bodies. This is an example-specific choice; other proxy defaults and examples
may continue to use buffered request bodies. The Semantic Router Helm command
above explicitly enables `global.router.streamed_body`, allowing the router to
accumulate request chunks and process the complete body at end-of-stream.

agentgateway does not support `Streamed` mode; `FullDuplexStreamed` is its
streaming option. The deployable policy is in
`deploy/kubernetes/agentgateway/extproc-policy.yaml`, with the matching router
configuration passed through the Helm command in Step 5. See
[Streamed ExtProc and immediate responses](./streamed-extproc) for protocol
behavior and the verification checklist.

## Testing the Deployment

Start a port-forward to the agentgateway proxy:

```bash
kubectl port-forward -n agentgateway-system svc/agentgateway-proxy 8080:80
```

In another terminal, send an OpenAI-compatible request with the stable
automatic-routing alias:

```bash
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [
      {"role": "user", "content": "What is the derivative of f(x) = x^3?"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Semantic Router should classify the math prompt, select the configured math route, and mutate the request model before agentgateway forwards the request to the vLLM-compatible backend. Use `-i` to inspect Semantic Router response headers such as the selected model metadata.

## Troubleshooting

**agentgateway proxy not ready:**

```bash
kubectl get gateway agentgateway-proxy -n agentgateway-system
kubectl get deployment agentgateway-proxy -n agentgateway-system
kubectl logs -n agentgateway-system deployment/agentgateway
```

**HTTPRoute or agentgateway backend not accepted:**

```bash
kubectl describe httproute semantic-router-vllm -n agentgateway-system
kubectl describe agentgatewaybackend semantic-router-vllm -n agentgateway-system
```

**Semantic Router not responding to ExtProc:**

```bash
kubectl get pods -n agentgateway-system
kubectl get svc semantic-router -n agentgateway-system
kubectl logs -n agentgateway-system deployment/semantic-router
kubectl describe agentgatewaypolicy semantic-router-extproc -n agentgateway-system
```

**Demo LLM not responding:**

```bash
kubectl get pods -n default -l app=vllm-llama3-8b-instruct
kubectl logs -n default deployment/vllm-llama3-8b-instruct
```

## Cleanup

To remove the entire deployment:

```bash
kubectl delete agentgatewaypolicy semantic-router-extproc -n agentgateway-system
kubectl delete httproute semantic-router-vllm -n agentgateway-system
kubectl delete agentgatewaybackend semantic-router-vllm -n agentgateway-system
kubectl delete gateway agentgateway-proxy -n agentgateway-system
kubectl delete deployment vllm-llama3-8b-instruct -n default
kubectl delete service vllm-llama3-8b-instruct -n default

helm uninstall semantic-router -n agentgateway-system
helm uninstall agentgateway -n agentgateway-system
helm uninstall agentgateway-crds -n agentgateway-system

kind delete cluster --name semantic-router-agentgateway
```
