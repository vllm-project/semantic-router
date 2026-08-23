---
title: Integrate with NVIDIA Dynamo
sidebar_label: NVIDIA Dynamo
description: Put Semantic Router in front of a Dynamo-managed inference fleet on Kubernetes.
---

# Integrate with NVIDIA Dynamo

Use this integration when NVIDIA Dynamo already owns model serving and you want
Semantic Router to choose the model or policy before Dynamo schedules the
request onto inference workers.

The two routers make decisions at different layers:

| Layer | Responsibility |
|-------|----------------|
| Semantic Router | Resolve an entrypoint, evaluate semantic signals and policy, run request plugins, and select a provider model. |
| Dynamo | Reconcile the inference graph, expose its OpenAI-compatible frontend, and select workers using its serving topology and cache-aware routing. |

Semantic Router must select a model name that the Dynamo frontend serves.
Dynamo can then choose among the workers for that model.

```text
Client
  -> Envoy Gateway
     -> Semantic Router (ExtProc)
        -> Dynamo frontend
           -> Dynamo router and inference workers
```

Semantic response caching and Dynamo KV-cache routing are independent. A
Semantic Router cache can reuse a prior response; Dynamo's router reuses or
predicts token-level prefix state while serving a new request.

## Prerequisites

You need:

- Kubernetes `1.33`–`1.36` with NVIDIA GPU nodes. This is the supported range
  for the pinned Envoy Gateway `v1.9.0` in this integration.
- Gateway API `v1.6.1` CRDs
- `kubectl` within the supported version skew for the cluster
- Helm 3
- NVIDIA GPU Operator, unless your cluster provider supplies equivalent GPU
  drivers and container runtime integration
- a Hugging Face token Secret when the selected model requires authentication

For cluster preparation, supported accelerators, and optional schedulers, use
the [NVIDIA Dynamo Kubernetes Quickstart](https://docs.nvidia.com/dynamo/dev/kubernetes/getting-started/quickstart).

## 1. Install the Dynamo platform

This guide pins Dynamo 1.4.0. Confirm the version and its compatibility matrix
in NVIDIA's [Dynamo release artifacts](https://docs.nvidia.com/dynamo/dev/reference/release-artifacts)
before installing or upgrading it.

```bash
export DYNAMO_NAMESPACE=dynamo-system
export DYNAMO_VERSION=1.4.0

helm upgrade --install dynamo-platform \
  "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${DYNAMO_VERSION}.tgz" \
  --namespace "$DYNAMO_NAMESPACE" \
  --create-namespace \
  --wait \
  --timeout 10m
```

Check the operator and platform services before deploying a model:

```bash
helm status dynamo-platform --namespace "$DYNAMO_NAMESPACE"
kubectl get pods --namespace "$DYNAMO_NAMESPACE"
kubectl get crd | grep -i dynamo
```

The platform chart installs Dynamo's control plane. It does not, by itself,
define the model topology that will serve your requests.

## 2. Deploy a model with Dynamo

Follow NVIDIA's [model deployment overview](https://docs.nvidia.com/dynamo/dev/kubernetes/model-deployment/introduction)
to choose a tuned recipe, generate a deployment with a
`DynamoGraphDeploymentRequest`, or apply a known-good
`DynamoGraphDeployment`. Those APIs and runtime images evolve with Dynamo, so
this guide does not duplicate their manifests.

After deployment, identify the frontend Service and confirm the served model
name:

```bash
kubectl get dynamographdeployments,dynamocomponentdeployments \
  --namespace "$DYNAMO_NAMESPACE"
kubectl get services --namespace "$DYNAMO_NAMESPACE"
```

Record the values that Semantic Router will use:

```bash
export DYNAMO_FRONTEND_SERVICE=your-frontend-service
export DYNAMO_FRONTEND_PORT=8000
export DYNAMO_MODEL=your-served-model-name
```

Before adding another routing layer, port-forward the frontend Service in a
separate terminal and send a direct OpenAI-compatible request. This isolates
Dynamo deployment problems from gateway or Semantic Router problems.

```bash
kubectl port-forward \
  --namespace "$DYNAMO_NAMESPACE" \
  "service/$DYNAMO_FRONTEND_SERVICE" \
  8000:"$DYNAMO_FRONTEND_PORT"
```

Use the request example from the Dynamo deployment you selected. Verify both
the model identifier and a chat completion against `http://localhost:8000`.

```bash
curl -fsS http://localhost:8000/v1/models
```

:::caution Compatibility of repository examples

The `deploy/kubernetes/dynamo/helm-chart` and
`dynamo-resources/dynamo-graph-deployment.yaml` examples in this repository
target an older Dynamo API and runtime. Do not combine them unchanged with the
1.4.0 platform. Use NVIDIA's current deployment guide for the Dynamo model
resources; use the repository files below only for the Semantic Router and
gateway integration.

:::

## 3. Configure Semantic Router for the Dynamo frontend

Download the integration values and edit them before installing the chart:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/semantic-router-values/values.yaml \
  -o semantic-router-dynamo-values.yaml
```

Replace the sample model and endpoint with the values from your Dynamo
deployment. The provider endpoint should use cluster DNS:

```text
<frontend-service>.<dynamo-namespace>.svc.cluster.local:<frontend-port>
```

Update `models[].connections[]` for the Dynamo frontend, then point each
Entrypoint assignment at the corresponding Model name. Each connection's
`model` must match an entry returned by the Dynamo frontend.

For development, install the continuously published Semantic Router chart:

```bash
export SEMANTIC_ROUTER_NAMESPACE=vllm-semantic-router-system

helm upgrade --install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-latest \
  --namespace "$SEMANTIC_ROUTER_NAMESPACE" \
  --create-namespace \
  --values semantic-router-dynamo-values.yaml \
  --wait
```

`0.0.0-latest` follows the main branch. For production, pin a tested release
of the chart and an explicit Semantic Router image tag.

## 4. Connect Envoy Gateway

The repository integration uses an `EnvoyPatchPolicy` to insert Semantic
Router as an ExtProc filter. Install Envoy Gateway with that extension enabled:

```bash
export ENVOY_GATEWAY_VERSION=v1.9.0

helm upgrade --install envoy-gateway \
  oci://docker.io/envoyproxy/gateway-helm \
  --version "$ENVOY_GATEWAY_VERSION" \
  --namespace envoy-gateway-system \
  --create-namespace \
  --values https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/envoy-gateway-values.yaml \
  --wait
```

If the cluster already has Envoy Gateway, confirm its version and Gateway API
CRDs are compatible before upgrading it. `EnvoyPatchPolicy` is version-sensitive
and can change gateway behavior; restrict permission to create or modify these
resources.
See the [Envoy Gateway installation guide](https://gateway.envoyproxy.io/docs/install/install-helm/)
and [EnvoyPatchPolicy security guidance](https://gateway.envoyproxy.io/docs/tasks/extensibility/envoy-patch-policy/).

Download the Gateway API integration manifest:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/gwapi-resources.yaml \
  -o semantic-router-dynamo-gateway.yaml
```

Before applying it, update these environment-specific references:

- the `HTTPRoute` backend Service name, namespace, and port
- the `ReferenceGrant` namespace when the Dynamo frontend is not in
  `dynamo-system`
- the Semantic Router Service address when you changed its release name or
  namespace
- the Gateway and route namespaces when `default` is not appropriate

Then apply it and inspect resource status:

```bash
kubectl apply --filename semantic-router-dynamo-gateway.yaml
kubectl get gateway,httproute --all-namespaces
kubectl describe envoypatchpolicy semantic-router-extproc-patch-policy \
  --namespace default
```

Do not continue until the Gateway and route are accepted and the patch policy
is programmed.

## 5. Verify the complete request path

Resolve the real Gateway address for your cluster and set `GATEWAY_URL`. The
[gateway test checklist](./gateway-testing) covers LoadBalancer, local cluster,
route-status, and log checks.

Send a request through the Gateway using an entrypoint configured in Semantic
Router. The sample values use the default automatic entrypoint:

```bash
curl -fsS -D - "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Explain why prefix caching can reduce inference latency."}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

Correlate one request across the Gateway, Semantic Router, and Dynamo frontend
logs. A successful HTTP response alone does not prove that the request passed
through the intended route or reached the selected Dynamo deployment.

## Troubleshooting

| Symptom | Check first |
|---------|-------------|
| Direct frontend request fails | Dynamo deployment status, GPU allocation, model credentials, and frontend logs |
| Direct frontend works but Gateway fails | `HTTPRoute` backend name and port, `ReferenceGrant`, and Gateway address |
| Patch policy is not accepted | Envoy Gateway extension setting, patch target namespace, and release compatibility |
| Physical model works but `auto` fails | Semantic Router entrypoint, provider model name, decisions, and classifier readiness |
| Request reaches the wrong Dynamo deployment | Selected-model headers, provider endpoint, Dynamo frontend model list, and both routing layers' logs |

## Cleanup

Delete only resources created for this integration. Remove the Gateway
resources first so new traffic cannot enter during teardown:

```bash
kubectl delete --filename semantic-router-dynamo-gateway.yaml \
  --ignore-not-found
helm uninstall semantic-router \
  --namespace "$SEMANTIC_ROUTER_NAMESPACE" \
  --ignore-not-found
```

Delete the DGD or DGDR using the name from the NVIDIA deployment workflow. If
this was a dedicated Dynamo installation, remove the platform last:

```bash
helm uninstall dynamo-platform \
  --namespace "$DYNAMO_NAMESPACE" \
  --ignore-not-found
```

Uninstall Envoy Gateway only when it was installed solely for this setup. Do
not delete shared namespaces or CRDs as part of routine application cleanup.

## Further reading

- [NVIDIA Dynamo Kubernetes Quickstart](https://docs.nvidia.com/dynamo/dev/kubernetes/getting-started/quickstart)
- [NVIDIA Dynamo model deployment overview](https://docs.nvidia.com/dynamo/dev/kubernetes/model-deployment/introduction)
- [NVIDIA Dynamo release artifacts](https://docs.nvidia.com/dynamo/dev/reference/release-artifacts)
- [Semantic Router Dynamo integration files](https://github.com/vllm-project/semantic-router/tree/main/deploy/kubernetes/dynamo)
