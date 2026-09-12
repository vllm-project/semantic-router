---
sidebar_position: 3
sidebar_label: Kubernetes Operator
title: Deploy with the Kubernetes Operator
description: Install the Semantic Router Operator, connect model services, and create a first Router deployment.
---

# Deploy with the Kubernetes Operator

The Semantic Router Operator reconciles `SemanticRouter` custom resources into
Router workloads, Services, configuration, storage, and optional platform
integrations. Use it when Kubernetes should own the Router lifecycle and model
backends are already exposed as Kubernetes services, KServe
`InferenceService`s, or Llama Stack services.

The Operator does not deploy model servers. It discovers or references them and
generates the provider bindings used by the Router.

## What the Operator manages

- Router Deployment and Service
- canonical Router configuration generated from the custom resource
- optional persistent model storage
- probes, resources, scheduling, autoscaling, and ingress settings
- OpenShift security defaults and optional Route creation
- standalone Envoy sidecar or integration with an existing Gateway

For the top-level field families and links to the installed schema, see the
[SemanticRouter CRD reference](../../api/semantic-router-crd).

## Prerequisites

- a supported Kubernetes or OpenShift cluster
- `kubectl` or `oc` configured for the target cluster
- Git, GNU Make, and Go 1.25 or newer for the source-based install below
- permission to install CRDs and cluster-scoped RBAC
- at least one reachable OpenAI-compatible model service

## Install the Operator

### Kubernetes with Kustomize

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router/deploy/operator

make install
make deploy IMG=ghcr.io/vllm-project/semantic-router/operator:latest
```

Verify the controller:

```bash
kubectl get pods -n semantic-router-operator-system
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager
```

Pin a released image tag or digest for a controlled environment rather than
using `latest`.

### OpenShift with OLM

Use the Kustomize flow above when deploying the controller directly on
OpenShift. The reconciler detects OpenShift and applies its platform-specific
workload defaults.

For an Operator Lifecycle Manager installation, first publish or select an OLM
catalog that contains the Semantic Router bundle, then create the
`CatalogSource`, `OperatorGroup`, and `Subscription` required by your cluster.
The repository's `make openshift-deploy` target creates only the namespace,
OperatorGroup, and Subscription; it assumes a `semantic-router-catalog`
CatalogSource already exists in `openshift-marketplace`. It is therefore a
maintainer convenience target, not a standalone installation command.

## Create a first Router

This example binds an existing Service named `model-server` in the same
namespace. The Operator creates the provider model and model card and uses the
first discovered model as the default.

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
  namespace: default
spec:
  replicas: 1
  vllmEndpoints:
    - name: local-backend
      model: local/model
      backend:
        type: service
        service:
          name: model-server
          port: 8000
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: "2"
      memory: 4Gi
```

Apply and wait for readiness:

```bash
kubectl apply -f my-router.yaml
kubectl get semanticrouter my-router -w
kubectl get deployment,service \
  -l app.kubernetes.io/instance=my-router
```

Send a direct request with the concrete model name, or add canonical routing
under `spec.config.routing` to expose automatic or policy-driven behavior.

## Add routing policy

`spec.config.routing` accepts the canonical routing object. The example below
adds a catch-all decision over the discovered model:

```yaml
spec:
  config:
    routing:
      strategy: priority
      modelCards:
        - name: local/model
          modality: text
          capabilities: [chat]
      decisions:
        - name: default-route
          description: Route unmatched requests to the discovered local model.
          priority: 1
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: local/model
```

The remaining `spec.config` fields are Operator adapters for shared Router
settings such as response cache, classifiers, tools, observability, and
reasoning families. They are translated into canonical `global` and provider
sections. Consult the CRD reference rather than copying fields from a local
`config.yaml` into arbitrary CR paths.

## Backend discovery

Each `spec.vllmEndpoints[]` entry declares one logical model and one way to
resolve its backend:

| `backend.type` | Use when | Required fields |
|----------------|----------|-----------------|
| `service` | An OpenAI-compatible Kubernetes Service already exists | `service.name`, `service.port`; optional `service.namespace` |
| `kserve` | KServe owns the model deployment | `inferenceServiceName` |
| `llamastack` | Services should be selected by labels | `discoveryLabels` |

Example for a service in another namespace:

```yaml
spec:
  vllmEndpoints:
    - name: qwen-backend
      model: qwen/assistant
      reasoning:
        family: qwen3
      backend:
        type: service
        service:
          name: qwen-vllm
          namespace: model-serving
          port: 8000
```

The `model` value must match the name served by the provider. The `name` value
identifies the generated backend reference. Optional LoRA declarations become
entries under the generated routing model card.

## Deployment modes

### Standalone

With no `spec.gateway`, the Operator deploys an Envoy sidecar next to the
Router. Client traffic enters the Service, Envoy invokes ExtProc, and Envoy
forwards the transformed request to the selected backend.

This mode is appropriate when the Router should be self-contained and the
cluster does not already provide a compatible gateway.

### Existing Gateway

There are two distinct ways to attach an existing Gateway.

#### Forward HTTP to the Operator's Envoy sidecar

For a Gateway API controller that forwards HTTP, keep the Router in standalone
mode: **omit `spec.gateway.existingRef`**. The Operator creates its Envoy
sidecar and exposes the inference listener as Service port **8801**
(`envoy-http`). Create an `HTTPRoute` targeting that port:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: my-router
  namespace: default
spec:
  parentRefs:
    - name: shared-gateway
      namespace: gateway-system
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /v1
      backendRefs:
        - name: my-router
          port: 8801
      timeouts:
        request: 300s
        backendRequest: 300s
```

The referenced Gateway listener must allow routes from the Router namespace
through `allowedRoutes.namespaces`. The route and Router Service above share a
namespace; moving the backend Service to another namespace also requires a
`ReferenceGrant`. Configure the Gateway's hostname, TLS and authentication for
your deployment. The Operator does not create an `HTTPRoute`; you own this
route's lifecycle.

The traffic path is Gateway → Router Service `envoy-http` → Envoy sidecar →
ExtProc → selected model backend. The `api` port (8080 by default) serves Router
management requests and must not be used as the inference route's backend.

A complete Router and route example is available in
[`vllm.ai_v1alpha1_semanticrouter_gateway.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_gateway.yaml).
After adapting its existing Gateway and model Service names, apply both
resources and check route acceptance before sending a real completion. These
commands use that sample's resource names and namespace:

```bash
kubectl -n vllm-serving get httproute semantic-router-gateway -o yaml
kubectl -n vllm-serving get service semantic-router-gateway -o jsonpath='{.spec.ports}'
kubectl -n vllm-serving port-forward service/semantic-router-gateway 8801:8801
# In another terminal, verify the same request through this local data plane.
curl --fail-with-body http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local/model","messages":[{"role":"user","content":"Hello"}]}'
```

Then send the request through your Gateway URL. Its `HTTPRoute` parent status
must report `Accepted=True` and `ResolvedRefs=True`; an accepted route alone
does not prove backend inference succeeds.

#### Let the existing Gateway invoke ExtProc

Use `spec.gateway.existingRef` only when the existing Gateway will own the
inference data plane and has the gateway-specific integration configured:

```yaml
spec:
  gateway:
    existingRef:
      name: shared-gateway
      namespace: gateway-system
```

This mode verifies the Gateway exists and omits the Envoy sidecar. You must
configure the Gateway's ExtProc policy to call the Router Service's **gRPC port
50051** (or your configured `service.grpc.port`), plus routes to the actual
model backends that honor the Router's selection. Referencing a Gateway alone
does not install those policies or routes. Follow the matching
[Kubernetes Gateway integration guide](gateways) for its processing modes and
backend selection contract. Do not point an inference `HTTPRoute` at the Router
management API; it does not implement `/v1/chat/completions`.

### OpenShift Route

On OpenShift, the Operator can create a Route:

```yaml
spec:
  openshift:
    routes:
      enabled: true
      tls:
        termination: edge
        insecureEdgeTerminationPolicy: Redirect
```

Omit the hostname to let OpenShift allocate one, or provide a hostname covered
by your DNS and certificate configuration.

## Secrets

Use Kubernetes Secrets for provider, registry, and model-download credentials.
Reference them from `spec.env` rather than placing literal values in the custom
resource:

```yaml
spec:
  env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: model-download-credentials
          key: token
```

Apply least-privilege RBAC and restrict who can read the generated ConfigMaps,
Secrets, logs, and custom resources. See
[Security Hardening](../security-hardening).

## Verify the deployment

```bash
kubectl get semanticrouter my-router -o wide
kubectl describe semanticrouter my-router
kubectl get pods,service,pvc \
  -l app.kubernetes.io/instance=my-router
kubectl logs deployment/my-router -c semantic-router
```

Check both reconciliation and data-plane behavior:

1. the custom resource's observed generation matches its generation;
2. the expected replicas are ready;
3. every discovered backend exists and serves the configured model name; and
4. a real completion succeeds through the Service or Gateway.

## Next

- [Operate an Operator deployment](operator-operations)
- [SemanticRouter CRD reference](../../api/semantic-router-crd)
- [Configuration Workflows](../configuration-workflows)
- [Kubernetes Gateway API](gateway-api-inference-extension)
- [Storage options](../storage-overview)
