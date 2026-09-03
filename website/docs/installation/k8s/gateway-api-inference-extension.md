---
title: Gateway API Inference Extension
description: Combine semantic model-pool selection with endpoint selection inside a Kubernetes InferencePool.
---

# Gateway API Inference Extension

Gateway API Inference Extension (GAIE) and Semantic Router solve different
parts of routing:

- **Semantic Router** chooses a logical model or pool from the request's
  meaning, policy, and recipe state.
- **GAIE** represents that pool as an `InferencePool` and lets an endpoint
  picker choose a ready replica.

Use them together when one public model ID can select among several model
pools, and each pool contains multiple interchangeable serving replicas. If
every model maps directly to one Kubernetes Service, use a simpler gateway
integration such as [Istio](istio) instead.

```text
client
  -> Gateway
  -> Semantic Router ExtProc
  -> HTTPRoute chosen by x-selected-model
  -> InferencePool
  -> endpoint picker
  -> model replica
```

## Choose and install the gateway stack

Install and verify GAIE with a gateway implementation supported by your
platform. Use one compatibility set from the upstream projects; do not combine
CRDs, charts, and examples copied from different releases.

- [GAIE documentation](https://gateway-api-inference-extension.sigs.k8s.io/)
- [Supported gateway implementations](https://gateway-api-inference-extension.sigs.k8s.io/implementations/gateways/)
- [GAIE releases](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases)
- [llm-d gateway providers](https://llm-d.ai/docs/infrastructure/gateway)

Before adding Semantic Router, verify that:

1. the `Gateway` reports `Programmed=True`;
2. each `HTTPRoute` reports accepted and resolved references;
3. each `InferencePool` has ready endpoints; and
4. a direct request reaches the expected pool.

Semantic Router does not install or own the gateway controller, GAIE CRDs,
endpoint picker, or model servers.

## Define the name contract

The provider model selected by Semantic Router becomes the request header used
by the Gateway API route. The exact value must agree across all three objects:

```yaml
# Router config fragment
providers:
  defaults:
    default_model: local/general
  models:
    - name: local/general
      provider_model_id: served-general
      api_format: openai
      backend_refs:
        - name: general-pool
          endpoint: general-pool.inference.svc.cluster.local:8000
          protocol: http
          weight: 100
```

```yaml
# Gateway API fragment
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: general-pool
  namespace: inference
spec:
  parentRefs:
    - name: inference-gateway
  rules:
    - matches:
        - headers:
            - type: Exact
              name: x-selected-model
              value: local/general
      backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: general-pool
          port: 8000
```

The Router writes `x-selected-model` on the request. It exposes the logical
selection to clients as `x-vsr-selected-model` on the response. GAIE owns the
later endpoint choice inside `general-pool`.

## Deploy Semantic Router

Create and validate a complete config before applying it:

```bash
vllm-sr validate --config config.yaml
```

Then deploy with the [Helm or Operator workflow](../configuration-workflows).
For direct Helm, pass the full canonical document through `configOverride`;
that replaces the chart sample config before the chart applies its
Kubernetes-owned rewrites.

## Re-evaluate the route after ExtProc

The gateway must call Semantic Router in the downstream request path and then
re-evaluate the route after `x-selected-model` is written. Keep this Router
setting enabled in the canonical config:

```yaml
global:
  router:
    clear_route_cache: true
```

This asks Envoy to discard the route selected before ExtProc and evaluate the
`HTTPRoute` header match again. Make sure the gateway's ExtProc policy preserves
that response flag. Place route-dependent authorization and other filters only
after the re-evaluated route, or verify their ordering explicitly.

- **Istio:** use an ExtProc `EnvoyFilter` and its service `DestinationRule`.
  The [Istio guide](istio) shows the direct-Service version of this attachment.
- **agentgateway:** attach an `AgentgatewayPolicy` in its pre-routing phase.
  See [agentgateway](agentgateway).
- **Envoy AI Gateway / Envoy Gateway:** use the gateway's supported ExtProc
  policy surface. See [Envoy AI Gateway](ai-gateway).

Do not apply attachment resources from one gateway implementation to another;
their policy APIs and processing modes are not interchangeable.

For chunked request bodies or immediate streamed responses, also configure the
body mode described in [Streamed ExtProc](streamed-extproc).

## Verify the combined path

Send a request using a virtual model exposed by your active config:

```bash
curl -i "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Explain this API error."}]
  }'
```

Check each layer:

```bash
kubectl get gateway,httproute -A
kubectl get inferencepools -A
kubectl get httproute general-pool -n inference \
  -o jsonpath='{.status.parents[*].conditions[*].type}{" "}{.status.parents[*].conditions[*].status}{"\n"}'
```

The response's `x-vsr-selected-model` should match the route's
`x-selected-model` value, and the endpoint picker should report a ready endpoint
from that pool. An HTTP 200 alone does not prove that semantic selection and
endpoint selection both ran.

## Troubleshooting by ownership

| Symptom | Start here |
| --- | --- |
| No `x-vsr-selected-model` response header | Semantic Router recipe selection and ExtProc attachment |
| Header is present but the route is not selected | `HTTPRoute` header value, route status, and gateway processing order |
| `ResolvedRefs=False` | `InferencePool` name, group, port, namespace, and reference permissions |
| Pool is selected but no backend responds | Endpoint-picker status, pool selector, and model-server readiness |
| Only streamed or large requests fail | ExtProc request-body mode, body limits, and timeouts |

## Production checklist

- Pin a tested set of Gateway API, GAIE, gateway, endpoint-picker, Router, and
  model-server releases.
- Decide whether ExtProc failure is fail-open or fail-closed for each route.
- Keep provider credentials and gateway TLS material in their owning Secret
  workflows, not in Router YAML.
- Test direct pool access, semantic pool selection, and endpoint scheduling as
  separate failure domains before enabling the combined path.
