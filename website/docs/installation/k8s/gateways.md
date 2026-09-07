---
title: Kubernetes Gateways
description: Choose how a Kubernetes gateway calls Semantic Router through Envoy ExtProc.
---

# Kubernetes Gateways

Semantic Router can run behind several Envoy-based Kubernetes gateways. The
routing policy stays the same; the gateway-specific resources determine when
ExtProc runs, how the selected model reaches a backend, and which component
owns authentication or traffic policy.

## Choose an integration

| Existing data plane | Start with | What it owns |
| --- | --- | --- |
| Envoy AI Gateway | [Envoy AI Gateway](ai-gateway) | Provider translation, provider credentials, rate limits, and Gateway API traffic policy. |
| agentgateway | [agentgateway](agentgateway) | Gateway API proxy, backend resources, and ExtProc phase policy. |
| Istio | [Istio Gateway](istio) | Ingress, `HTTPRoute` processing, and the Envoy filter that calls Semantic Router. |
| Gateway API Inference Extension | [GIE](gateway-api-inference-extension) | `InferencePool` endpoint selection after Semantic Router chooses a model pool. |

Use the gateway already operated by your platform. Do not install a second
gateway only to obtain semantic routing unless you have compared ownership,
security policy, and upgrade requirements.

## Shared contract

All integrations must agree on:

1. the public model or entrypoint requested by the client;
2. the model name written by Semantic Router;
3. the Gateway API match or provider backend for that model; and
4. the served model identity accepted by the inference endpoint.

The gateway owns client authentication and transport policy unless your
deployment explicitly assigns those controls elsewhere. Semantic signals such
as PII or jailbreak detection do not block traffic by themselves; a decision or
plugin must enforce the intended action.

## Request buffering and streaming

Start with the processing mode required by the selected integration. Change it
only when request size or immediate streamed responses require it. See
[Streamed ExtProc](streamed-extproc) for body buffering, mode override, and
streaming constraints.

## Verify before production

- send a direct request to each backend;
- send the same request through the gateway;
- inspect the selected-model and decision headers;
- confirm the gateway resolved the selected backend; and
- test credentials, request limits, streaming, and failure behavior.

[Test a Kubernetes Gateway Deployment](gateway-testing) provides a common
checklist without assuming cluster-assigned addresses.
