---
title: Backend Target Compatibility
description: See which backend target forms survive Docker, Helm, Operator, Dashboard, and recipe workflows.
---

# Backend Target Compatibility

Semantic Router uses `providers.models[].backend_refs[]` as the canonical
contract between a logical model name and one or more physical inference
targets.

This matrix covers **configuration production and preservation**. It does not
claim that an endpoint is reachable, healthy, or compatible with a particular
model protocol.

## Choose the right matrix

| Question | Source of truth |
| --- | --- |
| Which client endpoints and backend wire protocols work together? | [Protocol Compatibility](protocol-compatibility) |
| Which target fields survive CLI, Helm, Operator, Dashboard, and recipe workflows? | This page |
| Which deployment stacks and integrations does the project maintain? | [Deployment Support](support-matrix) |

The same canonical document can move through the CLI, Helm, and Dashboard. The
Operator instead accepts Kubernetes discovery inputs and translates its
supported subset into canonical backend references.

## Status meanings

- **Supported**: the surface accepts and preserves the canonical target form.
- **Adapter**: the surface accepts a narrower native input and generates a
  canonical backend reference.
- **Partial**: the form is accepted with the limitation stated below.
- **Not expressible**: the producer cannot represent that target form.

## Compatibility matrix

<!-- BEGIN BACKEND TARGET COMPATIBILITY MATRIX -->

| Target form | Canonical YAML | Docker / CLI | Helm | Operator | Dashboard | Maintained recipes |
| --- | --- | --- | --- | --- | --- | --- |
| Direct `endpoint` as `host[:port]` | Supported | Supported | Supported | Adapter | Supported | Supported |
| HTTP(S) `base_url`, including a path | Supported | Supported | Supported | Not expressible | Supported | Supported |
| Multiple weighted refs with shared route metadata | Supported | Supported | Supported | Adapter | Supported | Supported |
| Provider, API-version, path, and header metadata | Supported | Supported | Supported | Not expressible | Supported | Supported |
| Kubernetes Service DNS target | Supported | Supported | Supported | Adapter | Supported | Supported |
| KServe discovery | Not expressible | Not expressible | Not expressible | Partial | Not expressible | Not expressible |
| Label-selected Service discovery | Not expressible | Not expressible | Not expressible | Adapter | Not expressible | Not expressible |
| Different paths or request headers per weighted ref | Supported | Partial | Supported | Not expressible | Supported | Partial |

<!-- END BACKEND TARGET COMPATIBILITY MATRIX -->

The Docker / CLI path generates one route for a logical model. When a model has
several weighted refs, request headers, Host rewriting, and TLS SNI come from
the first ref. A path prefix is applied only when every ref uses the same path;
otherwise the local generator omits path rewriting. Keep those route-level
properties compatible across a model's refs. Endpoint or replica selection
using live inference telemetry is a separate data-plane contract tracked in
[#2332](https://github.com/vllm-project/semantic-router/issues/2332).

## Portable target forms

Use `endpoint` for a direct host and optional port:

```yaml
providers:
  models:
    - name: local/general
      provider_model_id: Qwen/Qwen3-8B
      api_format: openai
      backend_refs:
        - name: primary
          endpoint: model-server.default.svc.cluster.local:8000
          protocol: http
          provider: vllm
          weight: 100
```

Use `base_url` when the upstream identity includes a scheme or path. Keep
credentials in environment references rather than committed YAML:

```yaml
providers:
  models:
    - name: hosted/reasoning
      provider_model_id: provider/model-id
      api_format: openai
      backend_refs:
        - name: hosted-primary
          base_url: https://provider.example/v1
          provider: openai
          auth_header: Authorization
          auth_prefix: Bearer
          api_key_env: PROVIDER_API_KEY
          extra_headers:
            X-Tenant: production
          weight: 100
```

For portable configuration, do not put a URL scheme or path in `endpoint`.
Some local generation paths accept those forms, but not every maintained
producer agrees on their meaning. `base_url` is the canonical URL form.

`api_format` belongs to the model, not to an individual backend ref. It selects
the backend request and response codec; `protocol` on a backend ref selects the
HTTP transport. See [Protocol Compatibility](protocol-compatibility) before
choosing `api_format`.

## Producer behavior

| Producer | Behavior and boundary |
| --- | --- |
| Docker / local CLI | Translates refs into Envoy clusters and routes. It preserves host, port, HTTP or HTTPS, weight, a shared path prefix, environment-resolved authorization, and shared extra headers. Referenced model servers must already be reachable. |
| Helm | `configOverride` renders one complete canonical mapping without merging sample provider defaults. Reachability and model compatibility remain runtime checks. |
| Operator | `spec.vllmEndpoints[]` is a Kubernetes discovery adapter, not a copy of the full provider schema. It emits the supported canonical subset described below. |
| Dashboard | Reads and writes the supported canonical backend inventory, including provider identity, URL, auth metadata, API version, chat path, extra headers, and environment-key references. |

The Operator currently discovers:

- a named Kubernetes Service;
- a KServe `InferenceService`; and
- a label-selected Llama Stack Service.

Those adapters generate backend name, endpoint, protocol, and weight. Put a
complete external-provider target in canonical configuration supplied through
Helm or another canonical-config workflow. Broader CRD and Helm parity belongs
to [#2355](https://github.com/vllm-project/semantic-router/issues/2355).

KServe discovery currently assumes the conventional
`<InferenceService>-predictor.<namespace>.svc.cluster.local:8443` HTTPS target.
It does not resolve `status.url` or inspect the generated Service, so custom or
named-predictor service layouts require an explicit Service backend instead.

Unknown-field and supported-version parity across producers is tracked in
[#2469](https://github.com/vllm-project/semantic-router/issues/2469). Until that
work lands, do not rely on an unknown extension surviving a move between
producers.

## Validation boundary

The repository exercises this matrix at five layers:

- CLI tests validate direct targets, URL paths, TLS, weights, and headers in
  generated Envoy configuration.
- Helm validation renders a complete canonical override and checks that its
  backend fields survive without sample-default leakage.
- Operator tests validate Service discovery and the generated canonical ref.
- Dashboard frontend and backend tests validate the supported field inventory
  and persisted canonical output.
- The maintained-config contract parses its enumerated asset inventory and
  every recipe. The separate reference-config contract validates the shipped
  reference configuration, which exercises weighted direct and rich URL
  targets.

These checks prove configuration translation and preservation. They do not
replace a direct backend protocol check or a live request through the Router.
