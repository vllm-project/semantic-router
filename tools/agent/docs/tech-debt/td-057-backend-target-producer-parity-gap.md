# TD057: Backend-Target Producer Parity Gap

## Status

Open.

## Owner Plan

[PL-0032: Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

## Release Relevance

Configuration correctness. Moving a backend target between supported
deployment surfaces can currently reject, omit, or reinterpret fields without
one shared producer contract.

## Scope

- local CLI canonical configuration models and generation;
- Operator discovery adapters;
- Dashboard normalization and persistence;
- canonical backend-reference validation.

## Summary

The canonical backend reference supports direct endpoints, URL targets,
weights, paths, provider identity, and bounded auth or header metadata. Its
producers and consumers do not yet share one validation and preservation
policy: the Operator intentionally emits a discovery subset, the local CLI and
Dashboard do not preserve unknown extensions, and local routing applies path
and header metadata at the logical-model route rather than independently per
weighted target.

## Evidence

- `deploy/operator/controllers/backend_discovery.go` emits backend name,
  endpoint, protocol, and weight from the maintained discovery adapters.
- `src/vllm-sr/cli/models.py` models the known canonical fields but
  does not reject every unsupported target shape or unknown extension.
- `dashboard/frontend/src/pages/configPageModelFormSupport.ts` normalizes the
  known backend-field inventory.
- `src/vllm-sr/cli/config_generator.py` derives model-route path and headers
  from one backend ref for a weighted logical model.

## Why It Matters

An accepted configuration is not necessarily portable across the local CLI,
Helm, Operator, and Dashboard. Silent field loss or per-target metadata
collapse can direct traffic to the wrong upstream path or omit required
provider metadata.

## Desired End State

Every producer declares the backend-target forms it accepts, rejects unsupported
forms explicitly, and preserves the canonical fields it claims to support.
Translation limits remain visible in the public compatibility matrix and in
focused executable tests.

## Exit Criteria

- Complete unknown-field and supported-version parity tracked by
  [#2469](https://github.com/vllm-project/semantic-router/issues/2469).
- Define and implement the intended Operator and Helm schema parity tracked by
  [#2355](https://github.com/vllm-project/semantic-router/issues/2355).
- Reject divergent per-ref path and request-header metadata explicitly, or
  implement independent per-target routing without collapsing it to the first
  backend ref.
- Keep live inference-aware endpoint selection within
  [#2332](https://github.com/vllm-project/semantic-router/issues/2332).
- Retain focused CLI, Helm, Operator, Dashboard, and maintained-config tests
  aligned with the public backend-target compatibility matrix.
