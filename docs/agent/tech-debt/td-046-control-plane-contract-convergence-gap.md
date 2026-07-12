# TD046: Control Planes Do Not Share One Strict, Transactional Router Contract

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

Production hardening; remediation is split from the #2375 audit proof-of-fix.
Primary strict-contract tracker:
[#2469](https://github.com/vllm-project/semantic-router/issues/2469), coordinated
with transactional mutation
[#2326](https://github.com/vllm-project/semantic-router/issues/2326), rule
validation [#2122](https://github.com/vllm-project/semantic-router/issues/2122),
and operator parity
[#2355](https://github.com/vllm-project/semantic-router/issues/2355).

## Scope

Core config loading, Python CLI models, DSL compilation, dashboard config
mutation, dynamic Kubernetes APIs, and operator CRD/controller translation.

## Summary

Equivalent router configuration enters through several independently validated
representations. Version and unknown-field handling differ, boolean rule trees
have inconsistent validation/semantics, operator raw extensions are not always
validated by the core contract, and dashboard mutations do not share one
generation-aware transaction.

## Evidence

- [Core config loader](../../../src/semantic-router/pkg/config/loader.go)
- [Decision contract](../../../src/semantic-router/pkg/config/decision_config.go)
- [Python CLI models](../../../src/vllm-sr/cli/models.py)
- [Dashboard config handlers](../../../dashboard/backend/handlers/config.go)
- [Dynamic Kubernetes converter](../../../src/semantic-router/pkg/k8s/converter.go)
- [Operator API](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [Operator canonical builder](../../../deploy/operator/controllers/canonical_config_spec.go)

## Why It Matters

The same typo, future version, malformed rule tree, or model reference can be
accepted by one control plane and rejected or interpreted differently by
another. Concurrent writes and rollback can also leave disk, runtime, backups,
and callers observing different generations.

## Desired End State

All ingress paths produce one versioned canonical document and run the same
strict schema plus semantic validators. A shared rule-tree contract owns
operators, shape, depth, and cardinality. One transactional config service owns
generation/CAS, cross-process locking, atomic persistence, validation,
propagation, rollback, and audit events.

## Exit Criteria

- A shared golden corpus has identical accept/reject, field-path, normalized
  version, rule-tree, default, and model-reference results across every control
  plane.
- Unknown versions/fields and malformed boolean trees fail before rollout.
- Canonical and legacy CRD fields cannot silently own the same data.
- Every config mutation uses one CAS-capable transactional writer.
- Concurrency, crash-point, propagation-failure, rollback, and restart tests
  prove disk/runtime/backup generation consistency under `go test -race`.
