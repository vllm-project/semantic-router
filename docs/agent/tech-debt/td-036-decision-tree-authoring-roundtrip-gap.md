# TD036: Decision Tree Authoring Cannot Round-Trip Through Runtime Config

## Status

Resolved

## Scope

`deploy/recipes/{balance.dsl,balance.yaml}`,
`src/semantic-router/pkg/dsl/{decision_tree.go,decompiler.go,routing_contract.go}`,
the retention DSL tutorial, and any API or tooling surface that describes
round-trip fidelity between DSL authoring and canonical router config.

## Summary

Current-source recheck confirmed that the repository has intentionally narrowed
`DECISION_TREE` to DSL authoring sugar. The parser lowers tree branches into
flat `config.Decision` entries, runtime config remains the canonical flat
decision model, and `DecompileRouting()` emits flat `ROUTE` blocks. That is the
supported contract rather than a partial implementation of lossless tree
round-trip.

The remaining gap was stale governance and public-facing clarity: the debt entry
still pointed at a removed `spec/dsl.md` file and the maintained recipe and
tutorial docs did not explicitly state that only retention/config fields
round-trip, not the tree shape itself. This is now resolved by codifying the
flat-route decompile contract in tests and documenting the paired DSL/YAML
model for tree-authored recipes.

## Evidence

- [deploy/recipes/balance.dsl](../../../deploy/recipes/balance.dsl)
- [deploy/recipes/balance.yaml](../../../deploy/recipes/balance.yaml)
- [deploy/recipes/README.md](../../../deploy/recipes/README.md)
- [src/semantic-router/pkg/dsl/decision_tree.go](../../../src/semantic-router/pkg/dsl/decision_tree.go)
- [src/semantic-router/pkg/dsl/decompiler.go](../../../src/semantic-router/pkg/dsl/decompiler.go)
- [src/semantic-router/pkg/dsl/routing_contract.go](../../../src/semantic-router/pkg/dsl/routing_contract.go)
- [src/semantic-router/pkg/dsl/dsl_test.go](../../../src/semantic-router/pkg/dsl/dsl_test.go)
- [website/docs/tutorials/decision/retention.md](../../../website/docs/tutorials/decision/retention.md)
- [docs/agent/plans/pl-0012-dsl-conflict-free-routing-workstream.md](../plans/pl-0012-dsl-conflict-free-routing-workstream.md)

## Why It Matters

- Authors can express conflict-free routing with `IF`, `ELSE IF`, and `ELSE`,
  while downstream tooling correctly sees the flat canonical decision model.
- Config APIs, Kubernetes translation, and decompile/export paths no longer need
  to imply that tree metadata is preserved when it is not.
- Maintained examples can keep paired DSL/YAML assets without suggesting that
  runtime config is a lossless source for tree-authored DSL.

## Desired End State

Resolved by the narrowed contract:

- `DECISION_TREE` remains authoring sugar only.
- Canonical router config preserves flat `routing.decisions`, not tree
  authoring metadata.
- `DecompileRouting()` emits flat `ROUTE` blocks for tree-authored input.
- Maintained recipe docs and retention tutorial docs state that tree shape is
  not preserved by config-backed round-trip paths.
- Tests enforce this contract.

## Exit Criteria

- Contributor-visible contract explicitly disclaims decision-tree metadata
  preservation in runtime config and decompile paths.
- DSL tests verify tree-authored input decompiles to flat `ROUTE` blocks and
  recompiles without changing the decision set.
- Maintained recipe and tutorial docs describe the paired-source model and do
  not imply lossless tree-shape round-trip.
