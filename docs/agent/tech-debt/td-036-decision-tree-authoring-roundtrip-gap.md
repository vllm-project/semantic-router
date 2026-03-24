# TD036: Decision Tree Authoring Cannot Round-Trip Through Runtime Config

## Status

Open

## Scope

`spec/dsl.md`, `deploy/recipes/{balance.dsl,balance.yaml}`, `src/semantic-router/pkg/dsl/{decision_tree.go,decompiler.go}`, and any API or tooling surface that claims round-trip fidelity between DSL authoring and canonical router config

## Summary

The repository now supports `DECISION_TREE` / `IF ELSE` authoring by lowering tree branches into flat `config.Decision` entries at parse time. That keeps runtime routing, the config API, and canonical router config on one existing model, but it also means the original tree structure is discarded before decompile and API consumers ever see it. `DecompileRouting()` still reconstructs plain `ROUTE` blocks from `config.Decision`, so a DSL file authored as a decision tree cannot be losslessly recovered from runtime config or from the core config endpoints. The maintained `balance` recipe pair is therefore the current explicit paired-source model for showcasing richer DSL authoring without claiming runtime-config tree round-trip.

## Evidence

- [spec/dsl.md](../../../spec/dsl.md)
- [deploy/recipes/balance.dsl](../../../deploy/recipes/balance.dsl)
- [deploy/recipes/balance.yaml](../../../deploy/recipes/balance.yaml)
- [src/semantic-router/pkg/dsl/decision_tree.go](../../../src/semantic-router/pkg/dsl/decision_tree.go)
- [src/semantic-router/pkg/dsl/decompiler.go](../../../src/semantic-router/pkg/dsl/decompiler.go)
- [src/semantic-router/pkg/apiserver/route_classification_config.go](../../../src/semantic-router/pkg/apiserver/route_classification_config.go)

## Why It Matters

- Authors can now express conflict-free routing with `IF`, `ELSE IF`, and `ELSE`, but downstream tooling still only sees flattened routes and cannot recover the original partitioning intent.
- The core config API exposes canonical router config, not DSL source. Without preserved tree metadata, API-backed editors or future migration tools will silently degrade tree authoring into route lists.
- Maintained examples currently need both a DSL source file and a compiled YAML artifact because neither runtime config nor decompile can serve as the sole source of truth for tree-authored configs.
- The repo has already narrowed its public contract to `DECISION_TREE` as authoring sugar only, so debt remains open until every doc/example/API claim matches that narrower posture.

## Desired End State

- The repository has one explicit contract for `DECISION_TREE` round-trip behavior across authoring, config APIs, and decompile.
- Either canonical router config preserves enough decision-tree metadata to reconstruct `DECISION_TREE` authoring, or the repo narrows the supported round-trip claim and makes the paired-source model explicit across APIs, examples, and tooling.
- Tests enforce the chosen contract so tree-authored configs do not silently degrade across future compiler or API changes.

## Exit Criteria

- Canonical router config and the related API/tooling surfaces either preserve or explicitly disclaim decision-tree authoring metadata in one contributor-visible contract.
- `DecompileRouting()` or an equivalent supported export path can reproduce `DECISION_TREE` syntax when the repo claims tree round-trip support.
- Maintained examples, tests, and docs no longer rely on an implicit assumption that tree-authored DSL can be losslessly recovered from runtime config when it cannot.
