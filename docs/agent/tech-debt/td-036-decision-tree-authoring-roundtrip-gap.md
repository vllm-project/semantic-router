# TD036: Decision Tree Authoring Cannot Round-Trip Through Runtime Config

## Status

Closed

## Scope

`deploy/recipes/{balance.dsl,balance.yaml}`, `src/semantic-router/pkg/dsl/{decision_tree.go,decompiler.go,dsl_test.go,maintained_asset_roundtrip_test.go}`, `src/semantic-router/pkg/apiserver/{route_api_doc.go,route_config_deploy.go,route_config_runtime_sync_test.go}`, and contributor-facing docs that define the DSL/config/API contract

## Summary

The repository now has one explicit paired-source contract for `DECISION_TREE` / `IF ELSE`: tree syntax is DSL authoring sugar only, compile lowers it into flat `routing.decisions`, `DecompileRouting()` exports flat `ROUTE` blocks, `/config/router` exposes canonical config only, and any optional API `dsl` payload is archived as source text rather than treated as runtime metadata. The maintained `balance` DSL/YAML pair and focused regression tests now enforce that narrower contract directly.

## Evidence

- [deploy/recipes/balance.dsl](../../../deploy/recipes/balance.dsl)
- [deploy/recipes/balance.yaml](../../../deploy/recipes/balance.yaml)
- [src/semantic-router/pkg/dsl/decision_tree.go](../../../src/semantic-router/pkg/dsl/decision_tree.go)
- [src/semantic-router/pkg/dsl/decompiler.go](../../../src/semantic-router/pkg/dsl/decompiler.go)
- [src/semantic-router/pkg/dsl/dsl_test.go](../../../src/semantic-router/pkg/dsl/dsl_test.go)
- [src/semantic-router/pkg/dsl/maintained_asset_roundtrip_test.go](../../../src/semantic-router/pkg/dsl/maintained_asset_roundtrip_test.go)
- [src/semantic-router/pkg/apiserver/route_api_doc.go](../../../src/semantic-router/pkg/apiserver/route_api_doc.go)
- [src/semantic-router/pkg/apiserver/route_config_deploy.go](../../../src/semantic-router/pkg/apiserver/route_config_deploy.go)
- [src/semantic-router/pkg/apiserver/route_config_runtime_sync_test.go](../../../src/semantic-router/pkg/apiserver/route_config_runtime_sync_test.go)
- [website/docs/installation/configuration.md](../../../website/docs/installation/configuration.md)
- [website/docs/api/apiserver.md](../../../website/docs/api/apiserver.md)

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

- Satisfied on 2026-04-06: canonical router config, `/config/router`, and the contributor-facing API/config docs now explicitly disclaim decision-tree metadata preservation and state that tree syntax lowers into flat `routing.decisions`.
- Satisfied on 2026-04-06: `DecompileRouting()` is covered by regression tests that lock the current flat `ROUTE` export contract for tree-authored input instead of implying round-trip support.
- Satisfied on 2026-04-06: the maintained `balance` recipe pair declares the paired-source model directly, and tests enforce those markers so examples and docs do not imply lossless tree recovery from runtime config.

## Retirement Notes

- `RouterConfigUpdateRequest.DSL` is now documented and tested as an archived authoring source only, not as canonical config metadata.
- `GET /config/router` and the router apiserver docs now describe the canonical flat-config contract explicitly for tree-authored DSL.
- The maintained `balance` DSL/YAML assets now declare their paired-source responsibilities in-file, and regression tests fail if that contract drifts.
- Future work should reopen this item only if the repository decides to preserve decision-tree metadata inside canonical config or to support a separate round-tripping export path.
