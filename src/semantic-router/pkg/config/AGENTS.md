# Config Package Notes

## Scope

- `src/semantic-router/pkg/config/**`
- local rules for config schema hotspots, especially `config.go`

## Responsibilities

- Keep `config.go` focused on the central schema table and shared config contracts.
- Treat adjacent files as the place for plugin contracts, validators, and load/registry helpers.
- Keep canonical import/export and normalization in `canonical_*.go` instead of drifting that logic back into `config.go`.
- Keep plugin-family-specific contracts and backend decoders in dedicated files; do not let one plugin hotspot become a second schema table.
- Keep semantic validation split by family; `validator.go` can coordinate shared checks, but broad type-specific logic should move to focused support files.
- Treat canonical `version/listeners/providers/routing/global` parsing as the only steady-state runtime contract.
- Keep migration-only compatibility out of the runtime parser; legacy user layouts belong in explicit migration tooling, not in normal config loading.
- Keep canonical `providers` split readable:
  - `providers.defaults` owns default selection and reasoning-family metadata
  - `providers.models[]` owns concrete backend access bindings directly
- Keep canonical `global` layered, not flat:
  - `global.router` for router-engine control knobs
  - `global.services` for shared APIs and control-plane services
  - `global.stores` for shared storage-backed services
  - `global.integrations` for helper runtime integrations
  - `global.model_catalog` for router-owned model assets, stable system-model bindings, and module configs that resolve through that shared model catalog
- Keep layer-specific contracts distinct:
  - signal definitions own extraction-oriented config
  - decision config owns boolean composition and control logic
  - algorithm config owns per-decision model-selection policy
  - plugin config owns post-decision processing behavior
  - global config owns intentionally cross-cutting behavior
  - routing-surface catalogs own the supported signal/plugin/algorithm inventories used by examples, validators, and tests

## Change Rules

- Do not add new plugin structs, helper decoders, or utility walkers back into `config.go` if they can live in an adjacent file.
- Do not move canonical import/export or normalization helpers back into `config.go`; extend `canonical_*.go` or add focused support files instead.
- Do not collapse signal, decision, algorithm, plugin, and global config into one catch-all struct when separate contracts or support files can keep ownership clear.
- If you change supported signals, decision algorithms, or decision plugins, update the router-owned surface catalog and the repo-owned `config/` fragment tree in the same change.
- Config-contract changes must update the relevant public docs and proposal in the same change:
  - `config/README.md`
  - `website/docs/installation/configuration.md`
  - `website/docs/proposals/unified-config-contract-v0-3.md`
  - `website/docs/tutorials/{signal,decision,algorithm,plugin,global}/`
- Preferred split:
  - core schema tables in `config.go`
  - signal / decision / algorithm / plugin contracts in dedicated support files where the schema is large enough to justify separation
  - plugin contracts/helpers in dedicated `*_plugin.go` or support files
  - validation in `validator.go`
  - load/registry behavior in `loader.go` and `registry.go`
- `validator.go` is a ratcheted hotspot. Keep the package entrypoint there, but move modality/image-gen/RAG/provider or other family-specific checks into sibling validator files when extending them.
- `rag_plugin.go` is a plugin-family hotspot. New backend-specific payload types, decode helpers, or validation branches should move into sibling support files before that file grows further.
- Keep fragment-catalog enforcement tests adjacent to the config package so drift is caught by `go test ./pkg/config/...`.
- Keep exhaustive `config/config.yaml` reference-config enforcement adjacent to the config package too, and wire it into harness lint so public-schema drift is blocked before merge.
- Keep maintained `deploy/` and `e2e/` router config assets under the same guard as the canonical contract so repo-owned examples and harness profiles do not regress to legacy steady-state fields.
- Keep config-doc consistency tests adjacent to the config package too, so canonical terminology and fragment-directory guidance stay aligned with the implemented contract.
- Keep the latest tutorial tree aligned to the canonical doc taxonomy: `signal/decision/algorithm/plugin/global`, not ad hoc feature folders.
- External control-plane consumers should use `src/semantic-router/pkg/routercontract/` and `src/semantic-router/pkg/routerauthoring/` instead of importing `pkg/config` or `pkg/dsl` directly.
- If a change touches `config.go`, prefer a net reduction in file size or responsibility count.
