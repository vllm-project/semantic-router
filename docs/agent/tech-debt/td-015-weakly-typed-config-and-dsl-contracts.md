# TD015: Weak Typing Still Leaks Through Dashboard Editor Models and DSL Serialization Helpers

## Status

Open

## Scope

- `dashboard/frontend/src/components/EditModal.tsx`
- `dashboard/frontend/src/pages/ConfigPage.tsx`
- `dashboard/frontend/src/pages/configPageSupport.ts`
- `dashboard/frontend/src/stores/dslStore.ts`
- `dashboard/frontend/src/lib/dslMutations.ts`
- `src/semantic-router/pkg/dsl/decompiler.go`
- `src/semantic-router/pkg/dsl/emitter_yaml.go`
- related typed transport and validation seams already cleaned in:
  - `src/semantic-router/pkg/config/canonical_config.go`
  - `src/semantic-router/pkg/config/canonical_global.go`
  - `src/semantic-router/pkg/config/plugin_config.go`
  - `src/semantic-router/pkg/config/rag_plugin.go`
  - `src/semantic-router/pkg/config/image_gen_plugin.go`
  - `src/semantic-router/pkg/config/registry_hf.go`
  - `src/semantic-router/pkg/dsl/ast_json.go`
  - `dashboard/backend/handlers/config.go`
  - `dashboard/backend/handlers/config_global_raw.go`
  - `dashboard/backend/handlers/deploy.go`
  - `dashboard/backend/handlers/setup.go`
  - `deploy/operator/controllers/backend_discovery.go`
  - `deploy/operator/controllers/canonical_config_builder.go`
  - `deploy/operator/controllers/semanticrouter_controller.go`

## Summary

The branch has already removed the worst contract-level weak typing from the router config package: plugin payloads now use `StructuredPayload`, RAG/image-generation backends use typed accessors, canonical global sparse overrides no longer hang off a raw `interface{}`, Hugging Face card unions use explicit string-list wrappers, and the DSL AST JSON bridge exports named recursive node types instead of raw field maps.

The remaining gaps are narrower and now cluster around two places:

- the dashboard manager/editor still uses schema-driven generic form state and `any` in modal/edit flows
- the DSL YAML emit/decompile helpers still fall back to raw `map[string]interface{}` / `interface{}` while formatting arbitrary field bags

## Evidence

- [EditModal.tsx](../../../dashboard/frontend/src/components/EditModal.tsx)
  - generic modal state still typed as `any`
- [ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
  - edit modal callbacks and config mutation flow still typed as `any`
- [configPageSupport.ts](../../../dashboard/frontend/src/pages/configPageSupport.ts)
  - several global/editor sections still use `Record<string, unknown>` or array-of-record fallbacks
- [dslStore.ts](../../../dashboard/frontend/src/stores/dslStore.ts)
  - builder mutation APIs still pass field bags as `Record<string, unknown>`
- [dslMutations.ts](../../../dashboard/frontend/src/lib/dslMutations.ts)
  - route/model/plugin serialization helpers still use unbounded field maps
- [decompiler.go](../../../src/semantic-router/pkg/dsl/decompiler.go)
  - plugin config normalization still formats through `map[string]interface{}`
- [emitter_yaml.go](../../../src/semantic-router/pkg/dsl/emitter_yaml.go)
  - canonical/CRD emit paths still assemble YAML through raw infra maps
- This change already retired the worst transport-level weak typing in:
  - [canonical_config.go](../../../src/semantic-router/pkg/config/canonical_config.go)
  - [canonical_global.go](../../../src/semantic-router/pkg/config/canonical_global.go)
  - [plugin_config.go](../../../src/semantic-router/pkg/config/plugin_config.go)
  - [rag_plugin.go](../../../src/semantic-router/pkg/config/rag_plugin.go)
  - [image_gen_plugin.go](../../../src/semantic-router/pkg/config/image_gen_plugin.go)
  - [registry_hf.go](../../../src/semantic-router/pkg/config/registry_hf.go)
  - [ast_json.go](../../../src/semantic-router/pkg/dsl/ast_json.go)
  - [config.go](../../../dashboard/backend/handlers/config.go)
  - [config_global_raw.go](../../../dashboard/backend/handlers/config_global_raw.go)
  - [deploy.go](../../../dashboard/backend/handlers/deploy.go)
  - [setup.go](../../../dashboard/backend/handlers/setup.go)
  - [backend_discovery.go](../../../deploy/operator/controllers/backend_discovery.go)
  - [canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)

## Why It Matters

- dashboard generic editor state weakens manager/global safety exactly where users expect config surfaces to be authoritative
- raw DSL field-bag helpers still hide contract changes from the compiler, so builder/import/export regressions surface late
- these weakly typed seams undermine the v0.3 canonical-config cleanup by leaving the last high-traffic authoring paths dynamically shaped

## Desired End State

- dashboard editor state and callbacks use typed per-surface models or reusable typed form abstractions, not `any`/unbounded record blobs
- DSL builder/store/export helpers use named field/value abstractions instead of raw maps
- branch-level config/dashboard/DSL changes can be validated without reopening raw field-bag questions in each surface

## Exit Criteria

- dashboard config editor hotspots no longer require `any` for modal state and save callbacks
- DSL builder/store/export hotspots no longer require raw `map[string]interface{}` / `Record<string, unknown>` field bags
- active config/dashboard/DSL branch diffs can pass harness lint without new weak-typing exemptions for these surfaces
