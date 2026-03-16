# TD015: Weakly Typed Config and DSL Contracts Still Leak Through Plugin, RAG, AST, and Dashboard Editor Paths

## Status

Open

## Scope

- `src/semantic-router/pkg/config/plugin_config.go`
- `src/semantic-router/pkg/config/rag_plugin.go`
- `src/semantic-router/pkg/dsl/ast_json.go`
- `dashboard/frontend/src/components/EditModal.tsx`
- `dashboard/frontend/src/pages/ConfigPage.tsx`
- `dashboard/frontend/src/pages/configPageSupport.ts`
- related typed transport and validation seams already cleaned in:
  - `dashboard/backend/handlers/config.go`
  - `dashboard/backend/handlers/config_global_raw.go`
  - `dashboard/backend/handlers/deploy.go`
  - `dashboard/backend/handlers/setup.go`
  - `deploy/operator/controllers/backend_discovery.go`
  - `deploy/operator/controllers/canonical_config_builder.go`
  - `deploy/operator/controllers/semanticrouter_controller.go`

## Summary

The branch has moved the operator and dashboard backend config transport onto typed canonical v0.3 structs, but several active config and DSL surfaces still rely on raw `interface{}`, `map[string]interface{}`, or broad frontend `any`/`Record<string, unknown>` state.

The remaining gaps cluster around four places:

- decision plugins still store `configuration` as an untyped raw payload
- RAG backend config still uses untyped backend unions and raw filter/argument maps
- the DSL AST JSON bridge still serializes fields and boolean expressions through `map[string]interface{}` / `interface{}`
- the dashboard manager/editor still uses schema-driven generic form state instead of typed section models

## Evidence

- [plugin_config.go](../../../src/semantic-router/pkg/config/plugin_config.go)
  - `DecisionPlugin.Configuration interface{}`
  - `GetPluginConfig(...) interface{}`
  - map conversion helpers for raw YAML payloads
- [rag_plugin.go](../../../src/semantic-router/pkg/config/rag_plugin.go)
  - `RAGPluginConfig.BackendConfig interface{}`
  - `MCPRAGConfig.ToolArguments map[string]interface{}`
  - `OpenAIRAGConfig.Filter map[string]interface{}`
  - `HybridRAGConfig.PrimaryConfig/FallbackConfig interface{}`
- [ast_json.go](../../../src/semantic-router/pkg/dsl/ast_json.go)
  - `Fields map[string]interface{}`
  - `When interface{}`
  - `marshalValue(...) interface{}`
  - `marshalBoolExpr(...) interface{}`
- [EditModal.tsx](../../../dashboard/frontend/src/components/EditModal.tsx)
  - generic modal state still typed as `any`
- [ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
  - edit modal callbacks and config mutation flow still typed as `any`
- [configPageSupport.ts](../../../dashboard/frontend/src/pages/configPageSupport.ts)
  - several global/editor sections still use `Record<string, unknown>` or array-of-record fallbacks
- This change already retired the worst transport-level weak typing in:
  - [config.go](../../../dashboard/backend/handlers/config.go)
  - [config_global_raw.go](../../../dashboard/backend/handlers/config_global_raw.go)
  - [deploy.go](../../../dashboard/backend/handlers/deploy.go)
  - [setup.go](../../../dashboard/backend/handlers/setup.go)
  - [backend_discovery.go](../../../deploy/operator/controllers/backend_discovery.go)
  - [canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)

## Why It Matters

- raw plugin and RAG payloads make config behavior harder to validate and easier to drift across router, dashboard, DSL, and operator surfaces
- AST maps hide contract changes from the compiler, so DSL/frontend regressions surface late
- dashboard generic editor state weakens manager/global safety exactly where users expect config surfaces to be authoritative
- these weakly typed seams undermine the v0.3 canonical-config cleanup by leaving high-traffic configuration paths dynamically shaped

## Desired End State

- decision plugin config uses an explicit typed envelope or tagged union instead of raw `interface{}`
- RAG backend configuration uses explicit typed backend envelopes for each supported backend, including typed tool arguments and metadata filters where the contract is known
- DSL AST JSON exports explicit JSON structs or tagged union nodes instead of `map[string]interface{}` / `interface{}`
- dashboard editor state and callbacks use typed per-surface models or reusable typed form abstractions, not `any`/unbounded record blobs
- branch-level config/dashboard/DSL changes can be validated without reopening raw config-shape questions in each surface

## Exit Criteria

- `DecisionPlugin.Configuration` no longer uses `interface{}`
- `RAGPluginConfig.BackendConfig` and hybrid backend configs no longer use `interface{}`
- `ast_json.go` no longer exports field or bool-expression JSON through `map[string]interface{}` / `interface{}`
- dashboard config editor hotspots no longer require `any` for modal state and save callbacks
- active config/dashboard/DSL branch diffs can pass harness lint without new weak-typing exemptions for these surfaces
