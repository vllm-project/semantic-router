# TD015: Weak Typing Still Leaks Through DSL YAML Helpers and Dashboard Config Utilities

## Status

Open

## Scope

- `src/semantic-router/pkg/dsl/compiler.go`
- `src/semantic-router/pkg/dsl/decompiler.go`
- `src/semantic-router/pkg/dsl/emitter_yaml.go`
- `dashboard/frontend/src/types/config.ts`
- `dashboard/frontend/src/pages/configPageCanonicalization.ts`
- `dashboard/frontend/src/pages/configPageRouterDefaultsSupport.ts`
- `dashboard/frontend/src/pages/setupWizardSupport.ts`
- related typed transport and editor seams already cleaned in:
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
  - `dashboard/frontend/src/stores/dslStore.ts`
  - `dashboard/frontend/src/lib/dslMutations.ts`
  - `dashboard/frontend/src/pages/BuilderPage.tsx`
  - `dashboard/frontend/src/pages/builderPageSharedDslEditors.tsx`
  - `dashboard/frontend/src/pages/builderPageVisualShell.tsx`
  - `dashboard/frontend/src/pages/builderPageAddEntityForms.tsx`
  - `dashboard/frontend/src/pages/builderPageEntityDetailView.tsx`
  - `dashboard/frontend/src/pages/builderPageGenericFieldsEditor.tsx`
  - `dashboard/frontend/src/pages/builderPageGlobalSettingsEditor.tsx`
  - `dashboard/frontend/src/pages/builderPageGlobalSettingsObservabilitySections.tsx`
  - `dashboard/frontend/src/pages/builderPageGlobalSettingsRoutingSection.tsx`
  - `dashboard/frontend/src/pages/builderPageGlobalSettingsSafetySection.tsx`
  - `dashboard/frontend/src/pages/builderPageRouteEditorForm.tsx`
  - `dashboard/frontend/src/pages/builderPageAddRouteForm.tsx`
  - `dashboard/frontend/src/pages/builderPageRouteSharedControls.tsx`
  - `deploy/operator/controllers/backend_discovery.go`
  - `deploy/operator/controllers/canonical_config_builder.go`
  - `deploy/operator/controllers/semanticrouter_controller.go`

## Summary

The branch has already removed the worst contract-level weak typing from the router config package: plugin payloads now use `StructuredPayload`, RAG/image-generation backends use typed accessors, canonical global sparse overrides no longer hang off a raw `interface{}`, Hugging Face card unions use explicit string-list wrappers, and the DSL AST JSON bridge exports named recursive node types instead of raw field maps.

The dashboard visual builder and global-editor mutation layer now also uses named recursive field types (`DSLFieldObject` / `DSLFieldValue`) instead of passing `Record<string, unknown>` through the store, page orchestration, and shared editors.

The remaining gaps are narrower and now cluster around three places:

- the Go DSL YAML compile/decompile/emit helpers still fall back to raw `map[string]interface{}` / `interface{}` while formatting arbitrary field bags
- dashboard config type adapters still carry a hybrid canonical-versus-legacy view model and format-detection seam for editor and fallback rendering
- dashboard setup/canonicalization utilities still manipulate open-ended config fragments as mutable record bags instead of named helper payloads

## Evidence

- [compiler.go](../../../src/semantic-router/pkg/dsl/compiler.go)
  - compiler field lowering still emits `map[string]interface{}` / `interface{}` through `fieldsToMap` and `valueToInterface`
- [decompiler.go](../../../src/semantic-router/pkg/dsl/decompiler.go)
  - plugin config normalization still formats through `map[string]interface{}`
- [emitter_yaml.go](../../../src/semantic-router/pkg/dsl/emitter_yaml.go)
  - canonical/CRD emit paths still assemble YAML through raw infra maps
- [config.ts](../../../dashboard/frontend/src/types/config.ts)
  - dashboard config typing still carries `LegacyConfig`, `UnifiedConfig`, and format-detection helpers as a hybrid adapter seam around the canonical contract
- [configPageCanonicalization.ts](../../../dashboard/frontend/src/pages/configPageCanonicalization.ts)
  - legacy-to-canonical promotion still mutates config through `MutableRecord`
- [configPageRouterDefaultsSupport.ts](../../../dashboard/frontend/src/pages/configPageRouterDefaultsSupport.ts)
  - router-default/global-section helpers still traverse mutable `Record<string, unknown>` trees
- [setupWizardSupport.ts](../../../dashboard/frontend/src/pages/setupWizardSupport.ts)
  - first-run setup scaffolding still builds and masks config through open-ended record bags
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
  - [dslStore.ts](../../../dashboard/frontend/src/stores/dslStore.ts)
  - [dslMutations.ts](../../../dashboard/frontend/src/lib/dslMutations.ts)
  - [BuilderPage.tsx](../../../dashboard/frontend/src/pages/BuilderPage.tsx)
  - [builderPageSharedDslEditors.tsx](../../../dashboard/frontend/src/pages/builderPageSharedDslEditors.tsx)
  - [builderPageVisualShell.tsx](../../../dashboard/frontend/src/pages/builderPageVisualShell.tsx)
  - [builderPageAddEntityForms.tsx](../../../dashboard/frontend/src/pages/builderPageAddEntityForms.tsx)
  - [builderPageEntityDetailView.tsx](../../../dashboard/frontend/src/pages/builderPageEntityDetailView.tsx)
  - [builderPageGenericFieldsEditor.tsx](../../../dashboard/frontend/src/pages/builderPageGenericFieldsEditor.tsx)
  - [builderPageGlobalSettingsEditor.tsx](../../../dashboard/frontend/src/pages/builderPageGlobalSettingsEditor.tsx)
  - [builderPageGlobalSettingsObservabilitySections.tsx](../../../dashboard/frontend/src/pages/builderPageGlobalSettingsObservabilitySections.tsx)
  - [builderPageGlobalSettingsRoutingSection.tsx](../../../dashboard/frontend/src/pages/builderPageGlobalSettingsRoutingSection.tsx)
  - [builderPageGlobalSettingsSafetySection.tsx](../../../dashboard/frontend/src/pages/builderPageGlobalSettingsSafetySection.tsx)
  - [builderPageRouteEditorForm.tsx](../../../dashboard/frontend/src/pages/builderPageRouteEditorForm.tsx)
  - [builderPageAddRouteForm.tsx](../../../dashboard/frontend/src/pages/builderPageAddRouteForm.tsx)
  - [builderPageRouteSharedControls.tsx](../../../dashboard/frontend/src/pages/builderPageRouteSharedControls.tsx)
  - [backend_discovery.go](../../../deploy/operator/controllers/backend_discovery.go)
  - [canonical_config_builder.go](../../../deploy/operator/controllers/canonical_config_builder.go)

## Why It Matters

- raw DSL field-bag helpers still hide contract changes from the compiler, so import/export regressions surface late
- dashboard type adapters still preserve more than one effective steady-state config shape in the editor layer, which keeps compatibility logic close to the normal editing surface
- dashboard setup/canonicalization record bags still bypass the same named config seams used by the steady-state editor
- these weakly typed seams undermine the v0.3 canonical-config cleanup by leaving YAML/setup helpers dynamically shaped even after the main editor surface was typed

## Desired End State

- dashboard setup/canonicalization helpers use named helper payloads instead of mutable record bags
- dashboard config typing presents one canonical steady-state editor model, with legacy compatibility isolated behind import or canonicalization seams
- Go DSL compile/decompile/emit helpers use named value abstractions or `yaml.Node` transforms instead of raw maps
- branch-level config/dashboard/DSL changes can be validated without reopening raw field-bag questions in setup/import/export helpers

## Exit Criteria

- dashboard setup/import/defaults helpers no longer require mutable `Record<string, unknown>` bags for config shaping
- dashboard config typing no longer keeps long-lived hybrid `UnifiedConfig` or legacy-only fallback semantics on the same primary editor seam
- DSL compiler/decompile/export hotspots no longer require raw `map[string]interface{}` / `interface{}` field bags
- active config/dashboard/DSL branch diffs can pass harness lint without new weak-typing exemptions for these surfaces
