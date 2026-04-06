# TD015: Weak Typing Still Leaks Through DSL YAML Helpers and Dashboard Config Utilities

## Status

Closed

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

The branch has now removed the remaining contract-level weak typing that was still leaking through the dashboard config editor seams and the DSL compile/decompile/export helpers. The steady-state dashboard editor model now treats the fetched config as canonical `v0.3` config instead of preserving a long-lived hybrid format adapter, setup/canonicalization helpers use named config-tree payloads instead of mutable record bags, and the DSL compiler/decompiler/emitter now route arbitrary payload shaping through named `StructuredPayload`, `JSONObject` / `JSONValue`, and `YAMLObject` / `YAMLList` abstractions instead of raw `map[string]interface{}` field bags.

The earlier branch work that retired transport-level weak typing in router config structs, backend handlers, and the dashboard visual builder remains intact; this change closes the last editor/setup/export seams that kept reopening raw field-bag questions.

## Evidence

- [compiler.go](../../../src/semantic-router/pkg/dsl/compiler.go)
- structure/rag payload lowering now uses `StructuredPayload` helpers instead of raw field-bag conversion
- [decompiler.go](../../../src/semantic-router/pkg/dsl/decompiler.go)
  - plugin config normalization now formats through `JSONObject` / `JSONValue`
- [emitter_yaml.go](../../../src/semantic-router/pkg/dsl/emitter_yaml.go)
  - user/ordered/CRD/Helm emit paths now use named YAML tree abstractions
- [json_value_codec.go](../../../src/semantic-router/pkg/dsl/json_value_codec.go)
  - shared structured-payload and `JSONValue` formatting codec now owns recursive DSL payload encoding
- [yaml_tree.go](../../../src/semantic-router/pkg/dsl/yaml_tree.go)
  - emit helpers now share named YAML tree abstractions instead of raw YAML maps
- [config.ts](../../../dashboard/frontend/src/types/config.ts)
  - the dashboard config type surface no longer carries `LegacyConfig`, `UnifiedConfig`, or format-detection helpers
- [configTree.ts](../../../dashboard/frontend/src/types/configTree.ts)
  - recursive config-tree payloads now have a named dashboard-side contract
- [configPageCanonicalization.ts](../../../dashboard/frontend/src/pages/configPageCanonicalization.ts)
  - legacy-to-canonical promotion now isolates compatibility behind named legacy carrier/helper payloads
- [configPageRouterDefaultsSupport.ts](../../../dashboard/frontend/src/pages/configPageRouterDefaultsSupport.ts)
  - router-default/global-section helpers now traverse named config-tree payloads
- [setupWizardSupport.ts](../../../dashboard/frontend/src/pages/setupWizardSupport.ts)
  - first-run setup scaffolding now builds named setup payloads and keeps canonical routing ownership explicit
- [ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
  - the page now treats manager state as canonical config instead of preserving a legacy-vs-canonical format mode
- [ConfigPageSignalsSection.tsx](../../../dashboard/frontend/src/pages/ConfigPageSignalsSection.tsx)
  - signal rendering now consumes canonical `config.signals` instead of a flat legacy fallback
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

## Retirement Notes

- `dashboard/frontend/src/types/config.ts` no longer carries the hybrid `LegacyConfig` / `UnifiedConfig` adapter seam, and `ConfigPage.tsx` now treats fetched manager config as canonicalized `v0.3` state.
- `dashboard/frontend/src/types/configTree.ts`, `configPageCanonicalization.ts`, `configPageRouterDefaultsSupport.ts`, and `setupWizardSupport.ts` now share named config-tree/setup payloads, so legacy import and global-default shaping no longer depend on anonymous mutable record bags.
- `ConfigPageSignalsSection.tsx` now renders the canonical `config.signals` view only; the legacy flat-signal fallback remains isolated inside canonicalization instead of the steady-state editor surface.
- `src/semantic-router/pkg/dsl/compiler.go`, `validator.go`, and `json_value_codec.go` now encode structure and plugin payloads through `StructuredPayload` plus `JSONObject` / `JSONValue` helpers rather than `fieldsToMap` / `valueToInterface`.
- `src/semantic-router/pkg/dsl/decompiler.go`, `emitter_yaml.go`, and `yaml_tree.go` now format exported DSL/YAML through named JSON/YAML tree abstractions instead of raw `map[string]interface{}` / `interface{}` field bags.

## Validation

- `cd /Users/bitliu/vs/dashboard/frontend && npm run type-check`
- `cd /Users/bitliu/vs/dashboard/frontend && npm run lint`
- `cd /Users/bitliu/vs/src/semantic-router && go test ./pkg/dsl`
- `make agent-validate`
- `make agent-lint AGENT_CHANGED_FILES_PATH=/tmp/vsr_td015_changed.txt`
- `make agent-ci-gate AGENT_CHANGED_FILES_PATH=/tmp/vsr_td015_changed.txt`
