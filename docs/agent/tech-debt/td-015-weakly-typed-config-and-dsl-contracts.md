# TD015: Weak Typing Still Leaks Through DSL YAML Helpers and Dashboard Config Utilities

## Status

Open

## Owner Plan

PL0033 v0.3 Themis Release Closure

## Release Relevance

v0.3 Themis

## Scope

Go DSL compile/decompile/emit helpers and dashboard config editor/setup
utilities that still move open-ended config payloads through raw maps or mutable
record bags.

## Summary

The config contract is much more typed than it used to be: router plugin
payloads, canonical global overrides, Hugging Face registry card unions, and
several DSL AST bridges now use named payload structures. The remaining weak
typing is narrower and concentrated in three places:

- Go DSL emit/export paths still assemble infra, CRD, and YAML payloads through
  `map[string]interface{}` or equivalent raw value bags.
- Dashboard config adapters still bridge more than one effective editor shape
  instead of exposing one canonical steady-state model.
- Setup and config-page utilities still mutate open-ended config fragments
  through generic records.

## Evidence

- [src/semantic-router/pkg/dsl/compiler.go](../../../src/semantic-router/pkg/dsl/compiler.go)
- [src/semantic-router/pkg/dsl/field_payload.go](../../../src/semantic-router/pkg/dsl/field_payload.go)
- [src/semantic-router/pkg/dsl/decompiler.go](../../../src/semantic-router/pkg/dsl/decompiler.go)
- [src/semantic-router/pkg/dsl/emitter_yaml.go](../../../src/semantic-router/pkg/dsl/emitter_yaml.go)
- [dashboard/frontend/src/types/config.ts](../../../dashboard/frontend/src/types/config.ts)
- [dashboard/frontend/src/pages/configPageCanonicalization.ts](../../../dashboard/frontend/src/pages/configPageCanonicalization.ts)
- [dashboard/frontend/src/pages/configPageRouterDefaultsSupport.ts](../../../dashboard/frontend/src/pages/configPageRouterDefaultsSupport.ts)
- [dashboard/frontend/src/pages/setupWizardSupport.ts](../../../dashboard/frontend/src/pages/setupWizardSupport.ts)

## Why It Matters

- v0.3 depends on config, DSL, dashboard, and operator paths agreeing on one
  contract shape.
- Raw maps make it hard to distinguish supported config from arbitrary payload
  passthrough.
- Dashboard editor state becomes harder to validate when canonicalization,
  fallback rendering, and setup scaffolding share mutable record utilities.

## Desired End State

- DSL compile/decompile/emit paths use named AST or payload helpers before
  crossing into YAML, CRD, or structured payload encoding.
- Dashboard editor state exposes one canonical steady-state config model.
- Import/canonicalization code owns any non-canonical input handling before data
  reaches normal editor and deploy flows.

## Exit Criteria

- Remaining DSL YAML helpers no longer use open-ended map/value lowering for
  supported router fields.
- Dashboard config utilities no longer need long-lived hybrid editor types for
  normal editing, preview, or deploy.
- Setup wizard scaffolding and router-default helpers operate on typed payloads
  or narrow helpers instead of mutable record bags.
