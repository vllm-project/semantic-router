# TD001: Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard

## Status

Closed

## Scope

configuration architecture

## Summary

The v0.3 canonical-config rollout is complete for the public config contract. Steady-state runtime config is now canonical-only, router-owned defaults are the single default source, and repo-owned config assets are enforced against the supported routing surface catalog.

## Evidence

- [src/semantic-router/pkg/config/canonical_config.go](../../../src/semantic-router/pkg/config/canonical_config.go)
- [src/semantic-router/pkg/config/canonical_defaults.go](../../../src/semantic-router/pkg/config/canonical_defaults.go)
- [src/semantic-router/pkg/config/canonical_export.go](../../../src/semantic-router/pkg/config/canonical_export.go)
- [src/semantic-router/pkg/config/config.go](../../../src/semantic-router/pkg/config/config.go)
- [src/semantic-router/pkg/dsl/emitter_yaml.go](../../../src/semantic-router/pkg/dsl/emitter_yaml.go)
- [src/semantic-router/pkg/dsl/routing_contract.go](../../../src/semantic-router/pkg/dsl/routing_contract.go)
- [dashboard/backend/handlers/setup.go](../../../dashboard/backend/handlers/setup.go)
- [dashboard/frontend/src/lib/dslLanguage.ts](../../../dashboard/frontend/src/lib/dslLanguage.ts)
- [dashboard/frontend/src/lib/dslMutations.ts](../../../dashboard/frontend/src/lib/dslMutations.ts)
- [dashboard/frontend/src/pages/DslEditorPage.tsx](../../../dashboard/frontend/src/pages/DslEditorPage.tsx)
- [dashboard/frontend/src/pages/builderPageImportModal.tsx](../../../dashboard/frontend/src/pages/builderPageImportModal.tsx)
- [dashboard/frontend/src/pages/ConfigPageRouterConfigSection.tsx](../../../dashboard/frontend/src/pages/ConfigPageRouterConfigSection.tsx)
- [dashboard/frontend/src/pages/ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [config/config.yaml](../../../config/config.yaml)
- [config/README.md](../../../config/README.md)
- [examples/runtime/README.md](../../../examples/runtime/README.md)
- [e2e/config/README.md](../../../e2e/config/README.md)
- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [deploy/operator/controllers/semanticrouter_controller.go](../../../deploy/operator/controllers/semanticrouter_controller.go)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/vllm-sr/cli/parser.py](../../../src/vllm-sr/cli/parser.py)
- [website/docs/installation/configuration.md](../../../website/docs/installation/configuration.md)
- [website/docs/proposals/unified-config-contract-v0-3.md](../../../website/docs/proposals/unified-config-contract-v0-3.md)
- [website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/installation/installation.md](../../../website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/installation/installation.md)
- [website/versioned_docs/version-v0.1/installation/installation.md](../../../website/versioned_docs/version-v0.1/installation/installation.md)
- [docs/agent/plans/pl-0003-v0-3-config-contract-rollout.md](../plans/pl-0003-v0-3-config-contract-rollout.md)

## Why It Matters

- The original issue was schema drift between router, CLI, dashboard, and Kubernetes-facing workflows.
- That drift also made repo-owned config assets hard to keep in sync, because the example tree did not encode which fragments were required to cover the active routing surface.
- As long as the router kept accepting a second legacy user-config shape directly, contributors could accidentally treat migration compatibility as an equal steady-state contract.

## Desired End State

- One canonical config contract with Go as schema owner and thin adapters for CLI, dashboard, DSL, and Kubernetes deployment.
- The DSL owns only routing semantics, while provider deployment bindings and global runtime overrides stay outside the DSL.
- No user-visible docs or dashboard entrypoints imply that `.vllm-sr/router-defaults.yaml` or a second router defaults file is a normal file users must edit.
- Dashboard config-management views understand the canonical split between `routing.modelCards` and `providers.models` instead of reconstructing earlier mixed layouts.
- Repo-owned config assets are organized around canonical `config/config.yaml` plus `signal/decision/algorithm/plugin` fragments, while runtime examples and harness manifests live outside `config/`.
- CLI typed schema models only canonical v0.3 fields, and migration-only compatibility stays isolated to explicit `config migrate` codepaths.
- The router parser itself accepts only canonical v0.3 config for steady-state runtime loading; migration remains explicit.

## Exit Criteria

- Router, CLI, dashboard, operator, Helm, and DSL use the same canonical top-level config layout for the common path.
- Adding a config feature no longer requires parallel structural edits across independent schemas for local, dashboard, and Kubernetes workflows.
- Legacy steady-state config generation paths such as `router-config.yaml`, `router-defaults.yaml`, and nested provider endpoint/auth model layouts are retired or reduced to explicit migration tooling only.
- Migration-only provider-model compatibility fields are no longer part of the normal CLI typed config path.
- Router-side migration helpers do not silently expand back into a second user-facing config contract.
- Dashboard runtime-defaults surfaces are either backed by router-owned defaults APIs or relabeled so they no longer imply a required local defaults file.
- Current docs, translated docs, and maintained versioned docs all describe the same canonical `version/listeners/providers/routing/global` contract for the active workflow.

All exit criteria are satisfied as of 2026-03-14. Remaining hotspot size and structure-rule gaps are tracked separately in [TD006](td-006-structural-rule-target-vs-legacy-hotspots.md), not as a second config contract.
