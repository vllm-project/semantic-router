# TD001: Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard

## Status

Closed

## Scope

configuration architecture

## Summary

The repo-wide rollout to the canonical v0.3 contract is now closed for the active steady-state workflow. The router parser accepts only canonical `version/listeners/providers/routing/global`, maintained deploy/E2E/example assets parse as canonical config under contract tests, dashboard config management rewrites legacy input to canonical before editing or saving, and the operator/Helm/DSL paths all emit or consume the same public structure. The in-process `IntelligentPool` / `IntelligentRoute` controller path now re-enters the same canonical parser through `global.router.config_source: kubernetes`.

## Evidence

- [src/semantic-router/pkg/config/canonical_config.go](../../../src/semantic-router/pkg/config/canonical_config.go)
- [src/semantic-router/pkg/config/canonical_defaults.go](../../../src/semantic-router/pkg/config/canonical_defaults.go)
- [src/semantic-router/pkg/config/canonical_export.go](../../../src/semantic-router/pkg/config/canonical_export.go)
- [src/semantic-router/pkg/config/config.go](../../../src/semantic-router/pkg/config/config.go)
- [src/semantic-router/pkg/config/loader.go](../../../src/semantic-router/pkg/config/loader.go)
- [src/semantic-router/pkg/dsl/emitter_yaml.go](../../../src/semantic-router/pkg/dsl/emitter_yaml.go)
- [src/semantic-router/pkg/dsl/routing_contract.go](../../../src/semantic-router/pkg/dsl/routing_contract.go)
- [dashboard/backend/handlers/setup.go](../../../dashboard/backend/handlers/setup.go)
- [dashboard/frontend/src/pages/ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
- [dashboard/frontend/src/pages/setupWizardSupport.ts](../../../dashboard/frontend/src/pages/setupWizardSupport.ts)
- [dashboard/frontend/src/lib/dslLanguage.ts](../../../dashboard/frontend/src/lib/dslLanguage.ts)
- [dashboard/frontend/src/lib/dslMutations.ts](../../../dashboard/frontend/src/lib/dslMutations.ts)
- [dashboard/frontend/src/pages/DslEditorPage.tsx](../../../dashboard/frontend/src/pages/DslEditorPage.tsx)
- [dashboard/frontend/src/pages/builderPageImportModal.tsx](../../../dashboard/frontend/src/pages/builderPageImportModal.tsx)
- [dashboard/frontend/src/pages/ConfigPageRouterConfigSection.tsx](../../../dashboard/frontend/src/pages/ConfigPageRouterConfigSection.tsx)
- [dashboard/frontend/src/pages/ConfigPage.tsx](../../../dashboard/frontend/src/pages/ConfigPage.tsx)
- [dashboard/frontend/src/pages/configPageCanonicalization.ts](../../../dashboard/frontend/src/pages/configPageCanonicalization.ts)
- [dashboard/backend/handlers/config.go](../../../dashboard/backend/handlers/config.go)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [config/config.yaml](../../../config/config.yaml)
- [config/README.md](../../../config/README.md)
- [deploy/examples/runtime/README.md](../../../deploy/examples/runtime/README.md)
- [e2e/config/README.md](../../../e2e/config/README.md)
- [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types.go)
- [deploy/operator/controllers/semanticrouter_controller.go](../../../deploy/operator/controllers/semanticrouter_controller.go)
- [website/docs/installation/k8s/operator.md](../../../website/docs/installation/k8s/operator.md)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/vllm-sr/cli/parser.py](../../../src/vllm-sr/cli/parser.py)
- [website/docs/installation/configuration.md](../../../website/docs/installation/configuration.md)
- [website/docs/proposals/unified-config-contract-v0-3.md](../../../website/docs/proposals/unified-config-contract-v0-3.md)
- [src/semantic-router/pkg/config/maintained_asset_contract_test.go](../../../src/semantic-router/pkg/config/maintained_asset_contract_test.go)
- [bench/README.md](../../../bench/README.md)
- [bench/hallucination/config.yaml](../../../bench/hallucination/config.yaml)
- [bench/hallucination/config-7b.yaml](../../../bench/hallucination/config-7b.yaml)
- [bench/cpu-vs-gpu/config-bench.yaml](../../../bench/cpu-vs-gpu/config-bench.yaml)
- [bench/cpu-vs-gpu/config-bench-candle.yaml](../../../bench/cpu-vs-gpu/config-bench-candle.yaml)
- [e2e/config/config.e2e.yaml](../../../e2e/config/config.e2e.yaml)
- [e2e/config/config.response-api.yaml](../../../e2e/config/config.response-api.yaml)
- [e2e/config/config.multi-provider.yaml](../../../e2e/config/config.multi-provider.yaml)
- [e2e/profiles/llm-d/values.yaml](../../../e2e/profiles/llm-d/values.yaml)
- [deploy/kserve/configmap-router-config.yaml](../../../deploy/kserve/configmap-router-config.yaml)
- [deploy/operator/config/crd/bases/vllm.ai_semanticrouters.yaml](../../../deploy/operator/config/crd/bases/vllm.ai_semanticrouters.yaml)
- [website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/installation/installation.md](../../../website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/installation/installation.md)
- [website/versioned_docs/version-v0.1/installation/installation.md](../../../website/versioned_docs/version-v0.1/installation/installation.md)
- [docs/agent/plans/pl-0003-v0-3-config-contract-rollout.md](../plans/pl-0003-v0-3-config-contract-rollout.md)

## Why It Matters

- The original issue was schema drift between router, CLI, dashboard, and Kubernetes-facing workflows.
- That drift also made repo-owned config assets hard to keep in sync, because the example tree did not encode which fragments were required to cover the active routing surface.
- As long as maintained deploy/E2E assets and helper code still encode a second legacy user-config shape, contributors can accidentally reintroduce it as if it were an equal steady-state contract.

## Desired End State

- One canonical config contract with Go as schema owner and thin adapters for CLI, dashboard, DSL, and Kubernetes deployment.
- The DSL owns only routing semantics, while provider deployment bindings and global runtime overrides stay outside the DSL.
- Provider deployment bindings are expressed consistently as `providers.models[]` with direct `backend_refs` and related access fields.
- Router-wide overrides are expressed consistently as `global.router/services/stores/integrations/model_catalog`, with model-backed module settings under `global.model_catalog.modules`.
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
- Dashboard config pages, onboarding helpers, and topology/editor types no longer assume pre-`providers.defaults` or pre-`providers.models[]` shapes.
- Current docs, translated docs, and maintained versioned docs all describe the same canonical `version/listeners/providers/routing/global` contract and the current `global.router/services/stores/integrations/model_catalog` hierarchy, including `global.model_catalog.modules`, for the active workflow.

## Retirement Notes

- The steady-state router parser now rejects both deprecated canonical-user fields and top-level legacy runtime layouts; migration remains explicit via `vllm-sr config migrate`.
- Repo-owned maintained config assets are enforced by `go test ./pkg/config/...` to stay on canonical v0.3, including embedded `config.yaml` payloads in KServe manifests, `values.yaml`-embedded config blocks in deploy/E2E profiles, and benchmark-owned config templates/fragments under `bench/`.
- Dashboard import/edit/save flows canonicalize legacy config before writing it back, so config management no longer re-emits `model_config`, `vllm_endpoints`, or `provider_profiles` as steady-state output.
- Operator, Helm, DSL, CLI, config fragments, and latest docs now describe or emit the same canonical `providers/routing/global` shape.
- If a future change reintroduces a second steady-state user config contract or allows maintained repo assets to drift back to legacy runtime layout, reopen this item.
