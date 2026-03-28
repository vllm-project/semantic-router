# Taxonomy Classifier Platform Loop Execution Plan

## Goal

- Complete the taxonomy-classifier platform refactor as one resumable workstream across router config/runtime, built-in assets, router API surface, dashboard backend/frontend, playground reveal, and image packaging.
- Remove routing-authored DSL classifier declarations so taxonomy classifiers live only in canonical `global.model_catalog.classifiers[]`.
- Ship one router-built-in taxonomy classifier package with router-owned defaults while keeping custom classifier CRUD and UI management available.

## Scope

- `docs/agent/plans/**`
- `src/semantic-router/pkg/config/**`
- `src/semantic-router/pkg/classification/**`
- `src/semantic-router/pkg/decision/**`
- `src/semantic-router/pkg/dsl/**`
- `src/semantic-router/pkg/apiserver/**`
- `src/semantic-router/pkg/extproc/**`
- `src/semantic-router/pkg/services/**`
- `src/vllm-sr/**`
- `dashboard/backend/**`
- `dashboard/frontend/src/**`
- `config/**`
- `deploy/recipes/privacy/**`
- `tools/docker/**`
- targeted tests and repo-native validation for the changed surfaces
- nearest local rules for `pkg/config`, `pkg/classification`, `pkg/extproc`, `vllm-sr/cli`, `dashboard/frontend/src`, `dashboard/frontend/src/pages`, `dashboard/frontend/src/components`, and `dashboard/backend/handlers`

## Exit Criteria

- DSL no longer supports top-level `CLASSIFIER` declarations; taxonomy signals still compile/decompile against classifier names declared in global config.
- Router defaults ship one built-in taxonomy classifier package from a stable built-in asset path, and runtime images copy the package for `vllm-sr`, `vllm-sr-rocm`, `extproc`, and `extproc-rocm`.
- Router apiserver exposes classifier list/read/add/update/delete APIs with asset-directory management for custom classifiers and runtime config refresh.
- Dashboard backend proxies or owns matching classifier-management endpoints without breaking existing manager/global flows.
- Dashboard `System -> Global` UI exposes classifier inventory and custom-classifier management in the existing manager style, including tier/category inspection and taxonomy-signal compatibility context.
- Playground reveal surfaces matched taxonomy signals when present.
- Applicable repo-native validation is rerun until the active changed-set gates pass or a durable blocker is recorded.

## Task List

- [x] `TAX001` Create the indexed execution plan and lock the workstream scope around taxonomy classifier platform changes.
- [x] `TAX002` Remove DSL classifier declarations and realign compiler/decompiler/AST/tests around global-owned taxonomy classifiers only.
- [x] `TAX003` Move the privacy taxonomy assets into a router-built-in classifier package, wire the built-in classifier into canonical global defaults, and update maintained recipes/docs/tests.
- [x] `TAX004` Ensure built-in classifier assets are copied into `vllm-sr`, `vllm-sr-rocm`, `extproc`, and `extproc-rocm` images and any repo-native packaging/docs remain aligned.
- [x] `TAX005` Add router apiserver classifier CRUD APIs plus runtime/config persistence and targeted tests.
- [x] `TAX006` Add dashboard backend classifier-management routes/services and align them with the router API surface.
- [x] `TAX007` Add dashboard `System -> Global` classifier management UI for built-in/custom classifiers with tier/category visibility and taxonomy-signal reference context.
- [x] `TAX008` Surface matched taxonomy signals in the playground reveal/UI.
- [x] `TAX009` Run the applicable validation ladder, update this plan with results, and record any durable remaining gap as indexed debt if needed.

## Current Loop

- Completed on 2026-03-25 on branch `vsr/pr-1644-analysis`.
- Loop closure summary:
  - `L1` execution plan, repo-native instructions, and surface inventory were captured.
  - `L2` taxonomy classifiers were moved to global ownership and DSL classifier declarations were removed.
  - `L3` the built-in privacy classifier package was moved to `config/classifiers/privacy/`, wired into defaults, and shipped in runtime image build paths.
  - `L4` router apiserver classifier CRUD plus managed asset persistence landed with targeted tests.
  - `L5` dashboard backend/frontend classifier management and playground matched-taxonomy reveal landed.
  - `L6` validation evidence was refreshed and no new durable debt entry was required for this loop.
- Validation evidence:
  - repo-native commit hooks passed during the final signed commits, including `go fmt`, `go lint`, `md fmt`, `yaml/yml fmt`, `js/ts lint`, `black`, and the AST supply-chain scan
  - `go test ./pkg/config -count=1`
  - `go test ./pkg/dsl -count=1`
  - `go test ./handlers -run TestRouterClassifierProxyHandler -count=1`
  - `npm run type-check`
- Implementation checkpoints closed:
  - config/runtime: taxonomy classifier globals, taxonomy signals, and projection metrics landed under `src/semantic-router/pkg/{config,classification,decision,dsl}/`
  - built-in assets and packaging: `config/classifiers/privacy/`, `src/vllm-sr/Dockerfile{,.rocm}`, and `tools/docker/Dockerfile.extproc{,-rocm}` ship the built-in classifier package
  - router APIs: `src/semantic-router/pkg/apiserver/route_taxonomy_classifiers.go` and `taxonomy_classifier_store.go`
  - dashboard surfaces: `dashboard/backend/handlers/config_classifier_proxy.go` and `dashboard/frontend/src/pages/ConfigPageTaxonomyClassifiers.tsx`
  - playground reveal: `dashboard/frontend/src/components/{HeaderDisplay.tsx,HeaderReveal.tsx,chatRequestSupport.ts}` plus matched-taxonomy header emission in `src/semantic-router/pkg/{headers,extproc}/`

## Decision Log

- 2026-03-25: This is a long-horizon cross-surface change, so the canonical loop record lives in an execution plan rather than ad hoc chat state.
- 2026-03-25: Taxonomy classifiers remain global-owned runtime assets; routing DSL must reference classifier names without becoming a second source of truth for classifier definitions.
- 2026-03-25: The built-in privacy taxonomy package should move out of the maintained recipe directory and into a router-owned asset path that images can ship by default.
- 2026-03-25: Built-in and custom classifiers should both be visible, but the dashboard management requirement is centered on custom classifiers; built-ins may remain read-only if that keeps ownership clearer.
- 2026-03-25: The loop closed without opening a new indexed debt entry; existing subsystem debt items remain the right place for broader hotspot follow-up beyond this feature work.

## Follow-up Debt / ADR Links

- Reuse existing debt first if this loop leaves architectural seams only partially reduced:
  - [TD020 Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators](../tech-debt/td-020-classification-subsystem-boundary-collapse.md)
  - [TD023 Extproc Request Pipeline Phases Have Collapsed Across Request Filters](../tech-debt/td-023-extproc-request-pipeline-phase-collapse.md)
  - [TD025 Dashboard Backend Runtime-Control Slice Still Collapses Handler Transport, Config Persistence, and Status Collection](../tech-debt/td-025-dashboard-backend-runtime-control-slice-collapse.md)
  - [TD030 Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers](../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
  - [TD034 Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract](../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
