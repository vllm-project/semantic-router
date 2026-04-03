# Model Runtime Lifecycle Orchestration Execution Plan

## Goal

- Replace implicit startup and constructor-side model lifecycle behavior with an explicit runtime lifecycle contract.
- Make model initialization and preload-heavy classifier construction concurrent where safe, while preserving native-binding order constraints that must remain serialized.
- Close this workstream only after startup, classifier assembly, warmup, docs, and targeted validation all agree on the same lifecycle seams.

## Scope

- `docs/agent/adr/**`
- `docs/agent/plans/**`
- `src/semantic-router/cmd/**`
- `src/semantic-router/pkg/modelruntime/**`
- `src/semantic-router/pkg/classification/**`
- `src/semantic-router/pkg/extproc/**`
- targeted tests for the runtime executor and lifecycle seams
- nearest local rules for `pkg/classification`, `pkg/config`, and `pkg/extproc`

## Exit Criteria

- The repository has a dedicated runtime-orchestration seam for bounded-parallel lifecycle execution instead of relying only on startup call order.
- Classifier construction and classifier runtime initialization are separate operations, and router assembly chooses the explicit init point.
- Embedding, complexity, and knowledge-base preload-heavy construction no longer serialize by default when they can be built independently.
- Startup embedding-family initialization is planned explicitly and runs concurrently where native-binding constraints allow it.
- Warmup behavior such as tools loading is owned by an explicit lifecycle step, not only by ad hoc post-constructor checks.
- Targeted repo-native validation passes for the changed surfaces, or any remaining architecture gap is promoted into the indexed debt inventory.

## Task List

- [x] `MRL001` Create and index the durable ADR and execution plan for model runtime lifecycle orchestration.
- [x] `MRL002` Introduce a dedicated runtime task executor with dependency-aware bounded parallelism and best-effort versus required task semantics.
- [x] `MRL003` Separate classifier construction from classifier runtime initialization while preserving a compatibility wrapper for existing callers.
- [x] `MRL004` Parallelize preload-heavy classifier component construction, especially embedding, complexity, and knowledge-base setup.
- [x] `MRL005` Replace sequential embedding-family startup init with an explicit runtime lifecycle plan and controlled parallel execution.
- [x] `MRL006` Move router warmup ownership onto explicit lifecycle steps and tighten the handoff between startup bootstrap and extproc assembly.
- [x] `MRL007` Extend lifecycle usage to reload and remaining startup seams that still depend on implicit init order.
- [x] `MRL008` Run the validation ladder for the affected surfaces, record the results here, and update indexed debt entries only for gaps that remain after the current loop.

## Current Loop

- Loop status: second implementation loop completed on 2026-04-03; the planned lifecycle seams for this workstream are now closed.
- Completed in this loop:
  - audited the current download, startup, extproc, classifier, and preload ownership paths
  - identified the main lifecycle breakpoints: asset preflight, runtime init, and warmup
  - confirmed the native-binding constraint that the embedding factory path is not freely parallelizable because of `OnceLock` ownership
  - locked the durable architecture direction in `ADR 0005` and accepted it after landing the first implementation pass
  - created and indexed this execution plan for resumable follow-up work
  - added `pkg/modelruntime` as a dedicated bounded-parallel lifecycle executor with unit coverage
  - split classifier build from classifier runtime initialization through `BuildClassifier` and `InitializeRuntime`
  - parallelized preload-heavy classifier component construction so embedding, complexity, knowledge-base, and contrastive-jailbreak setup no longer serialize by default
  - moved startup embedding-family initialization onto explicit lifecycle tasks and tightened tools database warmup to require the right embedding backend
  - switched the extproc router assembly path onto explicit classifier runtime initialization instead of constructor side effects
  - extracted startup-only runtime preparation into `pkg/modelruntime/router_runtime.go` so startup and reload now share the same explicit runtime-prep and warmup contract
  - folded semantic-cache BERT, vector-store BERT, modality classifier init, and tools database warmup into the same dependency-aware lifecycle seam used by startup
  - extended config reload so it now runs runtime preparation before building the candidate router and best-effort warmup before swapping the live router
  - removed the last classifier constructor-era implicit init edge by keeping `newClassifierWithOptions` build-only and routing all runtime init through `InitializeRuntime`
  - reran validation for the current loop:
    - `go test ./pkg/modelruntime`
    - `go test ./pkg/classification -run 'TestNewClassifierWithOptionsDefersRuntimeInitialization'`
    - `go test ./pkg/extproc -run 'TestReloadRouterFrom'`
    - `go test ./cmd/... -run '^$'`
    - `make agent-validate`
    - `make test-semantic-router`
- Next loop focus:
  - none; reopen only if a new lifecycle seam appears outside the current startup, extproc, classifier, or reload boundaries

## Decision Log

- 2026-04-03: use one execution plan because the work spans startup bootstrap, classifier construction, extproc assembly, native-binding constraints, and validation.
- 2026-04-03: prefer `planner + bounded-parallel orchestrator + explicit build/init split` over a pure factory-only refactor.
- 2026-04-03: keep order-sensitive native-binding paths explicit rather than pretending all model families are equally parallel-safe.
- 2026-04-03: treat preload-heavy classifier construction as part of lifecycle execution, not as an invisible constructor side effect.
- 2026-04-03: keep `NewClassifier` as a compatibility wrapper during migration, but move the router assembly path onto explicit initialization.
- 2026-04-03: close reload and remaining startup gaps by extracting a shared router-runtime seam instead of duplicating bootstrap logic in `cmd` and `pkg/extproc`.
- 2026-04-03: do not create a new debt entry for the remaining lifecycle/global-state gaps because TD031 and TD033 already track the unresolved composition-root and native-binding ownership issues this loop leaves behind.

## Follow-up Debt / ADR Links

- Successor workstream: [pl-0027-router-runtime-composition-root-convergence-loop.md](pl-0027-router-runtime-composition-root-convergence-loop.md)
- [adr-0005-model-runtime-lifecycle-orchestration.md](../adr/adr-0005-model-runtime-lifecycle-orchestration.md)
- [TD031 Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [TD033 Native Binding Runtime Parity and Lifecycle Still Diverge Across Candle and ONNX Backends](../tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md)
