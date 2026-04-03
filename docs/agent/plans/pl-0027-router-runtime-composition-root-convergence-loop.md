# Router Runtime Composition Root Convergence Loop

## Goal

- Retire the remaining process-wide runtime bootstrap and shared-service global-state seams that remain after `pl-0026` completed model runtime lifecycle orchestration.
- Move startup, reload, and steady-state runtime dependency publication toward an explicit composition root or narrow runtime registry instead of package-global setters and singleton watcher assumptions.
- Close this workstream only after runtime-owned services, reload observers, and API-server dependencies agree on one typed dependency contract.

## Scope

- `docs/agent/plans/**`
- `src/semantic-router/cmd/{main.go,runtime_bootstrap.go}`
- `src/semantic-router/pkg/apiserver/**`
- `src/semantic-router/pkg/config/{loader.go,*.go}`
- `src/semantic-router/pkg/extproc/**`
- `src/semantic-router/pkg/routerruntime/**`
- `src/semantic-router/pkg/services/**`
- `src/semantic-router/pkg/memory/**`
- `src/semantic-router/pkg/modelruntime/**`
- targeted tests for runtime bootstrap, reload fanout, and service-registry seams
- nearest local rules for `pkg/config` and `pkg/extproc`

## Exit Criteria

- Startup, reload, and shutdown each operate through explicit runtime-owned dependency seams instead of depending on package-global publication order.
- `apiserver` receives runtime services through constructors or one narrow runtime registry object rather than package-global setter state for steady-state operation.
- Config update fanout supports multiple consumers without relying on one implicit global watcher channel.
- Classification, memory, and vector-store service publication are no longer hidden behind process-global singleton registration for the common runtime path.
- Targeted repo-native validation covers the composition-root and service-registry changes, or any remaining gap is narrowed and re-recorded against the debt inventory instead of being left implicit.

## Task List

- [x] `RRC001` Create and index a successor execution plan for the post-`pl-0026` runtime composition-root and service-registry convergence work.
- [x] `RRC002` Inventory the remaining process-global runtime owners and split them into composition-root, runtime-registry, and legacy-compatibility buckets.
- [x] `RRC003` Introduce a typed router-runtime registry or composition-root seam that can own API-server, classification, memory, and vector-store dependencies without package-global setters.
- [x] `RRC004` Migrate `apiserver` steady-state dependency access away from package-global setters onto the explicit runtime seam.
- [x] `RRC005` Replace the single-consumer config watcher assumption with an explicit fanout or subscription contract that supports runtime reload consumers cleanly.
- [x] `RRC006` Move classification and adjacent runtime service publication off process-global registration in the common runtime path.
- [x] `RRC007` Split startup, reload, and shutdown orchestration into narrower lifecycle owners that can be tested independently for degraded and partial availability cases.
- [x] `RRC008` Run the validation ladder for the affected surfaces and update linked debt entries only for gaps that remain after the current loop.

## Current Loop

- Loop status: implementation loop completed on 2026-04-03; the planned runtime-registry and fanout seams for this workstream are now closed.
- Completed in this loop:
  - verified that `pl-0026` exit criteria and task list are now fully complete
  - confirmed the remaining optimization space belongs to process-wide runtime globals and composition-root ownership, not to the completed model-lifecycle workstream
  - scoped the follow-up work around the already-open debt tracked in `TD031` and the backend-lifecycle follow-on tracked in `TD033`
  - inventoried the remaining runtime globals into three buckets: composition-root owners in `cmd`, steady-state runtime registry consumers in `apiserver` and `extproc`, and legacy compatibility globals left as fallbacks only
  - introduced `pkg/routerruntime` as the typed registry/composition-root seam for runtime config, classification service, memory store, and vector-store runtime ownership
  - moved vector-store startup and shutdown ownership behind `routerruntime.VectorStoreRuntime`, so startup registers one shared runtime object instead of publishing file-store/embedder/pipeline globals directly from `cmd`
  - migrated API-server steady-state dependency access onto the runtime registry for classification, memory, vector-store manager, embedder, pipeline, and file-store access, while keeping package globals as compatibility fallback only
  - replaced the single-consumer config watcher assumption with subscription fanout in `config/loader.go`, and updated extproc Kubernetes reload watching to use explicit subscriptions
  - removed implicit common-path service publication by making classification globals explicit compatibility setters instead of constructor side effects and by stopping memory/vector-store startup from publishing directly through package globals
  - tightened reload publication so a rebuilt router carries the runtime registry and only publishes new steady-state services after warmup and swap
  - reran validation for the current loop:
    - `go test ./pkg/routerruntime ./pkg/config ./pkg/services -run 'TestRegistryPublishRouterRuntime|TestSubscribeConfigUpdatesFanout|TestClassificationService_BasicFunctionality'`
    - `go test ./pkg/apiserver -run 'TestRuntimeRegistryResolvesSharedDependencies|TestHandleOpenAIModelsUsesResolvedRuntimeConfig'`
    - `go test ./pkg/extproc -run 'TestReloadRouterFrom|TestReloadRouterFromConfigPublishesRuntimeRegistryAfterSwap'`
    - `go test ./pkg/classification -run 'TestNewClassifierWithOptionsDefersRuntimeInitialization'`
    - `go test ./pkg/routerruntime ./pkg/modelruntime ./pkg/config ./pkg/services ./pkg/apiserver ./pkg/extproc ./pkg/classification ./cmd/... -run '^$'`
    - `make agent-validate`
    - `make agent-ci-gate CHANGED_FILES=\"...\"`
    - `make test-semantic-router`
- Next loop focus:
  - none; reopen only if a new runtime-owned dependency still bypasses `routerruntime.Registry` in the common startup or reload path

## Decision Log

- 2026-04-03: keep `pl-0026` closed and create a successor plan instead of reusing it for composition-root work, because model-lifecycle orchestration and process-global runtime-state convergence are adjacent but distinct workstreams.
- 2026-04-03: treat `TD031` as the primary debt driver for this loop; `TD033` remains related but should only expand scope when backend capability or reset semantics become the active blocker.
- 2026-04-03: prefer a narrow runtime-registry or composition-root seam over additional package-level setter indirection.
- 2026-04-03: split the new runtime registry into `pkg/routerruntime` instead of `pkg/modelruntime`, because `classification` already depends on `pkg/modelruntime` for lifecycle execution and the registry must not create an import cycle.

## Follow-up Debt / ADR Links

- Predecessor workstream: [pl-0026-model-runtime-lifecycle-orchestration-loop.md](pl-0026-model-runtime-lifecycle-orchestration-loop.md)
- [adr-0005-model-runtime-lifecycle-orchestration.md](../adr/adr-0005-model-runtime-lifecycle-orchestration.md)
- [TD031 Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [TD033 Native Binding Runtime Parity and Lifecycle Still Diverge Across Candle and ONNX Backends](../tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md)
