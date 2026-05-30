# Architecture Scorecard Ratchet Execution Plan

## Goal

- Turn the 2026-05-24 whole-repository architecture review into a durable scorecard and staged improvement loop.
- Use a 100-point target to prioritize architecture work without creating a broad rewrite.
- Keep future score movement tied to source evidence, validation output, and indexed debt or plan entries.

## Scope

- `docs/agent/architecture-scorecard.md`
- `docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md`
- `docs/agent/plans/README.md`
- `docs/agent/README.md`
- `tools/agent/repo-manifest.yaml`
- Existing debt and execution-plan entries referenced from the scorecard

## Exit Criteria

- The scorecard is indexed from the canonical agent docs and repo manifest.
- This execution plan is indexed and can be resumed from the repository alone.
- PR0 records the current score baseline, evidence snapshot, scoring rule, PR chain, and validation commands.
- Each follow-up PR updates the scorecard only when its evidence and validation justify score movement.
- The open debt entries referenced by the PR chain are closed or narrowed as the corresponding implementation work lands.

## Task List

- [x] `ASR001` Create the architecture scorecard with the 100-point scoring rule, baseline scores, evidence snapshot, and PR-chain order.
- [x] `ASR002` Create and index this execution plan for the long-horizon scorecard ratchet.
- [x] `ASR003` Add scorecard and plan entries to the canonical docs index and repo manifest governance inventory.
- [x] `ASR004` Run the PR0 harness validation and record the result in this plan.
- [ ] `ASR005` Execute the Contract/API ratchet and update scores for TD015, TD036, TD039, and TD040; keep TD036, TD038, and TD040 as resolved-contract regression guards after closure.
- [ ] `ASR006` Execute the Runtime/State ratchet and update scores for TD031 and TD034.
- [ ] `ASR007` Execute the Router Pipeline ratchet and update scores for TD020; keep TD023 and TD029 as historical closed guards.
- [ ] `ASR008` Execute the Control-Plane implementation ratchet and update scores for TD027, TD028, TD030, and TD037; keep TD004 as a closed stale-debt guard.
- [ ] `ASR009` Execute the Release/Docs/Native ratchet and update scores for TD033 plus release and documentation freshness gaps.
- [x] `ASR010` Reconcile the scorecard once all dimensions reach at least 80, then define the next 90+ convergence loop.

## Current Loop

- Opened on 2026-05-24 from a maintainer-style whole-repository audit covering router runtime, API contracts, Dashboard, Operator, CLI, Fleet Sim, native bindings, E2E, release workflows, docs, and the agent harness.
- PR0 is intentionally documentation-only. It creates the scorecard, creates this execution plan, updates canonical indexes, and does not modify user-facing APIs or runtime behavior.
- Initial harness routing classified the change as `harness-contract-change`, documentation-only, lightweight loop mode, with `make agent-validate` as the minimum required gate.
- Completed in PR0:
  - `make agent-validate` passed.
  - `make agent-scorecard` passed with validation status `pass` and 0 validation errors.
  - `make agent-scorecard` reported 105 indexed harness docs, 104 governed docs, 19 open technical debt items, and 59 open execution-plan tasks after this plan was indexed and `ASR004` was closed.
  - `make agent-ci-gate CHANGED_FILES="docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/plans/README.md,docs/agent/README.md,tools/agent/repo-manifest.yaml"` passed.
- Current status: PR0 governance artifacts are complete. Follow-up implementation work starts at `ASR005`.
- ASR005 progress on 2026-05-24:
  - Current-source recheck found TD038 already resolved; no open Contract/API work remains there.
  - TD040 is resolved by replacing the one-off K8s embedding validator mirror with `config.ValidateKubernetesConfigContracts`, backed by the shared config validator dispatch table.
  - Added config-package dispatch coverage and reconciler-level K8s validation canaries for embedding modality and global advanced tool filtering.
  - Validation passed: `go test ./pkg/config ./pkg/k8s`; `make test-semantic-router`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/plans/README.md,docs/agent/README.md,docs/agent/tech-debt/td-040-reconcile-path-skips-family-validators.md,tools/agent/repo-manifest.yaml,src/semantic-router/pkg/config/validator.go,src/semantic-router/pkg/config/validator_embedding.go,src/semantic-router/pkg/config/validator_test.go,src/semantic-router/pkg/k8s/reconciler.go,src/semantic-router/pkg/k8s/reconciler_embedding_modality_test.go"`.
  - Local smoke was blocked by environment, not by test failure: `docker info` could not connect to `/Users/bitliu/.docker/run/docker.sock`.
  - Current-source recheck found TD036 partially stale: `spec/dsl.md` no longer exists, and `pl-0012` already records the decision to keep `DECISION_TREE` as authoring sugar instead of adding tree metadata to canonical config.
  - TD036 is resolved by codifying that narrowed contract: `DecompileRouting()` emits flat `ROUTE` blocks for tree-authored input, maintained DSL/YAML recipe pairs document the authoring/runtime split, and the retention tutorial now states that retention fields round-trip while tree shape does not.
  - Validation passed: `go test ./pkg/dsl`; `make test-semantic-router`.
  - Local smoke remains blocked by environment, not by test failure: `docker info` could not connect to `/Users/bitliu/.docker/run/docker.sock`.
  - TD015 remains open, but the compiler/validator field-bag slice is narrowed: `src/semantic-router/pkg/dsl/field_payload.go` now owns the named DSL payload conversion seam used by structure signals, conversation signals, and RAG `backend_config` payloads.
  - Validation passed: `go test ./pkg/dsl`; `make test-semantic-router`.
  - A second TD015 DSL slice is narrowed: `src/semantic-router/pkg/dsl/decompiler.go` now formats structure/conversation feature objects and tools `dynamic_retrieval` through AST `Value` / `ObjectValue` helpers instead of raw map helpers on typed decompile paths.
  - Removed obsolete decompiler raw-map helpers for dynamic retrieval, structure feature/source, conversation feature/source, and numeric predicates; raw map normalization remains only for unknown plugin payload compatibility.
  - Validation passed: `go test ./pkg/dsl -run 'TestDecompileRoutingPreservesStructureSequenceLists|TestDecompileRoutingOmitsRemovedStructureNormalizer|TestDecompileRoutingRoundTripsToolsDynamicRetrievalPluginConfig|TestDecompileRoutingPreservesRawPluginConfigMaps'`; `go test ./pkg/dsl`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/dsl/decompiler.go"`, including `make test-semantic-router`.
  - Local smoke remains blocked by environment, not by this code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.
  - TD039 remains open, but one current Dashboard backend cross-boundary import is narrowed: tools DB path resolution moved from `dashboard/backend/router/core_routes.go` into `dashboard/backend/routercontract/tools.go`, with route-level fallback coverage and adapter-level parse coverage.
  - Validation passed: `go test ./router ./routercontract` from `dashboard/backend`; `make dashboard-check`.
  - TD039 has a second narrowed Dashboard backend slice: OpenClaw model gateway discovery no longer parses router listener YAML inside `handlers/openclaw.go`; `dashboard/backend/routercontract.ReadFirstListenerEndpoint` owns the dashboard-facing listener adapter.
  - Validation passed: `go test ./handlers ./routercontract -run 'TestResolveOpenClawModelBaseURL|TestDefaultOpenClawModelBaseURL|TestReadFirstListenerEndpoint|TestOpenClawModelGatewayContainerName'` from `dashboard/backend`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="..."`, including `make agent-validate`, `make vllm-sr-test`, `make vllm-sr-sim-test`, `make test-semantic-router`, and `make dashboard-check`.
  - Feature integration and local smoke remain blocked by environment, not by this code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.
  - 2026-05-25 current-source recheck found the Dashboard security-policy apply TODO still live: generated policy fragments were deep-merged and written before validating the merged router config.
  - TD039 has a third narrowed Dashboard backend slice: `dashboard/backend/handlers/security_policy_apply.go` now owns generated-policy write/hot-reload application, and `dashboard/backend/handlers/security_policy_config_validation.go` validates the generated config with canonical endpoint rules, the router parser, and parsed endpoint address checks before it is written or applied.
  - `dashboard/backend/handlers/security_policy.go` dropped from 409 to 354 lines, removing the production handler's structure-warning breach in the changed-set gate.
  - Validation passed: `go test ./handlers -run 'TestValidateMergedSecurityConfigAcceptsPolicyMerge|TestApplySecurityFragmentRejectsInvalidMergedConfigBeforeWrite'` from `dashboard/backend`; `make dashboard-check`.
  - Scorecard movement recorded in `docs/agent/architecture-scorecard.md`; ASR005 remains open for the rest of TD015 and TD039.
- ASR006 progress on 2026-05-24:
  - Current-source recheck found TD034 partially stale for ML pipeline workflow state: jobs and progress events already use `dashboard/backend/workflowstore`, but restart recovery left `pending` jobs non-terminal.
  - ML pipeline recovery now fails both `pending` and `running` jobs after dashboard restart and appends a durable `recovered_after_restart` progress event for UI/API catch-up.
  - Validation passed: `go test ./workflowstore` from `dashboard/backend`; `make dashboard-check`.
  - Current-source recheck found another TD034 slice narrower than the original text: router runtime already has a vector-store Postgres metadata registry and startup load path, but CLI local runtime provisioning did not inject the required `metadata_postgres` block.
  - CLI storage defaults now inject vector-store metadata Postgres connection defaults independently from semantic-cache Milvus defaults and skip disabled vector-store metadata.
  - Vector-store docs now use the current `backend_type` / `metadata_store` contract instead of stale `provider` syntax.
  - Validation passed: `python3 -m pytest src/vllm-sr/tests/test_storage_backends.py -q`; `make vllm-sr-test`.
  - Current-source recheck found router replay backend selection narrower than the original TD034 text: canonical defaults and docs prefer durable Postgres replay storage, but extproc runtime backend resolution still fell back to memory whenever `store_backend` was omitted.
  - Router replay backend resolution now uses the full `RouterReplayConfig`; explicit `store_backend` wins, durable backend config blocks imply their backend, and only an otherwise empty config falls back to memory for compatibility.
  - Validation passed: `go test ./pkg/extproc -run 'TestResolveReplayStoreBackend|TestInitializeReplayRecordersUsesGlobalReplayDefault|TestApplyDecisionResultToContextUsesEffectiveRouterReplayConfig|TestBuildReplayPostgresConfigUsesUnifiedTableName' -v`.
  - Current-source recheck found the Response API portion of TD034 stale: Response API storage defaults to Redis, CLI local runtime injects Redis defaults, state taxonomy records Redis as `shared_durable_workflow_state`, and `e2e/testcases/response_api_restart_recovery.go` verifies Redis-backed restart recovery in the Redis profile.
  - `src/semantic-router/pkg/responsestore/redis_store_integration_test.go` now carries an `integration` build tag so default package tests no longer require a local Redis instance; current troubleshooting now describes Redis Response API connection failures instead of claiming Redis storage is unimplemented.
  - Validation passed: `go test ./pkg/responsestore`; `cd e2e && go test ./testcases`; `cd src/vllm-sr && python3 -m pytest tests/test_runtime_support.py tests/test_storage_backends.py -q`.
  - Current-source recheck found the model-selection learning-state portion of TD034 narrower than the original text: Elo already has optional file-backed ratings, and RL-driven had a storage seam, but RL-driven user personalization and session context were not saved.
  - RL-driven selection now persists user-specific preferences and session context through the existing `storage_path` format using reserved encoded keys, `FileEloStorage.Close()` no longer blocks when auto-save was never started, and stale active model-research campaign references were removed from the TD034 state scope because current source no longer contains `dashboard/backend/modelresearch`.
  - Validation passed: `go test ./pkg/selection -run 'TestFileEloStorage_CloseWithoutAutoSaveDoesNotBlock|TestFileEloStorage_SaveAndLoad|TestRLDrivenSelector_StoragePersistsUserAndSessionState'`; `go test ./pkg/selection`; `make agent-ci-gate CHANGED_FILES="..."`.
  - Current-source recheck found TD031 partly stale for the common runtime path: classification, memory, vector-store, file-store, and config fanout already have `pkg/routerruntime` ownership, but feedback/ratings/RL-state APIs still read model-selection state from `selection.GlobalRegistry`.
  - A focused config-update fanout recheck found the old single-consumer channel wording stale: `config.SubscribeConfigUpdates` owns per-subscriber buffered channels, `config.Replace` fanouts to all current subscribers, and `WatchConfigUpdates` is now only a compatibility wrapper.
  - Validation passed: `go test ./pkg/config -run TestSubscribeConfigUpdatesFanout -count=1`.
  - `routerruntime.Registry` now carries the model-selection registry, `extproc` publishes `OpenAIRouter.ModelSelector` into it on startup/reload, and API handlers prefer that runtime registry before falling back to the legacy global.
  - Validation passed: `go test ./pkg/routerruntime`; `go test ./pkg/apiserver -run 'TestHandleFeedback|TestHandleGetRatings|TestRuntimeRegistryResolvesSharedDependencies'`; `go test ./pkg/extproc -run TestReloadRouterFromConfigPublishesRuntimeRegistryAfterSwap`.
  - Changed-set validation passed after extracting feedback selector updates back out of the API hotspot: `make agent-ci-gate CHANGED_FILES="..."`, including `make agent-validate`, `make vllm-sr-test`, `make vllm-sr-sim-test`, `make test-semantic-router`, and `make dashboard-check`.
  - Feature integration and local smoke remain blocked by environment, not by this code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.
  - A later current-source recheck found the memory-store runtime-boundary debt narrower than the original broad global-state wording: ExtProc already publishes `OpenAIRouter.MemoryStore` into `routerruntime.Registry`, but `apiserver.InitWithRuntime` still adopted `memory.GetGlobalMemoryStore()` when the registry had not yet published a store.
  - Runtime-owned API startup now uses only `runtimeRegistry.MemoryStore()` when a registry is provided, while the legacy nil-registry `Init` path keeps the compatibility global fallback. Request-time memory APIs still observe later registry publication through `currentMemoryStore()`.
  - Validation passed: `go test ./pkg/apiserver -run 'TestResolveMemoryStoreUsesRuntimeRegistryOnly|TestResolveMemoryStorePreservesLegacyGlobalFallback|TestRuntimeRegistryResolvesSharedDependencies'`; `go test ./pkg/routerruntime`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/apiserver/server.go,src/semantic-router/pkg/apiserver/server_memory_test.go,docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md"`; `make agent-scorecard`; `git diff --check`.
  - Local dev/smoke remains blocked by environment, not by this code path: `docker info` and `make agent-dev ENV=cpu` both failed because Docker cannot connect to `/Users/bitliu/.docker/run/docker.sock`.
  - Current-source recheck found another TD034 router runtime status gap: `/startup-status` already resolved Redis-backed startup state, but `/ready` still read only the file-backed status path and could false-negative in Redis/shared-status deployments.
  - `/ready` now uses the same startup-state resolver as `/startup-status`, a private loader seam covers the shared resolver path without a live Redis service, and the runtime file-backend warning now names the current `startup_status.store_backend` key.
  - Validation passed: `go test ./pkg/apiserver -run 'TestHandleReady|TestHandleStartupStatus' -count=1`.
  - Current-source recheck found the browser-local dashboard state gap narrower than the original TD034 text: conversation history already has a browser-local disclaimer and saved caps, while queued playground tasks were restored and persisted without shape validation or bounds.
  - `playgroundQueueStorage.ts` now owns pure queue normalization/pruning, `usePlaygroundQueue` uses it on load and persistence writes, and Vitest covers malformed restore data plus task/conversation caps.
  - Validation passed: `npm run test:unit`; `npm run type-check`; `npm run lint`.
  - A follow-up recheck found `useConversationStorage` still accepted malformed or overlarge localStorage records during restore. `conversationStorage.ts` now owns pure conversation normalization/pruning, and Vitest covers malformed restore filtering, duplicate ID normalization, typed payload validation, and save-time caps.
  - Validation passed: `npm run test:unit -- conversationStorage playgroundQueueStorage routeManifest dashboardPageOverview`; `npm run type-check`; `npm run lint`.
  - The auth-token browser storage slice is also narrowed: `authFetch` now normalizes and bounds localStorage token values before they reach headers, cookies, or query-token transports, and `AuthContext` rejects invalid login tokens before marking the session authenticated.
  - Validation passed: `npm run test:unit -- authFetch conversationStorage playgroundQueueStorage routeManifest dashboardPageOverview`; `npm run type-check`; `npm run lint`; `make dashboard-check`.
  - Dashboard auth now has a server-owned session-cookie step: login/bootstrap set an HttpOnly `vsr_session` cookie, `POST /api/auth/logout` clears it, the router proxies the logout route, and frontend logout calls the server while preserving immediate local cleanup.
  - Validation passed: `go test ./auth ./router` from `dashboard/backend`; `npm run test:unit -- authFetch conversationStorage playgroundQueueStorage routeManifest dashboardPageOverview`; `npm run type-check`; `npm run lint`; `make dashboard-check`.
  - A follow-up auth reload slice closes the cookie-first frontend gap: `AuthProvider` now probes `/api/auth/me` with same-origin credentials on startup, the new `authSession` support module treats a returned user as an authenticated server session even without local token material, and backend coverage verifies cookie-only request authentication through `AuthenticateRequest`.
  - Validation passed: `npm run test:unit -- authSession authFetch conversationStorage playgroundQueueStorage routeManifest dashboardPageOverview`; `npm run type-check`; `npm run lint`; `go test ./auth ./router` from `dashboard/backend`.
  - A backend auth intake and revocation slice now matches the frontend token boundary: bearer headers, `vsr_session` cookies, and `authToken` query parameters all pass through one bounded token normalizer before JWT parsing; newly issued tokens carry persisted session ids; logout revokes the current session before clearing the cookie; session-cookie issue/clear responses carry explicit expiry timestamps; expired/revoked session rows are indexed and pruned on auth-store open after the retention window; and cookie/session tests moved to focused files so broad handler/store tests do not grow as hotspots.
  - Validation passed: `go test -count=1 ./auth ./router` from `dashboard/backend`; `make dashboard-check`.
  - A follow-up Dashboard auth store ownership slice moved role/permission persistence into `permission_store.go` and audit persistence into `audit_store.go`, leaving `store.go` focused on SQLite construction/lifecycle plus user CRUD and `session_store.go` focused on auth-session lifecycle/pruning.
  - `dashboard/backend/auth/store.go` dropped from 525 to 245 lines and no longer triggers the changed-set structure warning.
  - Validation passed: `go test -count=1 ./auth ./router` from `dashboard/backend`; `make dashboard-check`.
  - A follow-up Dashboard auth handler ownership slice removed the 556-line `http_handlers.go` hotspot by moving public bootstrap/login/me handlers to `public_handlers.go`, admin user CRUD to `admin_user_handlers.go`, admin permission/audit/password handlers to `admin_meta_handlers.go`, and session-user helpers to `session_user.go`.
  - The largest new auth handler file is 290 lines, so the changed set no longer trips auth handler/store structure warnings.
  - Validation passed: `go test -count=1 ./auth ./router` from `dashboard/backend`; `make dashboard-check`.
  - GMTRouter local personalization state now has a sibling storage helper that creates nested storage directories, snapshots state before encoding, writes atomically, backs up corrupted JSON, and normalizes loaded state before use.
  - Validation passed: `go test ./pkg/selection -run 'TestGMTRouterSelector_Persistence|TestGMTRouterStateStorageCreatesDirectoryAndNormalizes|TestGMTRouterStateStorageBacksUpCorruptedFile|TestGMTRouterSelector_PersonalizedSelection|TestRLDrivenSelector_StoragePersistsUserAndSessionState'`; `go test ./pkg/selection`.
  - 2026-05-25 current-source recheck found a narrow CLI runtime KB bootstrap-state gap: `.vllm-sr/knowledge_bases/.bootstrap-state.yaml` was written directly and corrupted YAML could abort runtime config resolution.
  - CLI runtime support now treats invalid KB bootstrap marker YAML as empty state, rewrites the marker through a same-directory temporary file plus `os.replace`, and records the marker contract in the state taxonomy.
  - Validation passed: `python -m pytest src/vllm-sr/tests/test_runtime_support.py`; `make vllm-sr-test`.
  - A follow-up recheck found the behavior fix still left `runtime_support.py` as the owner of runtime paths, KB state, and config mutation.
  - `runtime_help.py`, `runtime_paths.py`, `runtime_kb.py`, and `runtime_config_mutation.py` now own those seams; `runtime.py` dropped from 457 to 398 lines, `runtime_support.py` dropped from 703 to 205 lines, and KB coverage moved into `test_runtime_kb.py` so the changed-set structure gate no longer warns on the CLI runtime command/support/test files.
  - Validation passed: `python -m pytest src/vllm-sr/tests/test_runtime_support.py src/vllm-sr/tests/test_runtime_kb.py src/vllm-sr/tests/test_cli_main.py -q`.
  - Current-source recheck found RLDriven and GMTRouter selector persistence had one remaining config-runtime reachability gap: docs and selectors exposed persistence and tuning fields, but extproc selector construction only forwarded decision-scoped Elo and RouterDC config.
  - Router config structs, Python CLI schema, and Dashboard DSL field schema now expose the documented RLDriven/GMTRouter fields, `buildModelSelectionConfig` forwards decision-scoped learning selector config into runtime construction, and `router_selection_config_test.go` plus CLI tests cover mapping/default preservation.
  - Validation passed: `go test ./pkg/extproc -run 'TestBuildModelSelectionConfigUsesDecisionScopedLearningState|TestBuildModelSelectionConfigPreservesLearningDefaultsWhenUnset'`; `go test ./pkg/config -run 'TestReferenceConfig|TestConfigFragments'`; `python3 -m pytest src/vllm-sr/tests/test_algorithm_config.py -q`; `npm run type-check`; `npm run lint`.
  - Changed-set validation passed after formatting and test-structure cleanup: `make agent-ci-gate CHANGED_FILES="..."`, including `make agent-validate`, `make vllm-sr-test`, `make test-semantic-router`, and `make dashboard-check`; `make agent-scorecard` and `git diff --check` also passed.
  - Feature integration and local smoke remain blocked by environment, not by this code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`, and `make vllm-sr-test-integration` stops at `vllm-sr-build` with the same Docker daemon connection failure.
  - 2026-05-25 current-source recheck found the memory-store runtime-boundary rule had not fully propagated to API vector/file/selector helpers: registry-backed API servers could still fall back to process-wide vector manager, embedder, ingestion pipeline, file store, or `selection.GlobalRegistry` before runtime publication.
  - `runtime_dependencies.go` now treats a present `routerruntime.Registry` as authoritative for vector/file/selector helpers. It returns registry-published dependencies when present and nil before publication; legacy globals remain only for nil-registry compatibility callers.
  - Validation passed: `go test ./pkg/apiserver -run 'TestLegacyRuntimeGlobalsResolveThroughSynchronizedAccessors|TestRuntimeRegistrySuppressesLegacyVectorGlobalsUntilPublished|TestRuntimeRegistrySuppressesLegacySelectionGlobalUntilPublished|TestHandleFeedbackUsesRuntimeSelectionRegistry|TestHandleGetRatingsUsesRuntimeSelectionRegistry|TestResolveMemoryStoreUsesRuntimeRegistryOnly'`; `go test ./pkg/routerruntime`.
  - Package validation passed: `go test ./pkg/apiserver`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/apiserver/runtime_dependencies.go,src/semantic-router/pkg/apiserver/runtime_globals_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - Validation passed: `make agent-scorecard`; `git diff --check`.
  - Local smoke remains blocked by environment, not by this code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.
  - Scorecard movement recorded in `docs/agent/architecture-scorecard.md`; ASR006 remains open for TD031 and the broader TD034 state surfaces.
  - 2026-05-25 current-source recheck found one more `InitWithRuntime` startup-state leak: `resolveAPIServerConfig` could still fall back to `config.Get()` when a runtime registry existed but had not published its current config.
  - `server.go` now treats a present runtime registry as authoritative for API startup config, returning only `runtimeRegistry.CurrentConfig()` on the registry path while preserving `config.Get()` for the legacy nil-registry entrypoint.
  - `server_memory_test.go` now covers registry-only config resolution before and after runtime publication plus the preserved legacy fallback.
  - Validation passed: `go test ./pkg/apiserver -run 'TestResolveAPIServerConfigUsesRuntimeRegistryOnly|TestResolveAPIServerConfigPreservesLegacyGlobalFallback|TestResolveMemoryStoreUsesRuntimeRegistryOnly|TestResolveMemoryStorePreservesLegacyGlobalFallback'`.
  - Package validation passed: `go test ./pkg/apiserver`; `go test ./pkg/routerruntime`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/apiserver/server.go,src/semantic-router/pkg/apiserver/server_memory_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - Validation passed: `make agent-scorecard`; `git diff --check`.
  - Local smoke remains blocked by environment, not by this code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.
  - A follow-up current-source recheck found a separate live global-state leak in the API runtime config updater: registry-backed `liveRuntimeConfig.Update()` called `routerruntime.Registry.RefreshRuntimeConfig`, which refreshed the classification service and indirectly called `config.Replace()` through `ClassificationService.RefreshRuntimeConfig`.
  - `ClassificationService.RefreshRuntimeConfig` is now service-local. The preserved legacy global publication path is explicit in `buildConfigUpdater(nil, ...)`, and `ClassificationService.UpdateConfig` keeps its compatibility behavior.
  - Focused tests now cover service-local refresh without replacing `config.Get()`, the legacy nil-registry global publication path, and the registry-backed updater path that refreshes registry/service state without rewriting process-wide config.
  - Validation passed: `go test ./pkg/services -run 'TestClassificationServiceRefreshRuntimeConfig'`; `go test ./pkg/apiserver -run 'TestBuildConfigUpdater|TestResolveAPIServerConfig|TestResolveMemoryStore'`.
  - Package validation passed: `go test ./pkg/services ./pkg/apiserver ./pkg/routerruntime`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/services/classification_runtime_config.go,src/semantic-router/pkg/services/classification_update_test.go,src/semantic-router/pkg/apiserver/server.go,src/semantic-router/pkg/apiserver/server_memory_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - Validation passed: `make agent-scorecard`; `git diff --check`.
  - Local smoke remains blocked by environment, not by this code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.
  - A follow-up current-source recheck found the same stale-state pattern in RL-driven looper construction: ExtProc owned the active selector registry on `OpenAIRouter.ModelSelector`, but `NewRLDrivenLooper` read `selection.GlobalRegistry` directly.
  - `looper.FactoryWithSelectionRegistry` now lets runtime callers pass the active selector registry, `NewRLDrivenLooperWithSelectionRegistry` resolves the RL-driven selector from that registry, and direct `Factory` / `NewRLDrivenLooper` calls keep the old global-registry compatibility behavior.
  - `handleLooperExecution` now creates loopers from `r.ModelSelector`, so RL-driven multi-round execution cannot accidentally bind to another runtime's process-wide selector state.
  - Validation passed: `go test ./pkg/looper -run 'TestFactory_RLDriven|TestFactoryWithSelectionRegistryUsesRuntimeRLDrivenSelector|TestNewRLDrivenLooperPreservesGlobalRegistryCompatibility|TestRLDrivenLooper_SelectorIntegration'`; `go test ./pkg/extproc -run 'TestShouldUseLooper|TestCreateLooperResponseIncludesTrackedHeaders'`.
  - Package and changed-set validation passed: `go test ./pkg/looper`; `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/looper/looper.go,src/semantic-router/pkg/looper/rl_driven.go,src/semantic-router/pkg/looper/rl_driven_test.go,src/semantic-router/pkg/extproc/req_filter_looper.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`.
  - A follow-up current-source recheck found another main-path config leak in ExtProc startup: `Server.Start()` read `config.Get().Looper.GetGRPCMaxMsgSize()` even when the server already had an active router config or runtime registry.
  - `server.go` now resolves gRPC max-message-size config from the active router config first, then the runtime registry, then the legacy process-wide fallback.
  - Focused tests cover router-config precedence, runtime-registry precedence, and preserved legacy fallback: `go test ./pkg/extproc -run 'TestConfiguredGRPCMaxMessageSize'`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_reload_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A follow-up current-source recheck found two request-time config leaks in ExtProc plugin paths: modality routing read `config.Get()` directly, and system-prompt injection preferred the process-wide config before router-local state while also dereferencing a missing decision before nil checking it.
  - `router_config.go` now centralizes router-owned config and decision precedence for those plugin paths, `handleModalityFromDecision` uses the router config first, and system-prompt injection returns unchanged body when no matching router/classifier decision exists.
  - Focused tests cover router/global modality config conflicts, router/global system-prompt decision conflicts, no global fallback when router config is present, and missing-decision no-panic behavior: `go test ./pkg/extproc -run 'TestHandleModalityFromDecision|TestAddSystemPrompt'`.
  - A final current-source recheck in the same startup cluster found `watchConfigAndReload` still chose file watching versus Kubernetes update watching from `config.Get()`.
  - `usesKubernetesConfigSource` now applies router-config, runtime-registry, legacy-global precedence to that watcher decision.
  - Focused tests cover router config overriding conflicting runtime/global source, runtime registry source winning over global fallback, and preserved legacy fallback: `go test ./pkg/extproc -run 'TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource'`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_reload_test.go,src/semantic-router/pkg/extproc/req_filter_modality.go,src/semantic-router/pkg/extproc/req_filter_modality_test.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A constructor-boundary recheck found the same stale-global issue one step earlier: `NewServer` entered `NewOpenAIRouter(configPath)` even when a `routerruntime.Registry` already held current config.
  - `resolveInitialRouterConfig` and `newOpenAIRouterForServer` now make server construction use runtime-registry config first, parse the explicit config file on the registry path when the registry is not yet populated, and reserve legacy `config.Replace` publication for nil-registry callers.
  - Focused tests cover registry config winning over a conflicting Kubernetes global without parsing a missing file or changing `config.Get()`, empty-registry file parsing without adopting a conflicting global, plus preserved legacy Kubernetes fallback: `go test ./pkg/extproc -run 'TestResolveInitialRouterConfig|TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource'`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_build.go,src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_reload_test.go,src/semantic-router/pkg/extproc/req_filter_modality.go,src/semantic-router/pkg/extproc/req_filter_modality_test.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A follow-up server-config resolver recheck tightened the empty-registry edge: once `Server.runtime` is present, `resolveServerConfig` returns only runtime-owned config instead of falling through to `config.Get()`. gRPC startup now uses the explicit default message size before publication, and Kubernetes watcher selection stays false instead of adopting a conflicting global config.
  - Focused validation passed: `go test ./pkg/extproc -run 'TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource|TestResolveInitialRouterConfig' -count=1`.
  - A follow-up reload recheck found registry-backed file reload still calling `replaceReloadConfig`, which could publish another runtime's file config into the process-wide compatibility cache after a successful router swap.
  - Registry-backed reload now updates the active router service plus `routerruntime.Registry` only. Legacy nil-registry reload still calls `replaceReloadConfig`, preserving the compatibility path.
  - Focused validation passed: `go test ./pkg/extproc -run 'TestReloadRouterFromFile|TestReloadRouterFromConfig|TestResolveInitialRouterConfig|TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource' -count=1`.
  - A request-time helper recheck found `OpenAIRouter.routerConfig()` and `decisionByName()` still allowed a present-but-unpublished runtime registry to fall through to the process-wide config cache.
  - `router_config.go` now treats the runtime registry as authoritative for request-time config and decision lookup: router-local config wins, registry config wins next, an empty registry returns nil, and only nil-registry legacy callers can use `config.Get()`.
  - Focused validation passed: `go test ./pkg/extproc -run 'TestRouterConfig|TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource|TestResolveInitialRouterConfig|TestReloadRouterFromConfig' -count=1`.
  - A structure-gate recheck found `server.go` still carried config watch/reload setup, debounce scheduling, file stat logging, and Kubernetes update handling inside the core server file, leaving it at 542 lines with a 132-line nested watcher function.
  - `server_config_watch.go` now owns the config watcher loop and reload event helpers. `server.go` keeps server lifecycle and runtime publication, dropping to 347 lines and clearing the changed-file file-size/function warnings for the core server file.
  - Focused validation passed: `go test ./pkg/extproc -run 'TestShouldReloadForConfigEvent|TestRouterConfig|TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource|TestResolveInitialRouterConfig|TestReloadRouterFromFile|TestReloadRouterFromConfig' -count=1`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/router_config_test.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_config_watch.go,src/semantic-router/pkg/extproc/server_reload_test.go,docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A follow-up structure recheck found `server_reload_test.go` was still a 683-line changed-file regression-suite hotspot even after the production watcher loop was split.
  - Runtime/config precedence coverage moved to `server_config_runtime_test.go`, reload runtime-publication coverage moved to `server_reload_runtime_test.go`, and `server_reload_test.go` now focuses on file reload order, AMD model preflight, and reload seam helpers at 283 lines.
  - Focused validation passed: `go test ./pkg/extproc -run 'TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource|TestResolveInitialRouterConfig|TestReloadRouterFromFile|TestReloadRouterFromConfig|TestShouldReloadForConfigEvent|TestRouterConfig' -count=1`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/router_config_test.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_config_watch.go,src/semantic-router/pkg/extproc/server_config_runtime_test.go,src/semantic-router/pkg/extproc/server_reload_test.go,src/semantic-router/pkg/extproc/server_reload_runtime_test.go,docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md"`, including `make agent-validate`, changed-file lint, Go structural lint, structure checks, and `make test-semantic-router`.
  - A vectorstore RAG runtime-boundary recheck found the older "global embedder/vector manager" wording stale: `OpenAIRouter` already resolves request-time RAG manager/embedder from `routerruntime.Registry`.
  - `extproc.router_runtime_services_test` now locks registry-only manager/embedder lookup for nil router, nil registry, pre-publication, and published-runtime states; `req_filter_rag_vectorstore.go` and response-memory naming now use router/runtime-owned wording instead of "global" wording.
  - Validation passed: `go test ./pkg/extproc -run 'TestRouterVectorStoreRuntime|TestScheduleResponseMemoryStore|TestResolveVectorStoreRetrievalParams|TestVectorStoreRAG'`.
  - A managed knowledge-base mutation recheck found one more API-side config publication leak: `persistConfigAndSync` could still call `config.Replace(newCfg)` when `runtimeConfig` was absent even if the API server carried a `routerruntime.Registry`.
  - `apiserver.publishConfigMutation` now centralizes config mutation publication: server-local state is updated, `liveRuntimeConfig` is preferred when present, runtime registries are updated without touching the process-wide compatibility config, and only legacy nil-registry callers use `config.Replace`.
  - Focused validation passed: `go test ./pkg/apiserver -run 'TestHandleKnowledgeBaseMutationWithRuntimeRegistryDoesNotReplaceGlobalConfig|TestHandleKnowledgeBaseLifecycle|TestBuildConfigUpdater|TestResolveAPIServerConfig' -count=1`.
  - A follow-up API classification-service recheck found the same stale-global pattern: registry-backed API startup and the live service resolver could still read `services.GetGlobalClassificationService()` before the runtime registry published a classification service.
  - `resolveClassificationService` and the live resolver now treat a present runtime registry as authoritative for classification service lookup, while the legacy nil-registry entrypoint keeps the retrying global-service fallback.
  - Focused validation passed: `go test ./pkg/apiserver -run 'TestResolveClassificationService|TestResolveAPIServerConfig|TestResolveMemoryStore|TestBuildConfigUpdater' -count=1`.
- Governance recheck on 2026-05-24:
  - TD023 and TD029 are already `Status: Closed`; ASR007 now targets TD020 only and treats those extproc request/response entries as historical guards.
- ASR007 progress on 2026-05-24:
  - Current-source recheck confirmed TD020 remains open, but the service layer still owned legacy category, PII, and jailbreak mapping bootstrap.
  - `classification.NewLegacyClassifierFromConfig` now owns that legacy bootstrap seam, and service assembly/refresh paths call it instead of reimplementing mapping loading.
  - The mapping gate tests moved from `pkg/services` to `pkg/classification` and now include nil-config coverage.
  - Validation passed: `go test ./pkg/classification ./pkg/services`.
  - Scorecard movement recorded in `docs/agent/architecture-scorecard.md`; ASR007 remains open for larger classifier, unified-classifier, discovery, and family-adapter slices.
  - A follow-up current-source recheck found a narrower live slice in `model_discovery.go`: directory normalization, model scanning, LoRA/legacy classification, preferred architecture selection, and legacy unified-classifier label loading still mixed multiple responsibilities into model discovery and initialization.
  - `classification.model_discovery_scan` now owns model directory normalization, scanning, LoRA registry/name classification, legacy fallback collection, and preferred architecture selection.
  - `classification.unified_legacy_labels` now owns legacy label loading, ordered mapping validation, and support for both jailbreak mapping field conventions.
  - `model_discovery.go` delegates scanning and label loading through those helpers, keeping the public auto-discovery API unchanged while reducing the discovery hotspot from 500 to 316 lines.
  - Focused tests cover successful ordered label loading, alternate jailbreak `label_to_id` / `id_to_label` mappings, and sparse mapping rejection.
  - Validation passed: `go test ./pkg/classification -run 'TestLoadLegacyUnifiedLabels|TestAutoDiscoverModels|TestValidateModelPaths|TestModelPathsIsComplete'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/model_discovery.go,src/semantic-router/pkg/classification/model_discovery_scan.go,src/semantic-router/pkg/classification/unified_legacy_labels.go,src/semantic-router/pkg/classification/unified_legacy_labels_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A services-side current-source recheck found `classification.go` still mixed core service lifecycle, PII and security request orchestration, PII response shaping, model recommendation, routing-decision helpers, and unified batch classification wrappers.
  - `services.classification_pii`, `services.classification_pii_response`, `services.classification_security`, `services.classification_recommendation`, and `services.classification_unified_batch` now own those service API slices. `classification.go` keeps service construction, auto-discovery assembly, intent classification, and config accessors, dropping from 627 to 219 lines and clearing the changed-file structure warning for that file.
  - Validation passed: `go test ./pkg/services -run 'TestDetectPII|TestBuildPIIResponse|TestClassificationService_ClassifyBatchUnified|TestGetRecommendedModel'`.
  - A unified-classifier current-source recheck found `unified_classifier.go` still mixed shared public types, stats, global access, native batch calls, C result decoding, initialization, mode selection, and LoRA lazy-init while the stub duplicated the public type surface.
  - `classification.unified_classifier_types` now owns shared public types, stats, global access, and compatibility wrappers; `classification.unified_classifier_cgo_results` owns native batch invocation plus C result decoding; the stub keeps only stub native-capability behavior. The cgo orchestrator dropped from 660 to 309 lines and the stub from 179 to 84 lines.
  - Robustness fixes in the same slice reject empty label sets before native pointer setup and remove the old fixed 1000-result unsafe array cap from LoRA/native result decoding.
  - Validation passed: `go test ./pkg/classification -run 'TestUnifiedClassifier|TestCurrentNativeBackendCapabilitiesShape|TestUnifiedClassifierRejectsUnsupportedNativeCapabilities|TestUnifiedClassifierStatsIncludeNativeCapabilities'`; `go test ./pkg/classification ./pkg/services`.
  - A follow-up unified-classifier LoRA lifecycle recheck found the narrowed cgo orchestrator still owned LoRA C binding declarations, lazy-init locking, mode/capability guards, and lifecycle logging.
  - `classification.unified_classifier_lora` now owns LoRA binding initialization, native capability checks, and lazy-init concurrency guards. `unified_classifier.go` keeps legacy native initialization plus `ClassifyBatch` dispatch only and dropped from 309 to 175 lines.
  - Validation passed: `go test ./pkg/classification -run 'TestUnifiedClassifier|TestCurrentNativeBackendCapabilitiesShape|TestUnifiedClassifierRejectsUnsupportedNativeCapabilities|TestUnifiedClassifierStatsIncludeNativeCapabilities'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/embedding_classifier.go,src/semantic-router/pkg/classification/embedding_classifier_backend.go,src/semantic-router/pkg/classification/embedding_classifier_preload_support.go,src/semantic-router/pkg/classification/embedding_classifier_scoring.go,src/semantic-router/pkg/classification/unified_classifier.go,src/semantic-router/pkg/classification/unified_classifier_lora.go,src/semantic-router/pkg/classification/unified_classifier_cgo_results.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A built-in classifier family recheck found `classifier.go` still mixed constructor/state-table ownership with category, jailbreak, and PII model enablement, initialization, and jailbreak request helpers.
  - `classification.classifier_builtin_models` now owns those built-in family behaviors. `classifier.go` keeps the dependency table, option wiring, and constructor, dropping from 383 to 187 lines.
  - Validation passed: `go test ./pkg/classification ./pkg/services`.
  - A request-time signal orchestration recheck found `classifier_signal_context.go` had become the live TD020 hotspot at 783 lines, mixing central signal dispatch setup with jailbreak, PII, embedding, and preference evaluators.
  - `classification.classifier_signal_jailbreak` and `classification.classifier_signal_pii` now own request-time safety/PII evaluator fanout, cached native inference results, rule evaluation, and result mutation.
  - The existing embedding and preference support files now own their request-time evaluators, and `classifier_signal_context.go` dropped to 391 lines, below the repo structure-rule warning threshold.
  - Validation passed: `go test ./pkg/classification ./pkg/services`; `go test ./pkg/classification -run 'TestPII|TestJailbreak|TestSignal|TestEmbedding|TestPreference|TestContext|TestClassifier'`.
  - A model-family lifecycle recheck found `classifier_model_select.go` still mixed candidate model selection with fact-check, hallucination, feedback, preference, and language model lifecycle/runtime methods.
  - `classification.classifier_signal_model_families` initially carried those auxiliary signal model-family methods, and `classifier_model_select.go` kept decision lookup plus candidate model selection, dropping from 423 to 146 lines.
  - Validation passed: `go test ./pkg/classification ./pkg/services`.
  - A follow-up model-family lifecycle recheck found `classifier_signal_model_families.go` had become a cross-family holding file for fact-check, hallucination, feedback, preference, and language lifecycle/readiness/public wrapper methods.
  - `classification.classifier_fact_hallucination_lifecycle` now owns fact-check and hallucination readiness, initialization, public calls, NLI fallback, and getters.
  - `classification.classifier_feedback_lifecycle`, `classification.classifier_preference_lifecycle`, and `classification.classifier_language_lifecycle` now own their respective family readiness, initialization, public calls where applicable, and logging helpers.
  - `classifier_signal_model_families.go` was deleted; replacement files are 158, 55, 62, and 28 lines.
  - Validation passed: `go test ./pkg/classification -run 'Test.*Hallucination|Test.*FactCheck|Test.*Feedback|Test.*Preference|Test.*Language|TestClassifier'`; `go test ./pkg/classification ./pkg/services`.
  - A signal evaluation bridge recheck found `classifier_signal_eval.go` still mixed public signal result contracts, used-signal dependency expansion, authz header evaluation, and decision-engine bridging after the request-time family split.
  - `classification.classifier_signal_results`, `classification.classifier_signal_usage`, `classification.classifier_signal_authz`, and `classification.classifier_signal_decision` now own those seams separately. `classifier_signal_eval.go` keeps only the public all-signal convenience entrypoints and dropped from 431 to 13 lines.
  - Validation passed: `go test ./pkg/classification ./pkg/services`; `go test ./pkg/classification -run 'Test.*Signal|Test.*Authz|Test.*Decision|Test.*Projection|Test.*PII|Test.*Jailbreak'`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_signal_eval.go,src/semantic-router/pkg/classification/classifier_signal_authz.go,src/semantic-router/pkg/classification/classifier_signal_decision.go,src/semantic-router/pkg/classification/classifier_signal_results.go,src/semantic-router/pkg/classification/classifier_signal_usage.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`.
  - A native family init/inference adapter recheck found `classifier_init.go` still mixed category, jailbreak, and PII native initializer/inference adapters, mmBERT variants, and API result structs in one cross-family file.
  - `classification.classifier_category_init`, `classification.classifier_jailbreak_init`, and `classification.classifier_pii_init` now own those family-specific native adapter seams separately, replacing the 403-line mixed file with 126-line, 164-line, and 129-line files.
  - Validation passed: `go test ./pkg/classification ./pkg/services`; `go test ./pkg/classification -run 'Test.*Classifier|Test.*Jailbreak|Test.*PII|Test.*Native|Test.*Unified'`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_init.go,src/semantic-router/pkg/classification/classifier_category_init.go,src/semantic-router/pkg/classification/classifier_jailbreak_init.go,src/semantic-router/pkg/classification/classifier_pii_init.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`.
  - A services signal contract recheck found `classification_signal_contract.go` still mixed API DTOs, eval execution, intent response shaping, matched-signal extraction, and unmatched-signal collection.
  - `services.classification_signal_types`, `services.classification_signal_response`, and `services.classification_signal_matched` now own those seams separately; `classification_signal_contract.go` keeps eval execution only and dropped from 451 to 69 lines.
  - Validation passed: `go test ./pkg/classification ./pkg/services`; `go test ./pkg/services -run 'Test.*Intent|Test.*Classification|Test.*Signal|Test.*Eval|Test.*Recommended|Test.*Routing'`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/services/classification_signal_contract.go,src/semantic-router/pkg/services/classification_signal_matched.go,src/semantic-router/pkg/services/classification_signal_response.go,src/semantic-router/pkg/services/classification_signal_types.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - An embedding classifier recheck found `embedding_classifier.go` still mixed embedding backend FFI indirection and model initialization with candidate preload orchestration, request classification flow, rule scoring, top-k match shaping, cosine similarity, and prototype-bank maintenance.
  - `classification.embedding_classifier_backend` now owns backend FFI helpers and keyword-embedding model initialization, `classification.embedding_classifier_preload_support` owns candidate collection, worker fanout, and preload result collection, while `classification.embedding_classifier_scoring` owns matched-rule scoring and prototype-bank rebuilds. `embedding_classifier.go` keeps classifier state, construction, and text/multimodal classification flow, dropping from 655 to 323 lines.
  - Validation passed: `go test ./pkg/classification -run 'Test.*Embedding|Test.*Prototype|Test.*Classifier|Test.*Signal|Test.*Multimodal'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/embedding_classifier.go,src/semantic-router/pkg/classification/embedding_classifier_backend.go,src/semantic-router/pkg/classification/embedding_classifier_preload_support.go,src/semantic-router/pkg/classification/embedding_classifier_scoring.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A follow-up embedding classifier recheck found the narrowed file still mixed constructor/state ownership with public text and multimodal request paths.
  - `classification.embedding_classifier_text` now owns text `Classify`, `ClassifyAll`, and detailed scoring orchestration, while `classification.embedding_classifier_multimodal` owns multimodal validation, request-image cache usage, multimodal scoring orchestration, and unsupported-audio errors.
  - `embedding_classifier.go` now keeps construction, state, model-type resolution, rules-by-modality indexing, and result DTOs only, dropping from 323 to 122 lines; validation passed: `go test ./pkg/classification -run 'TestEmbeddingClassifier|TestSignal.*Embedding|TestClassifier'`; `go test ./pkg/classification ./pkg/services`.
  - A keyword classifier recheck found `keyword_classifier.go` still mixed classifier construction, Rust-backed dispatch, per-call BM25/N-gram cache state, regex rule preprocessing, regex/fuzzy matching, Levenshtein scoring, and public classify wrappers.
  - `classification.keyword_classifier_regex`, `classification.keyword_classifier_dispatch`, and `classification.keyword_classifier_match` now own rule preprocessing, ordered dispatch/cache state, and regex/fuzzy matching respectively. `keyword_classifier.go` keeps construction/resource/public-wrapper behavior only and dropped from 569 to 165 lines.
  - Validation passed: `go test ./pkg/classification -run 'Test.*Keyword|Test.*BM25|Test.*Ngram|Test.*Fuzzy|Test.*Levenshtein'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/keyword_classifier.go,src/semantic-router/pkg/classification/keyword_classifier_dispatch.go,src/semantic-router/pkg/classification/keyword_classifier_match.go,src/semantic-router/pkg/classification/keyword_classifier_regex.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - An MCP classifier recheck found `mcp_classifier.go` still mixed MCP client construction, tool discovery, text response parsing, classification/category mapping calls, Classifier-level bootstrap, and entropy/metrics runtime logic.
  - `classification.mcp_classifier_client` now owns MCP transport/client lifecycle, tool discovery, shared text-response parsing, classification/probability calls, and `list_categories` mapping conversion.
  - `classification.mcp_classifier_runtime` now owns Classifier-level MCP enablement, category mapping bootstrap, timeout-scoped probability inference, category-name translation, threshold handling, entropy reasoning, and probability-quality metrics.
  - `mcp_classifier.go` keeps only public MCP contracts, the classifier struct, factory helpers, and option wiring, dropping from 593 to 80 lines.
  - Validation passed: `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/mcp_classifier.go,src/semantic-router/pkg/classification/mcp_classifier_client.go,src/semantic-router/pkg/classification/mcp_classifier_runtime.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A hallucination detector recheck found `hallucination_detector.go` still mixed basic hallucination detector lifecycle/span filtering with NLI config, NLI classification, enhanced result contracts, NLI-based span filtering, severity adjustment, and explanations.
  - `classification.hallucination_detector_nli` now owns NLI labels/results, enhanced hallucination result contracts, NLI initialization/classification, enhanced detection, span filtering, threshold defaults, and explanation/severity adjustment.
  - `hallucination_detector.go` keeps detector construction, Candle hallucination-model initialization, basic detection, basic span filtering, and basic readiness only, dropping from 400 to 160 lines.
  - Validation passed: `go test ./pkg/classification -run 'TestHallucination|Test.*Hallucination'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/hallucination_detector.go,src/semantic-router/pkg/classification/hallucination_detector_nli.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A classifier construction recheck found `classifier_construction.go` still mixed classifier-option orchestration, rule-family option construction, category/MCP backend wiring, and native category/jailbreak/PII dependency selection.
  - `classification.classifier_option_rules` now owns keyword, embedding, context, structure, event-context, reask, complexity, contrastive jailbreak, authz, and KB option builders.
  - `classification.classifier_option_backends` now owns category/MCP option wiring plus native jailbreak and PII dependency selection.
  - `classifier_construction.go` keeps construction orchestration, parallel option assembly, multimodal initialization, default embedding model selection, and heuristic initialization logging only, dropping from 398 to 135 lines.
  - Validation passed: `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_construction.go,src/semantic-router/pkg/classification/classifier_option_rules.go,src/semantic-router/pkg/classification/classifier_option_backends.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A KB classifier recheck found `category_kb_classifier.go` still mixed KB manifest loading, exemplar embedding preload, prototype-bank rebuilds, query classification, label/group matching, metric calculation, and result shaping.
  - `classification.category_kb_embeddings` now owns exemplar ref collection, parallel embedding preload, preload telemetry, and per-label prototype-bank rebuilds.
  - `classification.category_kb_scoring` now owns prototype label scoring, thresholded label matching, group scoring/matching, metric calculation, and `KBClassifyResult` shaping.
  - `category_kb_classifier.go` keeps the public KB classifier type, manifest loading, constructor, query embedding/classify orchestration, and label count only, dropping from 384 to 121 lines.
  - Validation passed: `go test ./pkg/classification -run 'TestKnowledgeBase|Test.*KB|Test.*Prototype'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/category_kb_classifier.go,src/semantic-router/pkg/classification/category_kb_embeddings.go,src/semantic-router/pkg/classification/category_kb_scoring.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A modality/category entropy recheck found `classifier_category_entropy.go` still mixed modality detection with category entropy routing, probability-distribution reasoning, fallback handling, and metrics.
  - `classification.classifier_modality` now owns modality result contracts plus classifier, keyword, and hybrid modality detection.
  - `classifier_category_entropy.go` keeps category-with-entropy orchestration, keyword/embedding category fallback, in-tree probability classification, reasoning-decision construction, fallback category selection, and entropy metrics only, dropping from 381 to 239 lines.
  - Validation passed: `go test ./pkg/classification -run 'Test.*Modality|Test.*Entropy|Test.*Category|Test.*Reasoning'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_category_entropy.go,src/semantic-router/pkg/classification/classifier_modality.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A request-time generic signal evaluator recheck found `classifier_signal_context.go` had regrown past a narrow context role by bundling readiness/orchestration with keyword, domain, fact-check, user-feedback, reask, context, complexity, and modality evaluator implementations.
  - `classification.classifier_signal_rule_evaluators` now owns keyword, domain, fact-check, user-feedback, reask, and context evaluator implementations.
  - `classification.classifier_signal_complexity` owns complexity result mutation and metric emission; `classification.classifier_signal_modality_eval` owns request-time modality signal result mutation and metric emission.
  - `classifier_signal_context.go` now keeps readiness, text selection, image-cache setup, dispatcher execution, and post-processing only, dropping from 391 to 102 lines.
  - Validation passed: `go test ./pkg/classification -run 'Test.*Signal|Test.*Modality|Test.*Complexity|Test.*Feedback|Test.*Domain|Test.*Context|Test.*Reask'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_signal_context.go,src/semantic-router/pkg/classification/classifier_signal_rule_evaluators.go,src/semantic-router/pkg/classification/classifier_signal_complexity.go,src/semantic-router/pkg/classification/classifier_signal_modality_eval.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A signal group recheck found `classifier_signal_groups.go` still mixed partition application, member resolution, winner/default selection, softmax math, projection trace construction, and embedding top-k output policy.
  - `classification.classifier_signal_group_resolution` now owns group winner/default selection and softmax scoring helpers.
  - `classification.classifier_signal_group_trace` now owns partition trace entry construction, raw winner scores, and margin calculation.
  - `classification.classifier_signal_output_policy` now owns matched-signal output limiting, including embedding top-k pruning.
  - `classifier_signal_groups.go` now keeps signal-group entrypoints and per-signal member resolution only, dropping from 371 to 68 lines.
  - Validation passed: `go test ./pkg/classification -run 'TestSignalGroup|TestApplySignalGroups|TestAnalyzeSoftmaxSignalGroupCentroids|Test.*Projection|Test.*Partition|Test.*TopK'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_signal_groups.go,src/semantic-router/pkg/classification/classifier_signal_group_resolution.go,src/semantic-router/pkg/classification/classifier_signal_group_trace.go,src/semantic-router/pkg/classification/classifier_signal_output_policy.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A projection recheck found `classifier_projections.go` still mixed projection execution, score dependency ordering, input accessor/value matching, output matching/confidence math, boundary distance calculation, and projection trace merging.
  - `classification.classifier_projection_order` now owns score dependency detection and topological ordering.
  - `classification.classifier_projection_inputs` now owns projection input accessors, typed value reads, match-set construction, and input matching.
  - `classification.classifier_projection_outputs` now owns output rule matching, confidence selection, and boundary-distance calculation.
  - `classification.classifier_projection_trace` now owns projection trace merging.
  - `classifier_projections.go` now keeps the projection execution entrypoint only, dropping from 366 to 43 lines.
  - Validation passed: `go test ./pkg/classification -run 'TestApplyProjections|TestProjection|Test.*Projection|Test.*Partition'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_projections.go,src/semantic-router/pkg/classification/classifier_projection_order.go,src/semantic-router/pkg/classification/classifier_projection_inputs.go,src/semantic-router/pkg/classification/classifier_projection_outputs.go,src/semantic-router/pkg/classification/classifier_projection_trace.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A model discovery recheck found `model_discovery.go` still mixed model path contracts, directory discovery, architecture detection, path validation, discovery-info response shaping, and unified classifier auto-initialization.
  - `classification.model_discovery_validation` now owns discovered model path validation and model-directory file checks.
  - `classification.model_discovery_info` now owns discovery-info response shaping and missing-model reporting.
  - `classification.unified_auto_init` now owns auto-discover plus LoRA/legacy unified classifier initialization.
  - `model_discovery.go` now keeps model path contracts, architecture models, public discovery entrypoints, and architecture detection only, dropping from 316 to 87 lines.
  - Validation passed: `go test ./pkg/classification -run 'TestAutoDiscoverModels|TestValidateModelPaths|TestModelPathsIsComplete|TestGetModelDiscoveryInfo|TestLoadLegacyUnifiedLabels|TestUnifiedClassifier'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/model_discovery.go,src/semantic-router/pkg/classification/model_discovery_validation.go,src/semantic-router/pkg/classification/model_discovery_info.go,src/semantic-router/pkg/classification/unified_auto_init.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A vLLM jailbreak recheck found `vllm_classifier.go` still mixed vLLM request construction, timeout/client handling, response-to-`ClassResult` mapping, parser selection, Qwen3Guard parsing, JSON parsing, simple keyword parsing, and category extraction.
  - `classification.vllm_jailbreak_parser` now owns parser selection, auto fallback, Qwen3Guard safety/severity/category parsing, JSON parsing, simple parsing, and category extraction.
  - `vllm_classifier.go` now keeps vLLM client setup and remote classification orchestration only, dropping from 325 to 116 lines.
  - `vllm_jailbreak_parser_test.go` covers parser-type selection, Qwen3Guard safety/severity/category parsing, JSON parsing, simple parsing, and auto fallback.
  - Validation passed: `go test ./pkg/classification -run 'TestVLLMJailbreakParser|TestNewVLLMJailbreakInference'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/vllm_classifier.go,src/semantic-router/pkg/classification/vllm_jailbreak_parser.go,src/semantic-router/pkg/classification/vllm_jailbreak_parser_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A contrastive preference current-source recheck found `contrastive_preference_classifier.go` still mixed public construction, concurrent rule embedding preload, query scoring, threshold/margin decisioning, example collection, and prototype-bank rebuilds.
  - `classification.contrastive_preference_embeddings` now owns concurrent rule-example embedding tasks, worker fanout, preload telemetry, and loaded embedding collection.
  - `classification.contrastive_preference_scoring` now owns detailed query scoring, deterministic score ordering, example collection, and prototype-bank rebuilds.
  - `contrastive_preference_classifier.go` keeps public types, construction/configuration, and `Classify` threshold/margin decisioning only, dropping from 356 to 126 lines.
  - Validation passed: `go test ./pkg/classification -run 'Test.*Preference|TestContrastivePreference'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/contrastive_preference_classifier.go,src/semantic-router/pkg/classification/contrastive_preference_embeddings.go,src/semantic-router/pkg/classification/contrastive_preference_scoring.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A preference wrapper current-source recheck found `preference_classifier.go` still mixed public construction, external LLM config/client/prompt setup, external response parsing, contrastive conversation-text extraction, and runtime dispatch.
  - `classification.preference_classifier_external` now owns external LLM preference construction, default prompt contract, route JSON shaping, LLM call orchestration, and tolerant route-output parsing.
  - `classification.preference_classifier_contrastive` now owns contrastive construction and conversation-message text extraction before delegating into the contrastive scorer.
  - `preference_classifier.go` now keeps public result/classifier types, top-level constructor dispatch, runtime dispatch, and readiness only, dropping from 266 to 65 lines.
  - `preference_classifier_test.go` now covers external parser compatibility for plain JSON, prose-wrapped JSON, single-quoted JSON, missing JSON, and empty route output.
  - Validation passed: `go test ./pkg/classification -run 'Test.*Preference|TestContrastivePreference'`; `go test ./pkg/classification ./pkg/services`.
  - A complexity classifier current-source recheck found `complexity_classifier_support.go` still mixed candidate embedding preload, query embedding loading, text/image rule scoring, signal fusion, difficulty labeling, and logging, while `complexity_classifier.go` still owned preload orchestration and prototype-bank rebuilds.
  - `classification.complexity_candidate_embeddings` now owns candidate task construction, worker fanout, embedding result collection, and text/image prototype-bank rebuilds.
  - `classification.complexity_query_embeddings` now owns text, multimodal-text, and optional request-image embedding loading, including the public image-classification path without an injected request cache.
  - `classification.complexity_rule_scoring` now owns per-rule text/image scoring, signal fusion, difficulty labeling, and result logging.
  - `complexity_classifier.go` keeps public types, construction, and request-level classify orchestration only, dropping from 222 to 148 lines; the deleted 330-line mixed support file is replaced by focused 250/60/116-line modules.
  - `complexity_classifier_test.go` covers public image classification without a request image cache.
  - Validation passed: `go test ./pkg/classification -run 'Test.*Complexity|TestEvaluateEmbeddingSignal_SharedCacheDedupsAcrossDivergentTargetDims|TestEvaluateEmbeddingAndComplexitySignalsShareRequestImageCache'`; `go test ./pkg/classification ./pkg/services`.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/complexity_classifier.go,src/semantic-router/pkg/classification/complexity_classifier_support.go,src/semantic-router/pkg/classification/complexity_candidate_embeddings.go,src/semantic-router/pkg/classification/complexity_query_embeddings.go,src/semantic-router/pkg/classification/complexity_rule_scoring.go,src/semantic-router/pkg/classification/complexity_classifier_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
  - A shared prototype-bank recheck found `prototype_scoring.go` still mixed prototype contracts, dedupe, similarity-matrix construction, clustering/medoid selection, representative copying, and runtime score aggregation used by embedding, contrastive preference, complexity, and KB classifiers.
  - `classification.prototype_bank` now owns prototype contracts, bank construction, uncompressed representatives, representative ordering, and safe representative copies.
  - `classification.prototype_clustering` now owns dedupe, similarity matrix construction, greedy clustering, tie-breaking, and medoid selection.
  - `prototype_scoring.go` now keeps score options and runtime query-to-bank aggregation only, dropping from 281 to 66 lines.
  - Validation passed: `go test ./pkg/classification -run 'TestPrototypeBank|TestEmbeddingClassifier|TestContrastivePreference|TestComplexityClassifier|TestKnowledgeBase'`; `go test ./pkg/classification ./pkg/services`.
- ASR008 progress on 2026-05-24:
  - Current-source recheck confirmed TD027 remains open, but threshold Pareto sweep/reporting was a cohesive optimizer feature still embedded in `optimizer/base.py`.
  - `fleet_sim.optimizer.threshold` now owns `ThresholdResult`, `threshold_pareto`, and `print_threshold_pareto`; `optimizer/__init__.py` exports that public seam directly.
  - `optimizer/base.py` keeps compatibility exports for direct `fleet_sim.optimizer.base` callers and dropped from 757 to 621 lines after formatter-compliant import layout.
  - Validation passed: `make vllm-sr-sim-test` with 304 tests passing.
  - TD028 remains open, but the controller-side canonical config builder slice is narrowed: base builder orchestration, backend discovery fan-in, and CRD spec-family translation now live in separate controller files.
  - `deploy/operator/controllers/canonical_config_builder.go` dropped from 314 to 58 lines; `canonical_config_backends.go` owns discovered backend fan-in and LoRA adapter conversion; `canonical_config_spec.go` owns config-family translation helpers.
  - Added focused controller coverage for strategy, reasoning-family defaults, tools, classifier modules, and complexity composer translation through `TestBuildCanonicalConfigAppliesOperatorSpecFamilies`.
  - Validation passed: `go test ./controllers -run 'TestBuildCanonicalConfigAppliesOperatorSpecFamilies|TestGenerateConfigYAMLIncludesLoRACatalogFromVLLMEndpoints'`; `go test ./...` from `deploy/operator`; `make generate-crd` with no CRD/Helm CRD drift.
  - Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="deploy/operator/controllers/canonical_config_builder.go,deploy/operator/controllers/canonical_config_backends.go,deploy/operator/controllers/canonical_config_spec.go,deploy/operator/controllers/canonical_config_spec_test.go"`.
  - Current-source recheck found TD030 partially stale for `App.tsx`: the top-level provider shell is already narrow, while `dashboard/frontend/src/app/AppRouter.tsx` had become the real route-shell hotspot with protected route registration and repeated layout wrappers.
  - `dashboard/frontend/src/app/AuthenticatedAppRoutes.tsx` now owns authenticated route registration and repeated shell wrapping, while `AppRouter.tsx` keeps setup/auth/public-route orchestration and dropped from 305 to 75 lines.
  - Validation passed: `npm run type-check`; `npm run lint`; `make dashboard-check`. A stricter exploratory `npm run lint -- --max-warnings=0` failed on 58 existing frontend warnings outside this slice before later warning cleanup.
  - Current-source recheck then confirmed the frontend unit-test gap remained current for `dashboard/frontend/src`, even though Playwright coverage exists under `dashboard/frontend/e2e`.
  - `dashboard/frontend/src/app/routeManifest.ts` now owns the serializable shell route and redirect manifest, `AuthenticatedAppRoutes.tsx` renders from it, and `routeManifest.test.ts` adds the first source-level Vitest unit test under `dashboard/frontend/src`.
  - `dashboard/frontend/package.json` adds `test:unit`, and `tools/make/dashboard.mk` wires `dashboard-test-frontend` into `make dashboard-check`.
  - Validation passed: `npm run test:unit`; `npm run type-check`; `npm run lint`; `make dashboard-check`.
  - `S005` dashboard page/container work has started with the overview page: `dashboardPageOverview.ts` now owns signal-breakdown and decision-preview row shaping, `DashboardPage.tsx` delegates to it and drops from 471 to 468 lines, and `dashboardPageOverview.test.ts` covers the pure row-model behavior.
  - Validation passed: `npm run test:unit`; `npm run type-check`; `npm run lint`; `make dashboard-check`.
  - A small config-page lint slice stabilized projection fallback arrays in `ConfigPageProjectionsSection.tsx`, reducing the Dashboard frontend lint backlog from 58 to 55 warnings.
  - Validation passed: `npm run lint`; `make dashboard-check`.
  - A frontend lint boundary cleanup then scoped `react-refresh/only-export-components` away from explicit support modules, allowed the `useAuth` context hook export, and updated `ColorBends` so prop synchronization no longer appears as a mount-effect dependency. Dashboard frontend lint now reports 0 warnings.
  - Validation passed: `npm run lint`; `npm run test:unit`; `npm run type-check`; `make dashboard-check`.
  - 2026-05-25 current-source recheck found TD004 stale: the Python CLI already has shared Docker/K8s deployment targets through `DeploymentBackend`, `K8sBackend`, and `vllm-sr serve/status/logs/stop/dashboard --target k8s`, with `test_deployment_backend.py` covering target routing and backend helpers.
  - TD004 is closed; the remaining Kind lifecycle, kubeconfig discovery, local image loading, and shared-suite topology work stays open under TD037.
  - Validation passed: `python -m pytest src/vllm-sr/tests/test_deployment_backend.py -q`; `make agent-ci-gate CHANGED_FILES="..."`; `make agent-scorecard`; `git diff --check`.
  - Scorecard movement recorded in `docs/agent/architecture-scorecard.md`; ASR008 remains open for TD027 residual work, TD028 residual work, TD030, and TD037.
- ASR009 progress on 2026-05-24:
  - Current-source recheck confirmed TD033 remains open, but the immediate router-side gap was narrower than rewriting native bindings: the Go classification runtime still inferred unified and LoRA classifier support from build tags and backend-compatible FFI stubs.
  - `classification.NativeBackendCapabilities` now exposes the selected native backend's support for unified batch classification, LoRA batch classification, batched embedding, multimodal embedding, modality routing, MLP selection, and explicit reset support.
  - Candle, ONNX, and non-CGO stub builds now publish their capability contract through build-tag-specific files, and unified/LoRA classifier entrypoints fail early when the selected backend does not advertise support.
  - Classifier stats now include the current native backend capability contract so runtime/debug surfaces no longer have to infer support from a build tag.
  - Validation passed: `go test ./pkg/classification -run 'TestCurrentNativeBackendCapabilitiesShape|TestUnifiedClassifierRejectsUnsupportedNativeCapabilities|TestUnifiedClassifierStatsIncludeNativeCapabilities|TestUnifiedClassifier_GetStats|TestUnifiedClassifier_ClassifyBatch' -v`.
  - Current-source recheck found a smaller TD033 release/build-hygiene slice in the native Rust binding: stale `cfg(feature = "metal")` branches for an undeclared feature in the MLP FFI.
  - `candle-binding/src/ffi/mlp.rs` now centralizes device selection in `device_from_type`; `device_type: 2` remains the existing CPU fallback until a real Metal feature and lockfile policy are added.
  - A direct Metal feature attempt failed validation by pulling Candle Metal dependency resolution into the current lockfile and conflicting on `once_cell`, so this slice documents the unsupported state instead of adding a broken feature.
  - Validation passed: `cargo fmt --check`; `cargo check --release --no-default-features`; `make test-binding-minimal`.
  - Feature validation attempted but did not pass locally: `make test-binding-lora` failed because the local environment lacks the LoRA/PII/Gemma model assets expected by those tests.
  - A 2026-05-25 follow-up warning/safety-hygiene pass removed unused imports from `candle-binding/src/ffi/generative_classifier.rs`, updated `qwen3_guard.rs` top-p sampling from deprecated `rand::thread_rng().gen()` to the current `rand::rng().random()` API, and made the generative raw-pointer exports explicit `unsafe extern "C"` functions with documented pointer ownership requirements without changing the C ABI.
  - The same slice split touched legacy hotspots instead of waiving them: Qwen3Guard FFI exports moved to `candle-binding/src/ffi/generative_guard.rs`, while Qwen3Guard prefix-cache generation, weight loading, and sampling helpers moved to `qwen3_guard_generation.rs`, `qwen3_guard_loading.rs`, and `qwen3_guard_sampling.rs`. The changed native files now have only structure warnings, not structure errors.
  - Direct default `cargo check --manifest-path candle-binding/Cargo.toml --lib` still fails locally before crate checking because the default-feature CUDA build requires `nvcc`; the validated local CPU path is `--no-default-features` / `make rust-ci`.
  - Validation passed: `cargo fmt --manifest-path candle-binding/Cargo.toml`; changed-file Rust lint through `tools/agent/scripts/agent_gate.py run-rust-lint`; `cargo check --release --no-default-features --lib` from `candle-binding`; `make rust-ci`; `make test-binding-minimal`; `make agent-ci-gate CHANGED_FILES="candle-binding/src/ffi/generative_classifier.rs,candle-binding/src/ffi/generative_guard.rs,candle-binding/src/ffi/mod.rs,candle-binding/src/model_architectures/generative/qwen3_guard.rs,candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_generation.rs,candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_loading.rs,candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_sampling.rs,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`.
  - Feature validation attempted but did not pass locally: `make test-binding-lora` failed in existing restricted-model paths, including missing `../models/mom-pii-classifier` / LoRA classifier assets and Gemma fallback dimension expectations.
  - Current-source recheck found a release-version contract drift: `.github/workflows/release.yml` validates both the Python package and Candle crate against `v*` tags, but `src/vllm-sr/scripts/release.sh` only updated `src/vllm-sr/pyproject.toml`.
  - `src/vllm-sr/scripts/release.sh` now updates the Python package plus `candle-binding/Cargo.toml` and `candle-binding/Cargo.lock` together for release and next-development commits; Candle is aligned to the current `0.3.0` package version.
  - Validation passed: `bash -n src/vllm-sr/scripts/release.sh`; Python version-contract parse check; `cargo metadata --no-deps --format-version 1` for `candle-binding`; `make vllm-sr-test`; `make test-binding-minimal`.
  - Changed-set validation initially exposed a repo-harness false positive rather than a release-script failure: `make shellcheck` scanned the local untracked `.venv-fresh` virtualenv. `tools/make/linter.mk` now excludes `.venv-*` directories so local agent/Python scratch environments do not poison changed-set gates.
  - Validation passed: `make shellcheck`; `make agent-ci-gate CHANGED_FILES="tools/make/linter.mk,src/vllm-sr/scripts/release.sh,candle-binding/Cargo.toml,candle-binding/Cargo.lock,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`.
  - A follow-up release-contract slice now removes the remaining local/CI parser
    split for the surfaces that are already mechanically checkable:
    `tools/release/check_version_contract.py` validates Python/Candle/Candle
    lockfile version alignment, Helm tag packaging override semantics, and
    Docker release image coverage in unified release notes.
  - `make release-check`, `src/vllm-sr/scripts/release.sh`, and
    `.github/workflows/release.yml` all call that shared checker, so local
    release prep and tag validation fail on the same contract drift.
  - Validation passed: `python3 tools/release/check_version_contract.py`;
    `python3 tools/release/check_version_contract.py --version 0.3.0`;
    `make release-check RELEASE_VERSION=0.3.0`;
    `bash -n src/vllm-sr/scripts/release.sh`.
  - A follow-up Operator release-image slice found that `operator-ci.yml`
    already publishes `semantic-router/operator` and
    `semantic-router/operator-bundle` on `v*` tags, but the shared checker and
    unified release body did not include those images.
  - `tools/release/check_version_contract.py` now parses tagged Operator images
    from `operator-ci.yml`, and `.github/workflows/release.yml` now lists the
    Operator and bundle image pull commands plus tag rows in the unified GitHub
    Release body.
  - Validation passed: `python3 tools/release/check_version_contract.py`;
    `python3 tools/release/check_version_contract.py --version 0.3.0`;
    `python3 -m py_compile tools/release/check_version_contract.py`;
    `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml tools/release/check_version_contract.py`;
    `make release-check RELEASE_VERSION=0.3.0`.
  - A follow-up upgrade-runbook slice found that the shared checker knew the
    full versioned release image set, but the upgrade/rollback runbook only
    documented a partial Docker upgrade set.
  - `tools/release/check_version_contract.py` now requires
    `website/docs/installation/upgrade-rollback.md` to mention every release
    image parsed from the Docker and Operator release workflows, and the runbook
    now lists `dashboard`, `extproc`, `extproc-rocm`, `llm-katan`, `operator`,
    `operator-bundle`, `vllm-sr`, and `vllm-sr-rocm`.
  - Validation passed: `python3 tools/release/check_version_contract.py`;
    `python3 tools/release/check_version_contract.py --version 0.3.0`;
    `python3 -m py_compile tools/release/check_version_contract.py`;
    `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml tools/release/check_version_contract.py`;
    `make release-check RELEASE_VERSION=0.3.0`.
  - A follow-up simulator release-workflow slice found that
    `pypi-publish-vllm-sr-sim.yml` already owns the independent
    `vllm-sr-sim-v*` tag stream, but the shared checker and upgrade runbook did
    not enforce that package/workflow contract.
  - `tools/release/check_version_contract.py` now validates the simulator
    workflow tag trigger/extraction, package/tag version guard, PyPI publish
    command, install snippet, unified release-note mention, and upgrade runbook
    coverage.
  - `website/docs/installation/upgrade-rollback.md` now documents the pinned
    `vllm-sr-sim` package upgrade command and the separate
    `vllm-sr-sim-v<version>` promotion flow.
  - Validation passed: `python3 tools/release/check_version_contract.py`;
    `python3 tools/release/check_version_contract.py --version 0.3.0`;
    `python3 -m py_compile tools/release/check_version_contract.py`;
    `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml tools/release/check_version_contract.py`;
    `make release-check RELEASE_VERSION=0.3.0`.
  - A follow-up native backend documentation slice found that the router-side
    capability seam was already present, but user-facing installation docs did
    not explain Candle, ONNX, or non-CGO capability boundaries or lifecycle
    expectations.
  - `website/docs/installation/native-backends.md` now documents backend
    selection, the Candle/ONNX/stub capability table, early unsupported-feature
    failure behavior, and the current no-explicit-reset lifecycle contract.
  - `website/sidebars.ts` links that guide from the Installation section.
  - Validation passed: `make docs-lint`; `cd website && npm run build:en`;
    `make agent-ci-gate CHANGED_FILES="..."`; `make agent-scorecard`;
    `git diff --check`.
  - Scorecard movement recorded in `docs/agent/architecture-scorecard.md`; ASR009 remains open for native lifecycle/reset ownership, deeper ONNX parity, and real upgrade/rollback fixture validation.
  - A follow-up upgrade-runbook fixture-check slice found that the shared release
    checker still accepted runbook image-name mentions without full tagged image
    refs or copyable pinned command examples.
  - `tools/release/check_version_contract.py` now requires every Docker and
    Operator release image to appear in the runbook as a full
    `ghcr.io/vllm-project/semantic-router/<image>:v<version>` reference, and it
    validates pinned Helm chart lookup/upgrade, safe value-merge, Helm rollback,
    Kubernetes rollout undo, Docker digest lookup, release Make targets, Helm
    values image pins, Python CLI upgrade, and independent `vllm-sr-sim` upgrade
    fixture markers.
  - Validation passed: `python3 tools/release/check_version_contract.py`;
    `python3 tools/release/check_version_contract.py --version 0.3.0`;
    `python3 -m py_compile tools/release/check_version_contract.py`;
    `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml tools/release/check_version_contract.py`;
    `make release-check RELEASE_VERSION=0.3.0`.
  - Scorecard movement recorded in `docs/agent/architecture-scorecard.md`;
    ASR009 remains open for native lifecycle/reset ownership, deeper ONNX parity,
    and live environment upgrade/rollback execution.
  - A follow-up Candle crate workflow ownership slice found that
    `.github/workflows/publish-crate.yml` already owns the tag trigger, crate
    version guard, CPU-only smoke/check/build, publish dry-run, crates.io
    publish, and GitHub Release native artifact attachment, but the shared
    release checker did not enforce those markers.
  - `tools/release/check_version_contract.py` now validates the Candle crate
    workflow plus Rust crate release-note snippet as part of `make
    release-check`, `make release`, and release tag validation.
  - `tools/release/release_contract_markers.py` now owns static marker lists, so
    the shared checker stays below the structure-rule warning threshold instead
    of becoming a new release hotspot.
  - Validation passed: `python3 tools/release/check_version_contract.py`;
    `python3 tools/release/check_version_contract.py --version 0.3.0`;
    `python3 -m py_compile tools/release/check_version_contract.py tools/release/release_contract_markers.py`;
    `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml tools/release/check_version_contract.py tools/release/release_contract_markers.py`;
    `make release-check RELEASE_VERSION=0.3.0`;
    `make agent-ci-gate CHANGED_FILES="tools/release/check_version_contract.py,tools/release/release_contract_markers.py,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`.
  - Scorecard movement recorded in `docs/agent/architecture-scorecard.md`;
    ASR009 remains open for native lifecycle/reset ownership, deeper ONNX parity,
    and live environment upgrade/rollback execution.
  - A follow-up Helm values schema slice found that the chart had template-time
    safety guards but no `values.schema.json` for the public deployment
    controls involved in those guards.
  - `deploy/helm/semantic-router/values.schema.json` now validates key router
    and dashboard replica, autoscaling, persistence, and safety-guard value
    types while keeping subchart values extensible.
  - Helm docs now describe schema validation as the type contract and template
    guards as the cross-field production safety contract.
  - Validation passed: `helm lint deploy/helm/semantic-router`;
    `helm template sr deploy/helm/semantic-router -f
    deploy/helm/semantic-router/values-prod.yaml`; invalid
    `safetyGuards.rejectMultiReplicaLocalSelectorState=maybe` and
    `replicaCount=oops` renders fail schema validation before templates run.
  - A follow-up Helm harness-routing slice found that the same Helm chart/schema
    change was still reported as documentation-only by `make agent-report`.
    `tools/agent/task-matrix.yaml` now owns a non-doc `helm-chart` rule for
    `deploy/helm/**`, `tools/agent/repo-manifest.yaml` and
    `tools/agent/skill-registry.yaml` map Helm into the Kubernetes deployment
    profile surfaces, and `docs/agent/change-surfaces.md` documents the routing.
  - `tools/make/helm.mk` now supports `HELM_REPO_UPDATE=false`, so local agent
    Helm feature validation can use the locked chart dependencies while default
    CI-style validation still refreshes chart repositories. The Helm validation
    render output now defaults under `/tmp/semantic-router-helm/` so a passing
    feature gate does not leave repo-local generated YAML that later global lint
    scans treat as source.
  - Validation passed: `make agent-report ENV=cpu CHANGED_FILES="..."` now
    reports the `helm-chart` and `make-and-ci` rules, `make helm-lint` fast
    gate, `make helm-ci-validate HELM_REPO_UPDATE=false` feature gate, required
    local smoke, affected `kubernetes` local E2E profile, and CI E2E mode
    `all`; `make helm-ci-validate HELM_REPO_UPDATE=false
    HELM_NAMESPACE=test-namespace` passed.
  - Default `make helm-ci-validate HELM_NAMESPACE=test-namespace` was attempted
    first and failed while refreshing the Prometheus chart repository with
    `context deadline exceeded`, before dependency build or template rendering.
    `make agent-e2e-affected CHANGED_FILES="..."` reached the Kubernetes
    profile and failed during Kind cluster discovery because the local
    Docker/Kind environment is unavailable.
  - A follow-up Helm safety-regression gate slice promoted the manual schema and
    local-state guard checks into `make helm-safety-validate`: production render
    succeeds, invalid selector guard and replica-count types fail schema
    validation, RL-driven and GMTRouter local-state multi-replica renders fail,
    and the explicit unsafe opt-out still renders.
  - A follow-up HPA bounds slice added template-time rejection for
    `autoscaling.minReplicas > autoscaling.maxReplicas` and folded that negative
    case into `make helm-safety-validate`.
  - `tools/agent/task-matrix.yaml` now runs `make helm-safety-validate
    HELM_REPO_UPDATE=false` as part of the Helm feature gate so those regressions
    are replayable by future agents.
  - Validation passed: `make helm-safety-validate HELM_REPO_UPDATE=false
    HELM_NAMESPACE=test-namespace`.
  - Scorecard movement recorded in `docs/agent/architecture-scorecard.md`;
    ASR009 remains open for native lifecycle/reset ownership, deeper ONNX parity,
    and live environment upgrade/rollback execution.
- ASR010 reconciliation on 2026-05-24:
  - All scorecard dimensions are now at or above 80 after the TD034 cookie-backed reload and GMTRouter local-state hardening slices.
  - Current score floor after the follow-up selector config-runtime, runtime
    memory-store boundary, API vector/file/selector registry-boundary,
    Operator release-image, upgrade-runbook image coverage, simulator
    release-workflow, native backend docs, upgrade-runbook fixture checks,
    Candle crate workflow ownership, API startup config resolver,
    classification refresh global-config, RL-driven looper runtime-selector,
    ExtProc startup config-precedence, ExtProc request-time plugin
    config-precedence, ExtProc config-source watcher precedence, and ExtProc
    initial router config precedence slices:
    maintainability is 97; API stability and performance are 91; memory and
    compute management is 91;
    release/upgrade
    completeness is 90.
  - Validation evidence for this reconciliation: `make agent-scorecard`; `make test-semantic-router`; changed-set gates for the Dashboard auth reload, GMTRouter local-state, and decision-scoped selector config-runtime slices.
  - Local smoke and manual E2E remain blocked by environment, not by the current code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.
- ASR006 follow-up on 2026-05-25:
  - Dashboard auth session robustness moved further server-side:
    backend token intake is now bounded and normalized across bearer,
    HttpOnly cookie, and query-token compatibility transports before JWT
    parsing; newly issued tokens carry persisted session ids; logout revokes
    the current server-side session before clearing the browser cookie; cookie
    issue/clear responses include explicit expiry timestamps; expired/revoked
    session rows are indexed and pruned after the retention window; and focused
    session tests sit outside `handlers_test.go` and `store.go` hotspots.
  - Validation evidence: `go test ./auth` from `dashboard/backend`.
  - A follow-up Helm deployment slice makes the same local-state boundary
    explicit for Kubernetes: `dashboard.persistence.enabled` creates or mounts a
    dashboard-local PVC, the dashboard deployment wires
    `DASHBOARD_AUTH_DB_PATH` and `DASHBOARD_WORKFLOW_DB_PATH` into that mounted
    path, `values-prod.yaml` enables the PVC, and template rendering fails when
    `dashboard.replicaCount > 1` because the current SQLite auth/session store
    is restart-safe local state rather than a shared HA store.
  - Validation evidence: `helm dependency build deploy/helm/semantic-router`;
    `helm template sr deploy/helm/semantic-router --set dashboard.enabled=true --set dashboard.persistence.enabled=true`;
    `helm template sr deploy/helm/semantic-router --set dashboard.enabled=true --set dashboard.replicaCount=2`
    failed with the expected guard.
  - A follow-up Helm selector-state slice makes the router-side local learning
    state boundary explicit for Kubernetes: multi-replica router renders now
    fail when active decisions use `algorithm.type: elo`, `rl_driven`, or
    `gmtrouter` unless
    `safetyGuards.rejectMultiReplicaLocalSelectorState=false` is set after
    accepting replica-local learning divergence.
  - Validation evidence: `helm lint deploy/helm/semantic-router`; `helm
    template sr deploy/helm/semantic-router -f
    deploy/helm/semantic-router/values-prod.yaml`; `helm template sr
    deploy/helm/semantic-router --set autoscaling.enabled=false --set
    replicaCount=2 --set 'config.routing.decisions[0].algorithm.type=rl_driven'`
    failed with the expected guard; the same render passed when
    `safetyGuards.rejectMultiReplicaLocalSelectorState=false`; HPA with
    `autoscaling.maxReplicas=2` and `algorithm.type=gmtrouter` also failed with
    the guard.
  - Docker-backed `agent-smoke-local` remains blocked by the local Docker daemon
    being unavailable at `/Users/bitliu/.docker/run/docker.sock`.

## 90+ Convergence Loop

The next loop should not chase stale debt text. Each slice must first re-read the current code and either close stale debt, narrow active debt, or add a new indexed entry only when the source still diverges from the intended architecture.

Priority order toward 90+:

1. Release/upgrade floor: static runbook fixture checks and Candle crate workflow ownership now live in the shared release checker; next reduce the remaining gap through live upgrade/rollback execution once Docker/cluster access is available and backend lifecycle/reset ownership can be tested.
2. Runtime shared-state floor: decide and implement production shared-store policy for selector learning state and Dashboard auth/session state instead of relying on local files, SQLite, or compatibility globals.
3. Control-plane maintainability: continue reducing Dashboard `ChatComponent`, config/editor stores, Operator CRD/webhook hotspots, Fleet Sim optimizer exports, and Python CLI workflow ownership with extraction-first PRs.
4. Contract/API consistency: continue TD015/TD039 by moving remaining Dashboard config/setup record bags, operator translation shapes, and CLI runtime assumptions behind contract-owned seams.
5. Router pipeline: continue TD020 by splitting remaining request-time signal orchestration without changing public routing behavior.
6. E2E and environment debt: unblock Docker/local smoke, then run `make agent-feature-gate ENV=cpu CHANGED_FILES="..."` and affected manual E2E profiles for behavior-visible runtime, CLI, dashboard, and operator changes.

## Decision Log

- 2026-05-24: use a scored architecture ratchet rather than a one-time audit report so follow-up PRs can show measurable movement.
- 2026-05-24: keep the first delivery documentation-only and avoid runtime/API changes in the same PR that introduces the governance mechanism.
- 2026-05-24: group implementation work into five PR-chain clusters: Contract/API, Runtime/State, Router Pipeline, Control Planes, and Release/Docs/Native.
- 2026-05-24: score improvements require passed validation; prose-only improvements do not move scores.
- 2026-05-24: PR0 validation closed with `make agent-validate`, `make agent-scorecard`, and changed-set `make agent-ci-gate`; no runtime or public API behavior changed.
- 2026-05-24: TD036 should be closed by the implemented authoring-sugar contract, not by adding tree metadata. Current source and `pl-0012` show that lossless `DECISION_TREE` runtime-config round-trip was intentionally rejected as excess contract complexity.
- 2026-05-24: TD015 should be ratcheted by removing concrete weak-typing seams one at a time. The compiler/validator field-bag path was small enough to narrow now; decompile/export maps and Dashboard setup/config adapters stay open rather than being papered over.
- 2026-05-24: TD015 decompiler work should prefer the existing AST `Value` vocabulary over adding another generic map abstraction. Structure/conversation features and tools dynamic retrieval can use `Value` formatting now; raw map compatibility remains appropriate only for unknown plugin payload fields and legacy emit/export paths until those surfaces get their own typed seams.
- 2026-05-24: TD039 should be ratcheted through small contract adapters first. The Dashboard tools DB path reader was a narrow live cross-boundary import and could move behind a backend-local adapter now; broader Dashboard handlers, Operator canonical translation, CLI ownership, and runtime globals remain separate cuts.
- 2026-05-24: keep Dashboard backend handler code transport- and product-specific when narrowing TD039. OpenClaw can decide URL precedence and host normalization, but router listener shape belongs in `dashboard/backend/routercontract` so the handler does not become another YAML contract owner.
- 2026-05-24: TD034 should be ratcheted by turning existing durable stores into explicit recovery contracts before broad state-system rewrites. ML pipeline pending/running recovery was small enough to harden now; router-side state and model-selection learning remain separate cuts.
- 2026-05-24: TD034 vector-store metadata debt was narrower than the original entry suggested. Router runtime already supports Postgres-backed metadata recovery; the live gap was CLI local runtime config injection and stale docs, so this loop fixed that seam instead of adding another metadata store.
- 2026-05-24: TD034 router replay durability should preserve compatibility for configs with no storage block, but it should not ignore durable backend blocks when `store_backend` is omitted. Backend inference from present config blocks is a safe middle ground between breaking raw legacy configs and continuing an unnecessary memory fallback.
- 2026-05-24: TD034 Response API state should no longer be treated as open response-store durability debt. Current source defaults to Redis, CLI local runtime injects Redis, and restart-recovery E2E exists; the live bug was stale docs plus Redis integration tests leaking into default package tests without an integration build tag.
- 2026-05-24: TD031 should be treated as a compatibility-global retirement loop, not as proof that the common runtime path lacks a registry. The next safe cut was to publish model-selection state into `routerruntime.Registry` and make API mutation/read handlers consume it, while leaving `selection.GlobalRegistry` as a compatibility fallback until selector construction itself moves behind a non-global owner.
- 2026-05-24: TD023 and TD029 should not be targeted by ASR007 because their debt files are already closed with resolution evidence. The active router-pipeline ratchet is TD020 classification only unless a new current-source extproc regression is found.
- 2026-05-24: TD020 should be narrowed by moving mapping/bootstrap ownership into `pkg/classification` before attempting broad classifier-file surgery. The service layer can compose and refresh classifiers, but it should not own classification mapping policy.
- 2026-05-24: TD027 should be ratcheted by extracting cohesive optimizer feature modules while preserving import stability. Threshold Pareto is a narrow first slice because it has its own dataclass, sweep, marking, and report helper, and existing public imports can be kept through compatibility exports.
- 2026-05-24: TD028 should be ratcheted by relieving controller translation before touching CRD schema or webhook validation. `canonical_config_builder.go` was safe to split into base builder, backend discovery, and config-family translation seams without changing generated CRDs or public Kubernetes API.
- 2026-05-24: TD030 should be interpreted against the current frontend app shell rather than the older `App.tsx` text. `App.tsx` is already narrow; the safe next route-shell cut was to move authenticated route registration and repeated shell wrapping out of `AppRouter.tsx` while leaving setup/auth/public-route behavior in place.
- 2026-05-24: frontend test debt should be judged against the current source tree. The repo had Playwright and prompt tests, but no `dashboard/frontend/src` unit tests; adding a serializable route manifest plus Vitest coverage gives future route/page extractions a small, fast source-level test seam.
- 2026-05-24: TD033 release-version work should make the local release helper obey the same surfaces as the tag workflow. The workflow already rejects Python/Candle version drift, so `make release` must update both surfaces instead of relying on a manual crate bump.
- 2026-05-24: TD033 release-contract validation should use one repo-level parser
  instead of workflow-local grep/sed snippets. The shared checker now owns the
  mechanically verifiable release contract for Python/Candle/Candle-lockfile
  versions, Helm tag packaging semantics, and Docker image coverage in release
  notes; remaining artifact ownership work should add surfaces to this checker
  rather than duplicating parser logic in individual workflows.
- 2026-05-24: TD030 page hotspot work should start with behavior-neutral pure support models. Moving overview row shaping behind a tested sibling helper gives coverage and a page-local seam before larger JSX-section extraction.
- 2026-05-24: the frontend lint backlog should be reduced through current-source fixes, not waived. The `ConfigPageProjectionsSection.tsx` warnings were real unstable fallback dependencies and could be fixed without altering UI behavior.
- 2026-05-24: Fast Refresh linting should follow current module ownership. Support modules are allowed to export constants, hooks, helpers, and small display fragments while they reduce page hotspots; component-boundary linting remains active elsewhere.
- 2026-05-24: TD033 should be ratcheted through explicit capability contracts before attempting broad native-binding rewrites. The first safe cut is a router-side backend capability seam with early unsupported-feature failures; lifecycle ownership and ONNX parity stay separate because they require backend-specific validation.
- 2026-05-24: TD033 native build hygiene should not introduce a Metal feature until the Candle Metal dependency set and lockfile policy are intentionally updated. The current MLP `device_type: 2` behavior was already CPU fallback, so removing dead Metal cfg branches preserves behavior and makes the unsupported state explicit.
- 2026-05-24: repo harness linters should ignore local `.venv-*` scratch environments. Changed-set gates should validate repository shell scripts, not third-party completion scripts installed into local agent/Python virtualenvs.
- 2026-05-24: Dashboard auth should move through compatibility-preserving server-owned session slices. Setting and clearing an HttpOnly cookie is safe now because the existing token response and frontend protected transport shims remain compatible; cookie-backed reload can be enabled through `/api/auth/me` without breaking the token response, while production session-store policy remains separate follow-up work.
- 2026-05-24: GMTRouter local-state hardening is worthwhile even though it does not close shared-store parity. Atomic local writes and corrupted-state backups reduce restart risk now; production multi-replica learning state still needs a shared selector-store decision.
- 2026-05-24: Decision-scoped selector persistence must be judged against the current config-to-runtime path, not only selector internals. RLDriven and GMTRouter local storage support was present, but it did not help per-decision algorithms until `buildModelSelectionConfig` forwarded those fields into runtime selector construction.

## Follow-up Debt / ADR Links

- [../architecture-scorecard.md](../architecture-scorecard.md)
- [../tech-debt/td-015-weakly-typed-config-and-dsl-contracts.md](../tech-debt/td-015-weakly-typed-config-and-dsl-contracts.md)
- [../tech-debt/td-020-classification-subsystem-boundary-collapse.md](../tech-debt/td-020-classification-subsystem-boundary-collapse.md)
- [../tech-debt/td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md](../tech-debt/td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md)
- [../tech-debt/td-028-operator-config-contract-boundary-collapse.md](../tech-debt/td-028-operator-config-contract-boundary-collapse.md)
- [../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md](../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
- [../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [../tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md](../tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md)
- [../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md](../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- [../tech-debt/td-036-decision-tree-authoring-roundtrip-gap.md](../tech-debt/td-036-decision-tree-authoring-roundtrip-gap.md)
- [../tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md](../tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md)
- [../tech-debt/td-038-custom-chat-completions-structs.md](../tech-debt/td-038-custom-chat-completions-structs.md)
- [../tech-debt/td-039-control-plane-contract-ownership-collapse.md](../tech-debt/td-039-control-plane-contract-ownership-collapse.md)
- [../tech-debt/td-040-reconcile-path-skips-family-validators.md](../tech-debt/td-040-reconcile-path-skips-family-validators.md)
