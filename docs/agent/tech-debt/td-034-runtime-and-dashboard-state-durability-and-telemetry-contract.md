# TD034: Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract

## Status

Open

## Scope

`src/semantic-router/pkg/{config,responsestore,routerreplay,vectorstore,startupstatus,selection}/**`, `dashboard/backend/{auth,evaluation,mlpipeline,handlers,workflowstore}/**`, `dashboard/frontend/src/{hooks,components,utils}/**`, `src/vllm-sr/cli/**`, and adjacent docs or defaults that present stateful runtime features

## Summary

The repository now exposes several user-visible stateful product surfaces beyond request-time routing, but the persistence and recovery contract for those surfaces is still implicit and fragmented. Router defaults still present memory-backed behavior for core runtime facilities such as semantic cache and some explicitly local-dev state paths, while dashboard defaults surface in-memory, SQLite, JSON-file, workspace-local, and browser-local expectations to operators. Dashboard features then split their own state across SQLite databases, JSON files under workspace-local directories, in-memory job maps and event channels, log-derived status collectors, and browser localStorage for chat history and auth propagation. Model-selection learning state has several optional persistence paths, including Elo file storage and RL-driven `storage_path` persistence, but those paths are still local-file oriented and not yet one shared durable store contract. The CLI mounts `.vllm-sr` and `dashboard-data` into the local container, but the repo still lacks one cross-stack contract that says which state is ephemeral, which must survive restart, which belongs in shared storage, and which telemetry or progress signals are durable enough to drive recovery.

Current-source recheck on 2026-05-24 found the ML pipeline portion narrower
than this original summary: ML jobs and typed progress events are already
server-owned in `dashboard/backend/workflowstore`. One real recovery gap was
closed in this loop: dashboard restart recovery now fails both `pending` and
`running` ML jobs and appends a durable `recovered_after_restart` progress event
instead of leaving a pre-run pending job stuck forever. The broader debt remains
open for router-side response/replay/vector metadata durability, dashboard
browser-local state, model-selection shared-store parity, and a shared state taxonomy
that spans router, dashboard, and CLI.

The same recheck found the vector-store metadata portion narrower than the
original summary. Router runtime already supports a Postgres-backed metadata
registry and loads store/file metadata on startup. The live local-runtime gap was
that the Python CLI detected `vector_store.metadata_store: postgres` and
provisioned Postgres but did not inject the required `metadata_postgres`
connection block. That seam is now covered by CLI defaults and tests. The debt
stays open because memory-backed vector metadata is still allowed for ephemeral
local use, and the broader state taxonomy still has unresolved router,
dashboard, and model-selection surfaces.

A later 2026-05-24 source recheck found a similar but narrower router replay
gap. Canonical defaults and docs now prefer durable Postgres replay storage, but
the extproc runtime still fell back to memory whenever `store_backend` was
omitted. Runtime replay backend selection now considers the full
`RouterReplayConfig`: explicit `store_backend` wins, durable backend config
blocks imply their backend, and only an otherwise empty config keeps the
compatibility memory fallback. At that point, the debt still stayed open for
Response API durability verification, real restart E2E, model-selection
learning state, dashboard/browser-local state, and shared durable projections.

A later 2026-05-24 recheck found the Response API portion stale. Canonical
defaults, runtime config comments, CLI local defaults, and the state taxonomy
now treat Redis as the default durable Response API store, with `memory`
documented as local-development only. The Redis profile also registers
`response-api-restart-recovery` E2E coverage. This loop fixed the live
remaining gaps around that surface: the current troubleshooting page no longer
claims Redis response storage is unimplemented, and the Redis integration tests
now require the `integration` build tag so plain package tests do not fail on
developer machines without Redis. The debt stays open for model-selection
learning-state shared-store parity, dashboard/browser-local state, derived
config projections, and full local smoke/E2E reruns once Docker is available.

A follow-up 2026-05-24 source recheck found the model-selection learning-state
portion narrower than the original text. Elo ratings already have file-backed
storage when `storage_path` is configured, and RL-driven selection already had
a storage seam, but that seam only saved global/category preferences. This loop
extends RL-driven persistence so user-specific preferences and session context
survive restart through the existing `storage_path` format. The debt stays open
because this is still a local-file-compatible store rather than a shared
production durability contract, GMTRouter still persists a local JSON graph,
and selector globals remain as compatibility fallback paths. A later local-state
hardening slice narrowed GMTRouter's current file seam: saves are now atomic,
nested storage directories are created, corrupted JSON is backed up, and loaded
state is normalized before use. Shared-store parity is still open.

A later 2026-05-24 source recheck found a config/runtime mismatch in that same
model-selection slice. The RLDriven and GMTRouter tutorials documented
`storage_path` and tuning fields, and the selectors already had local storage
support, but `buildModelSelectionConfig` only forwarded decision-scoped Elo and
RouterDC config into the selector factory. The router config schema, Python CLI
schema, and Dashboard DSL field schema now expose the documented RLDriven and
GMTRouter fields, the extproc selector config builder forwards decision-scoped
RLDriven/GMTRouter settings into runtime construction, and reference config plus
selection fragments exercise the public fields. The debt still stays open
because this closes config-to-runtime reachability, not the shared production
store contract. Changed-set validation passed with `make agent-ci-gate
CHANGED_FILES="..."`, `make agent-scorecard`, and `git diff --check`; feature
integration remains blocked by the local Docker daemon being unreachable at
`/Users/bitliu/.docker/run/docker.sock`.

A later 2026-05-24 source recheck found a router runtime status inconsistency
rather than a missing store implementation. `/startup-status` already resolved
Redis-backed startup state when `startup_status.store_backend: redis` was
configured, but `/ready` still read only the file-backed status path. That could
make readiness report `503` in deployments where the startup state was correctly
published to Redis. This loop makes `/ready` use the same startup-state resolver
as `/startup-status`, adds unit coverage through a private shared-state loader
seam, and corrects the runtime warning to reference the current
`startup_status.store_backend` config key. The debt stays open for the broader
state surfaces that still rely on browser storage, local files, derived
projections, or compatibility globals.

A later 2026-05-24 source recheck found the memory-store runtime-boundary debt
narrower than the original broad global-state wording. ExtProc startup already
publishes `OpenAIRouter.MemoryStore` into `routerruntime.Registry`, and memory
API handlers resolve request-time stores through that registry. The remaining
live gap was that `apiserver.InitWithRuntime` could still adopt
`memory.GetGlobalMemoryStore()` during startup if the registry had not published
a store yet, binding the API server to process-wide state from another runtime.
That path now stays runtime-owned: when a registry is provided, startup reads
only `runtimeRegistry.MemoryStore()`, while the legacy nil-registry `Init` path
keeps the global fallback for compatibility. The debt remains open for broader
compatibility globals, shared selector-store policy, production session-store
policy, and durable config projections.

A follow-up 2026-05-24 source recheck found the browser-local dashboard state
scope narrower than the original entry. `useConversationStorage` already marks
playground chat history as browser-local UX state, while OpenClaw room APIs own
supported multi-operator chat history. The first live browser-local gap in this
slice was `usePlaygroundQueue`: queued playground tasks were restored from
localStorage without shape validation or retention bounds. A later recheck found
the matching conversation-history restore path still accepted malformed or
overlarge localStorage records even though saves were capped. This loop adds
pure queue and conversation normalization/pruning helpers, wires them into load
and persist paths, and covers malformed restore data plus conversation/task caps
with frontend unit tests. A final browser-storage hardening slice in this loop
adds auth-token normalization at the frontend storage boundary so malformed or
oversized local values are removed before they can be reused in Authorization
headers, cookies, or query-token transports. A server session-cookie slice then
adds HttpOnly `vsr_session` issuance on login/bootstrap plus a logout endpoint
that clears the server cookie while preserving the existing token response
contract for compatibility. A follow-up reload slice makes the dashboard probe
`/api/auth/me` with same-origin credentials on provider startup, so a valid
HttpOnly cookie can restore the current user even when no local token is
readable. The debt stays open because production session-store policy,
server-owned chat/session decisions, shared selector stores, and durable config
projections still need separate product-level decisions.

A 2026-05-25 source recheck found a narrower CLI local-runtime state gap:
knowledge-base bootstrap markers under `.vllm-sr/knowledge_bases/` were written
directly and corrupted marker YAML could abort runtime config generation. The
CLI now treats invalid KB bootstrap marker YAML as empty state and rewrites the
marker through a same-directory temp file plus `os.replace`. This improves
restart-safe local state for generated runtime config, but the debt remains open
for production shared-store policy, durable config projections, shared selector
stores, and Docker-backed smoke/E2E reruns.

A follow-up 2026-05-25 recheck found that the behavior fix still left the CLI
runtime orchestrator as the owner of runtime paths, KB state, and config
mutation. Those responsibilities now live in focused sibling modules:
`runtime_paths.py` owns `.vllm-sr` path and runtime-config writes,
`runtime_kb.py` owns KB source resolution and bootstrap marker state, and
`runtime_config_mutation.py` owns algorithm plus AMD GPU-default config
mutation. `runtime_support.py` is back to orchestration, and KB tests now live in
their own file. This narrows local-state ownership and structure debt, while the
broader TD034 shared-store and production durability questions remain open.

A follow-up 2026-05-25 Dashboard auth recheck found the server side still had a
narrow intake gap after the frontend local-token hardening and HttpOnly cookie
slices: bearer headers, `vsr_session` cookies, and `authToken` query parameters
were passed to JWT parsing as any non-empty string. The backend auth middleware
now normalizes all three compatible token transports through one boundary,
rejecting malformed or oversized token material before parsing, and session
cookie issue/clear responses now include explicit expiry timestamps in addition
to MaxAge. The same slice closes the current logout-revocation gap for newly
issued tokens: tokens now carry a persisted session id, authentication checks the
server-side session record, and logout revokes that session before clearing the
cookie. Tokens without a session id remain on the legacy compatibility path. A
follow-up lifecycle pass adds expiry and revocation indexes plus startup pruning
for expired or revoked sessions older than the retention window, so the new
durable session table does not grow without bound in long-running local stacks.
Cookie-focused tests now live in a dedicated session-cookie test file, and the
SQLite session store has its own lifecycle and pruning tests instead of growing
the broader handler or store hotspots. A follow-up store ownership split moves
role/permission persistence and audit-log persistence into sibling store files,
leaving the main SQLite store file focused on construction, lifecycle, and user
CRUD. Another handler ownership split removes the broad auth HTTP handler file
by separating public auth, admin user, admin metadata, and session-user helper
surfaces. This narrows session robustness without changing the public login
response contract, while production multi-replica session-store policy remains
open.

A follow-up 2026-05-25 Helm recheck found a deployment-layer mismatch in that
same Dashboard auth/workflow state surface. The backend now has restart-safe
SQLite auth/session and workflow stores when they point at a persistent path,
but the Helm deployment still ran the dashboard with container-local `./data`
paths. The chart now has a dashboard-local PVC template, `dashboard.persistence`
values, mounted `/app/data` env wiring for `DASHBOARD_AUTH_DB_PATH` and
`DASHBOARD_WORKFLOW_DB_PATH`, production values that enable the PVC, and a
template-time guard that rejects `dashboard.replicaCount > 1` until a shared
auth/session store exists. This narrows restart durability for one dashboard
replica, but it intentionally does not close the production HA session-store
contract.

A follow-up 2026-05-25 Helm selector-state recheck found a similar production
misconfiguration path for online model-selection state. RL-driven and GMTRouter
state now reload from local `storage_path`, and Elo can autosave to a local JSON
file, but those stores are still replica-local. The Helm chart now rejects
multi-replica router renders when an active decision uses `algorithm.type:
elo`, `rl_driven`, or `gmtrouter`; operators must keep one router replica,
externalize selector state, or explicitly set
`safetyGuards.rejectMultiReplicaLocalSelectorState=false` after accepting
replica-local learning divergence. This narrows production misconfiguration
risk, but it intentionally does not close the shared selector-store contract.

A later 2026-05-25 router runtime-boundary recheck found a narrow stale-global
state leak in the API helpers. Memory startup had already stopped adopting a
process-wide store when a runtime registry exists, but vector-store manager,
embedder, ingestion-pipeline, file-store, and selector registry helpers could
still fall back to legacy globals before their runtime registry dependencies were
published. Those helpers now treat the runtime registry as authoritative when it
is present and return nil until the matching dependency is published. Legacy
global fallback remains only for nil-registry compatibility callers. This
reduces cross-runtime state bleed, but the debt remains open for production
shared-store policy, durable config projections, broader selector-store parity,
and Docker-backed smoke/E2E reruns.

A follow-up ExtProc runtime-boundary recheck found the vectorstore RAG comment
and test coverage lagging behind current code. Request-time RAG already resolves
manager/embedder dependencies through `OpenAIRouter.RuntimeRegistry`; a new
regression test now locks nil-router, nil-registry, pre-publication, and
published-runtime behavior so this path cannot silently drift back to
API-server globals. The debt remains open for production shared-store policy,
durable config projections, broader selector-store parity, and Docker-backed
smoke/E2E reruns.

A follow-up 2026-05-25 recheck found the same pattern in API startup config
resolution. `InitWithRuntime` could still resolve `config.Get()` when a runtime
registry existed but had not published a config yet, binding the API server to a
process-wide config snapshot from another runtime. The resolver now treats the
runtime registry as authoritative: registry-backed API startup reads only
`runtimeRegistry.CurrentConfig()`, and the `config.Get()` fallback is preserved
only for the legacy nil-registry entrypoint. The slice is covered by focused
config-resolver tests, `go test ./pkg/apiserver`, `go test ./pkg/routerruntime`,
and the changed-set `make agent-ci-gate`. This narrows config-state bleed, while
durable deployed-config projections remain open.

A follow-up 2026-05-25 recheck found a related leak in runtime config refresh.
Registry-backed API config updates refreshed `routerruntime.Registry`, but the
classification service refresh path still called `config.Replace()`, publishing
the new config through the process-wide compatibility cache. `RefreshRuntimeConfig`
is now service-local; the nil-registry API updater explicitly preserves legacy
global publication, while registry-backed updates refresh the registry and
classification service without rewriting `config.Get()`. This narrows
cross-runtime config-state bleed further, while the production deployed-config
projection and broader compatibility-global retirement remain open.

A follow-up 2026-05-25 recheck found the same stale-state pattern in RL-driven
looper construction. ExtProc already owned the current selector registry on
`OpenAIRouter.ModelSelector`, but `NewRLDrivenLooper` read
`selection.GlobalRegistry` directly. Runtime looper construction now passes the
router-owned selector registry through `FactoryWithSelectionRegistry`, while the
old direct factory path keeps legacy global-selector compatibility. This narrows
cross-runtime selector-state bleed for RL-driven multi-round execution, while
shared production selector-store policy and broader compatibility-global
retirement remain open.

A follow-up 2026-05-25 recheck found the same pattern in ExtProc server startup.
`Server.Start()` sized gRPC buffers from `config.Get()` even when the server had
an active router config or runtime registry. Startup now resolves the router
config first, then the runtime registry, and only then the legacy global fallback.
This prevents one runtime's process-wide config cache from silently changing
another server's gRPC max-message-size limits. A later empty-registry edge
tightening made the same boundary stricter: once `Server.runtime` is present,
server config resolution no longer falls through to `config.Get()` before
publication. gRPC startup uses the explicit default max-message-size value in
that state, and the Kubernetes watcher decision remains non-Kubernetes instead
of adopting a conflicting process-wide config.

A follow-up 2026-05-25 recheck found two request-time plugin paths with the same
stale-config risk. Modality routing read `config.Get()` directly, while
system-prompt injection preferred the process-wide config before router-local
state and could panic when no matching decision existed. `router_config.go` now
centralizes router-owned config and decision precedence for those plugin paths:
router config first, classifier-local decision config next, and the legacy
process-wide config only for nil-router-config compatibility callers. Focused
tests cover conflicting router/global modality settings, conflicting
router/global system-prompt decisions, no global fallback when router config is
present, and the missing-decision no-panic path.

The same helper was tightened again after a 2026-05-25 current-source recheck
found an empty-runtime-registry edge. `OpenAIRouter.routerConfig()` and
`decisionByName()` now treat a present runtime registry as the runtime ownership
boundary: router-local config still wins, registry-published config wins next,
and a present-but-empty registry returns nil instead of falling through to
`config.Get()`. This keeps request-time modality and system-prompt behavior from
adopting another runtime's process-wide config during pre-publication states.

A final 2026-05-25 recheck in this ExtProc startup cluster found one more
process-wide config decision: `watchConfigAndReload` chose file watching versus
Kubernetes update watching from `config.Get()`. The watcher source decision now
uses the same active router config, runtime registry config, and legacy global
fallback precedence as gRPC startup sizing. Focused tests cover router config
overriding conflicting runtime/global source, runtime registry source winning
over global fallback, and preserved legacy global fallback.

The next 2026-05-25 constructor-boundary recheck found the same issue one step
earlier: `NewServer` still entered `NewOpenAIRouter(configPath)` even when a
runtime registry already held the active config. Server construction now uses a
runtime-registry-first config resolver and skips legacy `config.Replace`
publication on the registry path. A follow-up edge-case recheck found that an
empty but present runtime registry still fell through to the legacy
Kubernetes-global fallback. That path now parses the explicit config file
instead, so the presence of a runtime registry remains the boundary between
runtime-owned construction and legacy process-wide compatibility. Focused tests
cover registry config winning over a conflicting Kubernetes global without
parsing a missing file or changing `config.Get()`, empty-registry file parsing
without adopting a conflicting global, plus the preserved legacy Kubernetes
global fallback. The same empty-registry principle is now also covered at
server-config resolution time for gRPC sizing and watcher source selection.

A follow-up 2026-05-25 reload-boundary recheck found one more process-wide
publication path after startup. File reloads on servers with a runtime registry
still called `replaceReloadConfig`, which could publish a registry-owned config
through the process-wide compatibility cache after a successful router swap. The
reload path now skips that global publication whenever `Server.runtime` is
present and relies on the swapped router service plus `routerruntime.Registry`;
nil-registry reloads still preserve the legacy global update behavior.

A follow-up 2026-05-25 API mutation recheck found the same boundary issue in
managed knowledge-base persistence. `persistConfigAndSync` could still publish
through `config.Replace(newCfg)` when `runtimeConfig` was absent, even if the
API server already had a runtime registry. Config mutation publication now flows
through `ClassificationAPIServer.publishConfigMutation`: server-local state is
updated, `liveRuntimeConfig` remains the preferred fanout when present, registry
state is updated without touching the process-wide compatibility config, and
`config.Replace` is reserved for legacy nil-registry callers. Focused coverage
proves managed KB mutation updates the runtime registry and server current
config while leaving a conflicting `config.Get()` value unchanged.

A follow-up 2026-05-25 API startup recheck found a matching classification
service leak. `InitWithRuntime` had already stopped adopting process-wide
config and memory state, but classification service resolution still retried
`services.GetGlobalClassificationService()` before the runtime registry
published a service. Registry-backed API startup and the live service resolver
now read only `runtimeRegistry.ClassificationService()`, while the retrying
global-service fallback remains available for legacy nil-registry callers.
Focused tests cover pre-publication nil, registry-published service, and
preserved legacy global fallback.

## Evidence

- [src/semantic-router/pkg/config/canonical_defaults.go](../../../src/semantic-router/pkg/config/canonical_defaults.go)
- [dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts](../../../dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts)
- [src/semantic-router/pkg/responsestore/factory.go](../../../src/semantic-router/pkg/responsestore/factory.go)
- [src/semantic-router/pkg/responsestore/memory_store.go](../../../src/semantic-router/pkg/responsestore/memory_store.go)
- [src/semantic-router/pkg/responsestore/redis_store_integration_test.go](../../../src/semantic-router/pkg/responsestore/redis_store_integration_test.go)
- [e2e/testcases/response_api_restart_recovery.go](../../../e2e/testcases/response_api_restart_recovery.go)
- [src/semantic-router/pkg/routerreplay/store/factory.go](../../../src/semantic-router/pkg/routerreplay/store/factory.go)
- [src/semantic-router/pkg/routerreplay/store/memory.go](../../../src/semantic-router/pkg/routerreplay/store/memory.go)
- [src/semantic-router/pkg/extproc/router_replay_setup.go](../../../src/semantic-router/pkg/extproc/router_replay_setup.go)
- [src/semantic-router/pkg/extproc/router_replay_enablement_test.go](../../../src/semantic-router/pkg/extproc/router_replay_enablement_test.go)
- [src/semantic-router/pkg/extproc/router_build.go](../../../src/semantic-router/pkg/extproc/router_build.go)
- [src/semantic-router/pkg/extproc/router_build_runtime_config_test.go](../../../src/semantic-router/pkg/extproc/router_build_runtime_config_test.go)
- [src/semantic-router/pkg/extproc/server.go](../../../src/semantic-router/pkg/extproc/server.go)
- [src/semantic-router/pkg/extproc/server_reload_test.go](../../../src/semantic-router/pkg/extproc/server_reload_test.go)
- [src/semantic-router/pkg/extproc/router_config.go](../../../src/semantic-router/pkg/extproc/router_config.go)
- [src/semantic-router/pkg/extproc/router_runtime_services.go](../../../src/semantic-router/pkg/extproc/router_runtime_services.go)
- [src/semantic-router/pkg/extproc/router_runtime_services_test.go](../../../src/semantic-router/pkg/extproc/router_runtime_services_test.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_vectorstore.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_vectorstore.go)
- [src/semantic-router/pkg/extproc/req_filter_modality.go](../../../src/semantic-router/pkg/extproc/req_filter_modality.go)
- [src/semantic-router/pkg/extproc/req_filter_modality_test.go](../../../src/semantic-router/pkg/extproc/req_filter_modality_test.go)
- [src/semantic-router/pkg/extproc/req_filter_sys_prompt.go](../../../src/semantic-router/pkg/extproc/req_filter_sys_prompt.go)
- [src/semantic-router/pkg/extproc/req_filter_sys_prompt_test.go](../../../src/semantic-router/pkg/extproc/req_filter_sys_prompt_test.go)
- [src/semantic-router/pkg/vectorstore/manager.go](../../../src/semantic-router/pkg/vectorstore/manager.go)
- [src/semantic-router/pkg/vectorstore/filestore.go](../../../src/semantic-router/pkg/vectorstore/filestore.go)
- [src/semantic-router/pkg/startupstatus/status.go](../../../src/semantic-router/pkg/startupstatus/status.go)
- [src/semantic-router/pkg/apiserver/server.go](../../../src/semantic-router/pkg/apiserver/server.go)
- [src/semantic-router/pkg/apiserver/runtime_config.go](../../../src/semantic-router/pkg/apiserver/runtime_config.go)
- [src/semantic-router/pkg/apiserver/runtime_dependencies.go](../../../src/semantic-router/pkg/apiserver/runtime_dependencies.go)
- [src/semantic-router/pkg/apiserver/runtime_globals_test.go](../../../src/semantic-router/pkg/apiserver/runtime_globals_test.go)
- [src/semantic-router/pkg/apiserver/server_memory_test.go](../../../src/semantic-router/pkg/apiserver/server_memory_test.go)
- [src/semantic-router/pkg/apiserver/route_taxonomy_runtime_config_test.go](../../../src/semantic-router/pkg/apiserver/route_taxonomy_runtime_config_test.go)
- [src/semantic-router/pkg/apiserver/route_startup_status.go](../../../src/semantic-router/pkg/apiserver/route_startup_status.go)
- [src/semantic-router/pkg/apiserver/server_ready_test.go](../../../src/semantic-router/pkg/apiserver/server_ready_test.go)
- [src/semantic-router/cmd/runtime_bootstrap.go](../../../src/semantic-router/cmd/runtime_bootstrap.go)
- [src/semantic-router/pkg/selection/storage.go](../../../src/semantic-router/pkg/selection/storage.go)
- [src/semantic-router/pkg/selection/storage_test.go](../../../src/semantic-router/pkg/selection/storage_test.go)
- [src/semantic-router/pkg/selection/elo.go](../../../src/semantic-router/pkg/selection/elo.go)
- [src/semantic-router/pkg/selection/rl_driven.go](../../../src/semantic-router/pkg/selection/rl_driven.go)
- [src/semantic-router/pkg/selection/rl_driven_test.go](../../../src/semantic-router/pkg/selection/rl_driven_test.go)
- [src/semantic-router/pkg/selection/gmtrouter.go](../../../src/semantic-router/pkg/selection/gmtrouter.go)
- [src/semantic-router/pkg/selection/gmtrouter_storage.go](../../../src/semantic-router/pkg/selection/gmtrouter_storage.go)
- [src/semantic-router/pkg/selection/gmtrouter_test.go](../../../src/semantic-router/pkg/selection/gmtrouter_test.go)
- [src/semantic-router/pkg/config/selection_config.go](../../../src/semantic-router/pkg/config/selection_config.go)
- [src/semantic-router/pkg/extproc/router_selection.go](../../../src/semantic-router/pkg/extproc/router_selection.go)
- [src/semantic-router/pkg/extproc/router_selection_config_test.go](../../../src/semantic-router/pkg/extproc/router_selection_config_test.go)
- [src/semantic-router/pkg/extproc/req_filter_looper.go](../../../src/semantic-router/pkg/extproc/req_filter_looper.go)
- [src/semantic-router/pkg/looper/looper.go](../../../src/semantic-router/pkg/looper/looper.go)
- [src/semantic-router/pkg/looper/rl_driven.go](../../../src/semantic-router/pkg/looper/rl_driven.go)
- [src/semantic-router/pkg/looper/rl_driven_test.go](../../../src/semantic-router/pkg/looper/rl_driven_test.go)
- [src/vllm-sr/cli/algorithms.py](../../../src/vllm-sr/cli/algorithms.py)
- [src/vllm-sr/tests/test_algorithm_config.py](../../../src/vllm-sr/tests/test_algorithm_config.py)
- [dashboard/frontend/src/lib/dslSchemas.ts](../../../dashboard/frontend/src/lib/dslSchemas.ts)
- [dashboard/backend/evaluation/db.go](../../../dashboard/backend/evaluation/db.go)
- [dashboard/backend/mlpipeline/runner.go](../../../dashboard/backend/mlpipeline/runner.go)
- [dashboard/backend/workflowstore/mlpipeline.go](../../../dashboard/backend/workflowstore/mlpipeline.go)
- [dashboard/backend/workflowstore/store_test.go](../../../dashboard/backend/workflowstore/store_test.go)
- [dashboard/backend/auth/http_handlers.go](../../../dashboard/backend/auth/http_handlers.go)
- [dashboard/backend/auth/session_cookie.go](../../../dashboard/backend/auth/session_cookie.go)
- [dashboard/backend/auth/handlers_test.go](../../../dashboard/backend/auth/handlers_test.go)
- [dashboard/backend/router/auth_routes.go](../../../dashboard/backend/router/auth_routes.go)
- [dashboard/frontend/src/utils/authFetch.ts](../../../dashboard/frontend/src/utils/authFetch.ts)
- [dashboard/frontend/src/utils/authFetch.test.ts](../../../dashboard/frontend/src/utils/authFetch.test.ts)
- [dashboard/frontend/src/contexts/authSession.ts](../../../dashboard/frontend/src/contexts/authSession.ts)
- [dashboard/frontend/src/contexts/authSession.test.ts](../../../dashboard/frontend/src/contexts/authSession.test.ts)
- [dashboard/frontend/src/contexts/AuthContext.tsx](../../../dashboard/frontend/src/contexts/AuthContext.tsx)
- [src/vllm-sr/cli/service_defaults.py](../../../src/vllm-sr/cli/service_defaults.py)
- [src/vllm-sr/tests/test_storage_backends.py](../../../src/vllm-sr/tests/test_storage_backends.py)
- [website/docs/tutorials/global/stores-and-tools.md](../../../website/docs/tutorials/global/stores-and-tools.md)
- [website/docs/troubleshooting/common-errors.md](../../../website/docs/troubleshooting/common-errors.md)
- [dashboard/backend/auth/store.go](../../../dashboard/backend/auth/store.go)
- [dashboard/backend/handlers/mlpipeline.go](../../../dashboard/backend/handlers/mlpipeline.go)
- [dashboard/backend/handlers/evaluation.go](../../../dashboard/backend/handlers/evaluation.go)
- [dashboard/backend/handlers/status_collectors.go](../../../dashboard/backend/handlers/status_collectors.go)
- [dashboard/backend/handlers/openclaw.go](../../../dashboard/backend/handlers/openclaw.go)
- [dashboard/backend/handlers/openclaw_rooms.go](../../../dashboard/backend/handlers/openclaw_rooms.go)
- [dashboard/frontend/src/hooks/useConversationStorage.ts](../../../dashboard/frontend/src/hooks/useConversationStorage.ts)
- [dashboard/frontend/src/hooks/conversationStorage.ts](../../../dashboard/frontend/src/hooks/conversationStorage.ts)
- [dashboard/frontend/src/hooks/conversationStorage.test.ts](../../../dashboard/frontend/src/hooks/conversationStorage.test.ts)
- [dashboard/frontend/src/hooks/usePlaygroundQueue.ts](../../../dashboard/frontend/src/hooks/usePlaygroundQueue.ts)
- [dashboard/frontend/src/hooks/playgroundQueueStorage.ts](../../../dashboard/frontend/src/hooks/playgroundQueueStorage.ts)
- [dashboard/frontend/src/hooks/playgroundQueueStorage.test.ts](../../../dashboard/frontend/src/hooks/playgroundQueueStorage.test.ts)
- [dashboard/frontend/src/utils/authFetch.ts](../../../dashboard/frontend/src/utils/authFetch.ts)
- [src/vllm-sr/cli/docker_start.py](../../../src/vllm-sr/cli/docker_start.py)
- [src/vllm-sr/cli/commands/runtime_support.py](../../../src/vllm-sr/cli/commands/runtime_support.py)
- [src/vllm-sr/cli/commands/runtime_paths.py](../../../src/vllm-sr/cli/commands/runtime_paths.py)
- [src/vllm-sr/cli/commands/runtime_kb.py](../../../src/vllm-sr/cli/commands/runtime_kb.py)
- [src/vllm-sr/cli/commands/runtime_config_mutation.py](../../../src/vllm-sr/cli/commands/runtime_config_mutation.py)
- [src/vllm-sr/tests/test_runtime_support.py](../../../src/vllm-sr/tests/test_runtime_support.py)
- [src/vllm-sr/tests/test_runtime_kb.py](../../../src/vllm-sr/tests/test_runtime_kb.py)
- [src/vllm-sr/README.md](../../../src/vllm-sr/README.md)
- [docs/agent/state-taxonomy-and-inventory.md](../state-taxonomy-and-inventory.md)
- [docs/agent/tech-debt/td-005-dashboard-enterprise-console-foundations.md](td-005-dashboard-enterprise-console-foundations.md)
- [docs/agent/tech-debt/td-021-milvus-adapter-duplication-across-runtime-stores.md](td-021-milvus-adapter-duplication-across-runtime-stores.md)
- [docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md](td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [docs/agent/tech-debt/td-032-training-evaluation-artifact-contract-drift.md](td-032-training-evaluation-artifact-contract-drift.md)

## Why It Matters

- Large-scale or user-visible runtime features can look production-ready while still defaulting to in-memory retention. Response API history now defaults to Redis and has restart-recovery coverage, and router replay now infers configured durable backend blocks when `store_backend` is omitted, but a router restart can still drop cache state or explicitly memory-backed replay records unless operators wire a stronger backend.
- Several restart paths only preserve part of the truth. Vector-store files can remain on disk while file metadata and store inventory vanish from process memory when `metadata_store: memory` is selected; the local CLI now supplies Postgres metadata defaults when that durable mode is selected, and runtime-registry-backed API servers no longer bind vector/file helpers to stale process-wide vector state before publication. ML pipeline jobs now recover to a durable failed terminal state after restart, but the rest of the state model is still fragmented. The earlier model-research campaign reference is stale in current source because the `dashboard/backend/modelresearch` package is no longer present.
- Dashboard and CLI state ownership is inconsistent across SQLite, JSON files, container-mounted workspace directories, log scraping, EventSource streams, and frontend localStorage. Helm now preserves dashboard-local SQLite auth/session/workflow state through an optional PVC and blocks unsupported multi-replica dashboard settings, but HA dashboard deployment still needs a shared auth/session store. The remaining mix makes recovery testing and contributor reasoning harder than necessary.
- Browser-local state is now more explicitly bounded for playground conversations, queued tasks, and auth-token reuse. Dashboard auth also issues, clears, and reloads from a server cookie, and the Helm production profile can preserve the local auth/session database across pod restarts, but the dashboard still needs product-level decisions for which chat/session and session-lifecycle surfaces are demo-only browser UX and which ones must move fully server-side.
- Online model-selection state also lacks a shared durability contract. RL-driven user preferences and session context now survive restart when `storage_path` is configured, decision-scoped RLDriven/GMTRouter storage config now reaches runtime selector construction, API servers with a runtime registry no longer fall back to `selection.GlobalRegistry` before selector publication, RL-driven looper execution now receives the runtime selector registry, GMTRouter local JSON state is now atomically saved and defensively loaded, and Helm blocks multi-replica renders for `elo`, `rl_driven`, and `gmtrouter` decisions by default. Selector state can still live in local files or process memory, so shared-store parity remains required before these selectors are fully production HA.
- Telemetry for control-plane progress is also not first class. Router readiness now consumes the same Redis/file startup-state resolver as `/startup-status`, but runtime status can still fall back to temp-owned JSON; dashboard health has historically depended on local collectors; and not every long-running dashboard workflow has durable typed status records.
- Without one state taxonomy, new features are likely to keep choosing storage and recovery behavior ad hoc, increasing open-source collaboration cost and making future platform hardening slower and riskier.

## Desired End State

- The repository defines one cross-stack state taxonomy for router and dashboard features, including at least: ephemeral request state, restart-safe local state, shared durable workflow state, and audit or analytics telemetry.
- User-visible runtime surfaces such as Response API, router replay, vector-store metadata, file metadata, workflow jobs, and dashboard chat or session features either use durable server-owned storage or are explicitly documented and gated as ephemeral. Browser token hardening, cookie-backed reload, the Helm single-replica PVC guard, and the Helm local learning-selector multi-replica guard are interim guards, not the final production shared-store contracts.
- Deployed config intent stays canonical in YAML or DSL, while dashboard-facing query, audit, and topology needs rely on a persisted derived projection rather than ad hoc reparsing or a second mutable primary source of truth.
- CLI-mounted local workspace state is treated as one adapter for local development rather than the only implicit persistence contract for production-like features.
- Background progress and health reporting move toward typed, restart-aware state records or events instead of log scraping, temp-file guesses, or browser-only localStorage.
- Existing subsystem debts such as TD005, TD031, and TD032 can converge on one shared control-plane state model instead of drifting independently (TD021 Milvus lifecycle duplication is retired; see archived entry).

## Exit Criteria

- The repo has one indexed, contributor-visible inventory that classifies the major runtime and dashboard state surfaces by owner, backend, restart behavior, and intended durability level.
- Canonical defaults, dashboard defaults, and docs no longer imply memory-backed retention for user-visible product surfaces unless the feature is explicitly marked ephemeral.
- Router-side metadata for vector stores, files, replay, response records, selector local state, and startup readiness can survive restart through a documented persistence seam rather than process memory alone. The Response API Redis path now satisfies this for response records, `/ready` and `/startup-status` now share Redis/file startup-state resolution, RLDriven and GMTRouter decision-scoped storage config now reaches selector runtime construction, GMTRouter has a restart-safer local file seam, and vector metadata plus replay have narrowed durable seams but still need full local smoke/E2E reruns once Docker is available.
- Dashboard workflow state for at least one long-running subsystem is restart-safe and server-owned, with progress and terminal state observable without relying on localStorage or log scraping.
- The active YAML or DSL deployment can be projected into a persisted read model for models, signals, decisions, plugins, and validation state without creating a second mutable primary config source.
- Validation or E2E coverage includes at least one restart-recovery contract for a stateful router or dashboard workflow.
