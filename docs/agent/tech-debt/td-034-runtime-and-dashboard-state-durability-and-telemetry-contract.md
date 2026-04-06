# TD034: Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract

## Status

Closed

## Scope

`src/semantic-router/pkg/{config,responsestore,routerreplay,vectorstore,startupstatus}/**`, `dashboard/backend/{auth,evaluation,mlpipeline,modelresearch,handlers}/**`, `dashboard/frontend/src/{hooks,components,utils}/**`, `src/vllm-sr/cli/**`, and adjacent docs or defaults that present stateful runtime features

## Summary

The repository now has one contributor-visible state taxonomy for restart-sensitive router and dashboard surfaces, plus concrete restart-aware seams for the two biggest gaps this debt was tracking. Router-side vector-store metadata, uploaded-file records, and ingestion statuses now persist under the configured file-storage root and recover cleanly across restart, while dashboard config apply and setup activation now persist a hash-guarded active-config projection under `.vllm-sr/active-config-projection.json` instead of forcing each query path to reparse YAML or DSL on demand. Response API and router replay defaults, dashboard defaults, and user-facing docs now describe Redis and Postgres as the durable defaults and fence `memory` to local development. The enterprise-console concerns that originally remained under [TD005](td-005-dashboard-enterprise-console-foundations.md) are now resolved or explicitly classified: browser auth uses server-owned `HttpOnly` sessions, and playground chat history is documented as browser-local convenience state rather than shared control-plane state.

## Evidence

- [src/semantic-router/pkg/config/canonical_defaults.go](../../../src/semantic-router/pkg/config/canonical_defaults.go)
- [dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts](../../../dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts)
- [src/semantic-router/pkg/responsestore/factory.go](../../../src/semantic-router/pkg/responsestore/factory.go)
- [src/semantic-router/pkg/responsestore/memory_store.go](../../../src/semantic-router/pkg/responsestore/memory_store.go)
- [src/semantic-router/pkg/routerreplay/store/factory.go](../../../src/semantic-router/pkg/routerreplay/store/factory.go)
- [src/semantic-router/pkg/routerreplay/store/memory.go](../../../src/semantic-router/pkg/routerreplay/store/memory.go)
- [src/semantic-router/pkg/vectorstore/manager.go](../../../src/semantic-router/pkg/vectorstore/manager.go)
- [src/semantic-router/pkg/vectorstore/filestore.go](../../../src/semantic-router/pkg/vectorstore/filestore.go)
- [src/semantic-router/pkg/vectorstore/pipeline.go](../../../src/semantic-router/pkg/vectorstore/pipeline.go)
- [src/semantic-router/pkg/vectorstore/local_metadata_registry.go](../../../src/semantic-router/pkg/vectorstore/local_metadata_registry.go)
- [src/semantic-router/pkg/vectorstore/persistence_test.go](../../../src/semantic-router/pkg/vectorstore/persistence_test.go)
- [src/semantic-router/pkg/routerruntime/vectorstore_runtime.go](../../../src/semantic-router/pkg/routerruntime/vectorstore_runtime.go)
- [src/semantic-router/pkg/startupstatus/status.go](../../../src/semantic-router/pkg/startupstatus/status.go)
- [src/semantic-router/pkg/selection/elo.go](../../../src/semantic-router/pkg/selection/elo.go)
- [src/semantic-router/pkg/selection/rl_driven.go](../../../src/semantic-router/pkg/selection/rl_driven.go)
- [src/semantic-router/pkg/selection/gmtrouter.go](../../../src/semantic-router/pkg/selection/gmtrouter.go)
- [dashboard/backend/evaluation/db.go](../../../dashboard/backend/evaluation/db.go)
- [dashboard/backend/mlpipeline/runner.go](../../../dashboard/backend/mlpipeline/runner.go)
- [dashboard/backend/modelresearch/manager.go](../../../dashboard/backend/modelresearch/manager.go)
- [dashboard/backend/auth/store.go](../../../dashboard/backend/auth/store.go)
- [dashboard/backend/handlers/config_projection.go](../../../dashboard/backend/handlers/config_projection.go)
- [dashboard/backend/handlers/config_projection_test.go](../../../dashboard/backend/handlers/config_projection_test.go)
- [dashboard/backend/handlers/mlpipeline.go](../../../dashboard/backend/handlers/mlpipeline.go)
- [dashboard/backend/handlers/evaluation.go](../../../dashboard/backend/handlers/evaluation.go)
- [dashboard/backend/handlers/modelresearch.go](../../../dashboard/backend/handlers/modelresearch.go)
- [dashboard/backend/handlers/status_collectors.go](../../../dashboard/backend/handlers/status_collectors.go)
- [dashboard/backend/handlers/openclaw.go](../../../dashboard/backend/handlers/openclaw.go)
- [dashboard/backend/handlers/openclaw_rooms.go](../../../dashboard/backend/handlers/openclaw_rooms.go)
- [dashboard/backend/handlers/runtime_config_apply.go](../../../dashboard/backend/handlers/runtime_config_apply.go)
- [dashboard/backend/handlers/setup.go](../../../dashboard/backend/handlers/setup.go)
- [dashboard/frontend/src/hooks/useConversationStorage.ts](../../../dashboard/frontend/src/hooks/useConversationStorage.ts)
- [dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts](../../../dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts)
- [dashboard/frontend/src/pages/InsightsPage.tsx](../../../dashboard/frontend/src/pages/InsightsPage.tsx)
- [dashboard/frontend/src/utils/authFetch.ts](../../../dashboard/frontend/src/utils/authFetch.ts)
- [src/vllm-sr/cli/docker_start.py](../../../src/vllm-sr/cli/docker_start.py)
- [src/vllm-sr/cli/commands/runtime_support.py](../../../src/vllm-sr/cli/commands/runtime_support.py)
- [src/vllm-sr/README.md](../../../src/vllm-sr/README.md)
- [docs/agent/state-taxonomy-and-inventory.md](../state-taxonomy-and-inventory.md)
- [docs/agent/tech-debt/td-005-dashboard-enterprise-console-foundations.md](td-005-dashboard-enterprise-console-foundations.md)
- [docs/agent/tech-debt/td-021-milvus-adapter-duplication-across-runtime-stores.md](td-021-milvus-adapter-duplication-across-runtime-stores.md)
- [docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md](td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [docs/agent/tech-debt/td-032-training-evaluation-artifact-contract-drift.md](td-032-training-evaluation-artifact-contract-drift.md)

## Why It Matters

- Large-scale or user-visible runtime features can look production-ready while still defaulting to in-memory retention. A router restart can therefore silently drop replay records, Response API conversations, or cache state unless operators discover and wire a stronger backend themselves.
- Several restart paths only preserve part of the truth. Vector-store files can remain on disk while file metadata and store inventory vanish from process memory; model-research campaigns persist state files but mark running work as failed on restart; ML pipeline jobs keep outputs on disk but lose job status because the scheduler lives only in memory.
- Dashboard and CLI state ownership is inconsistent across SQLite, JSON files, container-mounted workspace directories, log scraping, EventSource streams, and frontend localStorage. That makes HA, multi-replica dashboard deployment, recovery testing, and contributor reasoning harder than necessary.
- Online model-selection state also lacks a shared durability contract. Ratings, preferences, and session context can live in local JSON or process memory, which makes behavior drift between replicas and across restarts likely once those selectors are promoted beyond research or local smoke usage.
- Telemetry for control-plane progress is also not first class. Runtime status may fall back to temp-owned JSON, dashboard health is inferred from log tails, and long-running ML jobs expose progress by parsing subprocess output instead of writing durable typed status records.
- Without one state taxonomy, new features are likely to keep choosing storage and recovery behavior ad hoc, increasing open-source collaboration cost and making future platform hardening slower and riskier.

## Desired End State

- The repository defines one cross-stack state taxonomy for router and dashboard features, including at least: ephemeral request state, restart-safe local state, shared durable workflow state, and audit or analytics telemetry.
- User-visible runtime surfaces such as Response API, router replay, vector-store metadata, file metadata, workflow jobs, and dashboard chat or session features either use durable server-owned storage or are explicitly documented and gated as ephemeral.
- Deployed config intent stays canonical in YAML or DSL, while dashboard-facing query, audit, and topology needs rely on a persisted derived projection rather than ad hoc reparsing or a second mutable primary source of truth.
- CLI-mounted local workspace state is treated as one adapter for local development rather than the only implicit persistence contract for production-like features.
- Background progress and health reporting move toward typed, restart-aware state records or events instead of log scraping, temp-file guesses, or browser-only localStorage.
- Existing subsystem debts such as TD005, TD021, TD031, and TD032 can converge on one shared control-plane state model instead of drifting independently.

## Exit Criteria

- Satisfied on 2026-04-06: [state-taxonomy-and-inventory.md](../state-taxonomy-and-inventory.md) is now the indexed inventory for restart-sensitive router and dashboard state, classifying ownership, backend, restart behavior, and intended durability level.
- Satisfied on 2026-04-06: canonical defaults, dashboard defaults, Insights examples, and API or observability docs describe Redis and Postgres as the durable defaults for Response API and router replay, while `memory` is explicitly marked local-development-only.
- Satisfied on 2026-04-06: router-side vector-store metadata, uploaded-file records, and ingestion statuses now survive restart through the documented local metadata registry seam under `${file_storage_dir}/.vectorstore-metadata.json`, with startup reconciliation and interrupted-ingestion recovery.
- Satisfied before umbrella closure and reaffirmed on 2026-04-06: dashboard evaluation task metadata remains server-owned and restart-safe through SQLite-backed state under TD032, so at least one long-running dashboard workflow already meets the restart-safe requirement without relying on browser storage.
- Satisfied on 2026-04-06: the active YAML or DSL deployment can now be projected into a persisted read model at `.vllm-sr/active-config-projection.json` for models, signals, decisions, plugins, validation state, and DSL snapshot without introducing a second mutable primary config source.
- Satisfied on 2026-04-06: restart-recovery coverage now includes vector-store metadata persistence and interrupted-ingestion recovery in `pkg/vectorstore/persistence_test.go`, alongside the existing restart-aware workflow coverage carried by TD032 and response API recovery E2E.

## Retirement Notes

- `src/semantic-router/pkg/vectorstore/local_metadata_registry.go` now persists vector-store store metadata, uploaded-file records, and file-ingestion status records under the configured file-storage root, while `pkg/routerruntime/vectorstore_runtime.go` reconciles persisted metadata against the live backend, restores file counts, and marks interrupted ingestions as failed with a typed `interrupted` error.
- `src/semantic-router/pkg/routerprojection/active_config.go` and `dashboard/backend/handlers/config_projection.go` now define a derived, hash-guarded read model for models, signals, decisions, plugins, validation, and DSL snapshot, and `runtime_config_apply.go` plus `setup.go` persist or consume that projection during apply, restore, and setup activation flows.
- `docs/agent/state-taxonomy-and-inventory.md`, `dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts`, `dashboard/frontend/src/pages/InsightsPage.tsx`, and the API or observability docs now describe which surfaces are ephemeral, which are restart-safe local state, and which durable defaults operators should expect.
- The browser-auth and chat-history concerns that originally remained under TD005 are now explicitly resolved or classified: browser auth is server-owned through `HttpOnly` same-origin session cookies, while playground chat history stays browser-local convenience state unless a future product decision promotes it into shared workflow storage. This umbrella debt stays closed because the taxonomy now distinguishes those states instead of leaving them ambiguous.

## Validation

- `make agent-validate`
  Run in `/Users/bitliu/vs`
- `make test-semantic-router`
  Run in `/Users/bitliu/vs`
- `make agent-ci-gate AGENT_CHANGED_FILES_PATH=/tmp/vsr_td034_changed.txt`
  Run in `/Users/bitliu/vs`
- `make agent-dev ENV=cpu`
  Run in `/Users/bitliu/vs`
- `make agent-serve-local ENV=cpu`
  Run in `/Users/bitliu/vs`
- `make agent-smoke-local`
  Run in `/Users/bitliu/vs`
- `go test ./pkg/vectorstore ./pkg/routerprojection ./pkg/routerruntime`
  Run in `/Users/bitliu/vs/src/semantic-router`
- `go test ./handlers -run 'TestPersistActiveConfigProjectionWritesReadModel|TestApplyWrittenConfigPersistsActiveConfigProjection|TestSetupStateHandler|TestSetupValidateHandler'`
  Run in `/Users/bitliu/vs/dashboard/backend`
