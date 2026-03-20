# TD034: Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract

## Status

Open

## Scope

`src/semantic-router/pkg/{config,responsestore,routerreplay,vectorstore,startupstatus}/**`, `dashboard/backend/{auth,evaluation,mlpipeline,modelresearch,handlers}/**`, `dashboard/frontend/src/{hooks,components,utils}/**`, `src/vllm-sr/cli/**`, and adjacent docs or defaults that present stateful runtime features

## Summary

The repository now exposes several user-visible stateful product surfaces beyond request-time routing, but the persistence and recovery contract for those surfaces is still implicit and fragmented. Router defaults still present memory-backed behavior for core runtime facilities such as semantic cache, Response API storage, router replay, and vector-store metadata, while the dashboard defaults surface the same in-memory or workspace-local expectations to operators. Dashboard features then split their own state across SQLite databases, JSON files under workspace-local directories, in-memory job maps and event channels, log-derived status collectors, and browser localStorage for chat history and auth propagation. Model-selection learning state follows the same pattern with in-memory maps or local JSON autosave rather than a shared durable store. The CLI mounts `.vllm-sr` and `dashboard-data` into the local container, but the repo still lacks one cross-stack contract that says which state is ephemeral, which must survive restart, which belongs in shared storage, and which telemetry or progress signals are durable enough to drive recovery.

## Evidence

- [src/semantic-router/pkg/config/canonical_defaults.go](../../../src/semantic-router/pkg/config/canonical_defaults.go)
- [dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts](../../../dashboard/frontend/src/pages/configPageRouterDefaultsCatalog.ts)
- [src/semantic-router/pkg/responsestore/factory.go](../../../src/semantic-router/pkg/responsestore/factory.go)
- [src/semantic-router/pkg/responsestore/memory_store.go](../../../src/semantic-router/pkg/responsestore/memory_store.go)
- [src/semantic-router/pkg/routerreplay/store/factory.go](../../../src/semantic-router/pkg/routerreplay/store/factory.go)
- [src/semantic-router/pkg/routerreplay/store/memory.go](../../../src/semantic-router/pkg/routerreplay/store/memory.go)
- [src/semantic-router/pkg/vectorstore/manager.go](../../../src/semantic-router/pkg/vectorstore/manager.go)
- [src/semantic-router/pkg/vectorstore/filestore.go](../../../src/semantic-router/pkg/vectorstore/filestore.go)
- [src/semantic-router/pkg/startupstatus/status.go](../../../src/semantic-router/pkg/startupstatus/status.go)
- [src/semantic-router/pkg/selection/elo.go](../../../src/semantic-router/pkg/selection/elo.go)
- [src/semantic-router/pkg/selection/rl_driven.go](../../../src/semantic-router/pkg/selection/rl_driven.go)
- [src/semantic-router/pkg/selection/gmtrouter.go](../../../src/semantic-router/pkg/selection/gmtrouter.go)
- [dashboard/backend/evaluation/db.go](../../../dashboard/backend/evaluation/db.go)
- [dashboard/backend/mlpipeline/runner.go](../../../dashboard/backend/mlpipeline/runner.go)
- [dashboard/backend/modelresearch/manager.go](../../../dashboard/backend/modelresearch/manager.go)
- [dashboard/backend/auth/store.go](../../../dashboard/backend/auth/store.go)
- [dashboard/backend/handlers/mlpipeline.go](../../../dashboard/backend/handlers/mlpipeline.go)
- [dashboard/backend/handlers/evaluation.go](../../../dashboard/backend/handlers/evaluation.go)
- [dashboard/backend/handlers/modelresearch.go](../../../dashboard/backend/handlers/modelresearch.go)
- [dashboard/backend/handlers/status_collectors.go](../../../dashboard/backend/handlers/status_collectors.go)
- [dashboard/backend/handlers/openclaw.go](../../../dashboard/backend/handlers/openclaw.go)
- [dashboard/backend/handlers/openclaw_rooms.go](../../../dashboard/backend/handlers/openclaw_rooms.go)
- [dashboard/frontend/src/hooks/useConversationStorage.ts](../../../dashboard/frontend/src/hooks/useConversationStorage.ts)
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

- The repo has one indexed, contributor-visible inventory that classifies the major runtime and dashboard state surfaces by owner, backend, restart behavior, and intended durability level.
- Canonical defaults, dashboard defaults, and docs no longer imply memory-backed retention for user-visible product surfaces unless the feature is explicitly marked ephemeral.
- Router-side metadata for vector stores, files, replay, and response records can survive restart through a documented persistence seam rather than process memory alone.
- Dashboard workflow state for at least one long-running subsystem is restart-safe and server-owned, with progress and terminal state observable without relying on localStorage or log scraping.
- The active YAML or DSL deployment can be projected into a persisted read model for models, signals, decisions, plugins, and validation state without creating a second mutable primary config source.
- Validation or E2E coverage includes at least one restart-recovery contract for a stateful router or dashboard workflow.
