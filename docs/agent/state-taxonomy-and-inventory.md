# Router and Dashboard State Taxonomy and Inventory

This document is the canonical inventory for restart-sensitive router and dashboard state.

Use it to answer three questions before adding or changing a stateful feature:

- what kind of state is this
- who owns it
- what durability and recovery behavior users should expect

## State Taxonomy

### `ephemeral_request_state`

- Per-request or per-connection working state that may live only in memory.
- Examples: extproc request context, in-flight streaming buffers, active SSE or WebSocket client maps, short-lived retry state.
- Losing this state on process restart is acceptable.

### `restart_safe_local_state`

- State that should survive a local process restart in one workspace or one node, but does not need to be shared across replicas by default.
- Examples: local-dev bootstrap outputs under `.vllm-sr/`, local auth/evaluation SQLite files, generated runtime config snapshots, local artifact directories.
- This is acceptable for local development and single-node tooling, but it is not a substitute for product durability in multi-user or multi-replica deployments.

### `shared_durable_workflow_state`

- State that users or operators expect to survive restart and remain authoritative across processes, containers, or replicas.
- Examples: user accounts, audit logs, workflow jobs, campaign state, chat history if presented as a product surface, vector-store metadata, file registries, response history, replay history, model-selection learning state.
- This state should live behind a server-owned storage contract, not browser storage or process memory.

### `audit_analytics_telemetry`

- Append-oriented facts about progress, lifecycle, usage, or recovery that operators need for debugging, audits, or reporting.
- Examples: typed workflow events, startup progress records, replay aggregates, deploy history, durable status transitions.
- This is distinct from request-local logs and should not depend on log scraping alone.

### `derived_projection_state`

- A query-oriented projection derived from a canonical source of truth.
- This state may be rebuilt, but it should still be persisted if operators or the dashboard rely on it.
- Examples: normalized tables for deployed models, signals, decisions, plugins, DSL snapshots, topology views, or current active config version.

## Source Of Truth Rules

- Keep canonical router intent in YAML and DSL, not in a second mutable primary database model.
- If the dashboard needs fast querying, filtering, audit, or joins, create a persisted projection from the active YAML or DSL rather than introducing dual-primary writes.
- Treat `.vllm-sr/` and `dashboard-data/` as local-dev adapters unless a feature is explicitly documented as single-workspace only.
- Keep live connection registries in memory, but move user-visible entities and workflow progress behind durable records.

## Current Inventory

| Surface | Primary owner | Current backend / default | Current durability class | Restart behavior today | Scale risk | Recommended direction |
| --- | --- | --- | --- | --- | --- | --- |
| Response API stored responses and conversations | router runtime, `src/semantic-router/pkg/responsestore/**` | Default `redis`; optional `memory` for local dev only | `shared_durable_workflow_state` | Response and conversation history survives restart when using the default Redis backend. The `memory` backend emits a startup warning and loses all data on restart. | Replica-local only when `memory` is explicitly selected; Redis backend is shared across replicas | Keep metadata and conversation chain in a durable server-owned store by default for product use. Prefer relational storage for metadata and queryability; keep large payloads in blob/object storage only if needed later. |
| Router replay records | router runtime, `src/semantic-router/pkg/routerreplay/**`, `src/semantic-router/pkg/extproc/router_replay_setup.go` | Default `postgres`; optional `redis`, `milvus`, `memory` (local dev only) | `audit_analytics_telemetry` with `shared_durable_workflow_state` default | Replay history survives restart when using the default Postgres backend. The `memory` backend emits a startup warning and loses all records on restart. | Postgres provides SQL queryability for audit and compliance. Redis available for lightweight deployments. | Keep metadata and replay records in a durable server-owned store by default. Postgres is the default for long-term audit retention and compliance. Keep Milvus only when semantic replay search is explicitly needed. |
| Semantic cache entries | router runtime, `src/semantic-router/pkg/cache/**` | Default `memory`; optional Redis, Milvus, hybrid | `ephemeral_request_state` in local dev; shared cache in scaled deploys | Restart flushes cache; replicas do not share hot entries by default | Cold-start latency, inconsistent cache hit rates, and uneven behavior across replicas | Keep this as cache, not a database table. Prefer Redis or hybrid shared backends for scaled deployments; document memory backend as local/dev or single-node only. |
| RAG retrieval result cache | router runtime, `src/semantic-router/pkg/extproc/req_filter_rag_cache.go` | Process-wide singleton in-memory LRU with TTL | `ephemeral_request_state` | Restart flushes cache; cache is global per process, not per tenant or replica | Hidden shared mutable state, no observability, no durability, and no multi-replica coherence | Keep as optional cache only. Move to a pluggable shared cache backend if this becomes performance-critical, or document as local process optimization. |
| Agentic memory vectors | router runtime, `src/semantic-router/pkg/memory/**` | Disabled by default; vector content leans on Milvus config when enabled | `shared_durable_workflow_state` when enabled | Depends on backend choice; not enabled by default | Product semantics remain ambiguous between experimental memory and supported user data | Keep vector embeddings in Milvus or another vector store, but pair them with explicit metadata and lifecycle ownership in a durable server-owned contract. |
| Vector store collection registry | router runtime, `src/semantic-router/pkg/vectorstore/{manager.go,local_metadata_registry.go}` | Local metadata registry at `${file_storage_dir}/.vectorstore-metadata.json`; embeddings stay in the configured vector backend | `restart_safe_local_state` by default, carrying `shared_durable_workflow_state` metadata for the vector-store API surface | Store inventory and counts survive restart when the same `file_storage_dir` is mounted; startup reconciles persisted entries against backend collections and drops stale metadata | Metadata is still local to one mounted workspace or node unless the file-storage path itself is shared | Keep the local registry as the default restart-safe seam. If the vector-store API becomes a multi-replica product surface, lift metadata into shared server-owned storage while keeping embeddings in the vector backend. |
| Vector store file registry | router runtime, `src/semantic-router/pkg/vectorstore/{filestore.go,pipeline.go,local_metadata_registry.go}` | File bytes on local disk plus metadata and status records in `${file_storage_dir}/.vectorstore-metadata.json` | `restart_safe_local_state` by default with `shared_durable_workflow_state` semantics for file lifecycle metadata | Files, file inventory, and terminal ingestion statuses survive restart; `in_progress` ingestions are recovered as `failed` with an `interrupted` error so operators get a typed terminal state instead of silent loss | Metadata is still local to the mounted file-storage path and is not shared across replicas automatically | Keep file bytes and metadata separated, and preserve the restart-recovery contract. Move metadata or status records into shared storage only when multi-replica ownership is required. |
| Startup readiness and model download progress | router runtime, `src/semantic-router/pkg/startupstatus/status.go` | JSON file in config dir if writable, otherwise temp-owned path | `audit_analytics_telemetry` implemented as local temp/file convention | Dashboard may lose progress history or read from temp fallback paths | Status semantics depend on writable paths and log interpretation instead of typed lifecycle records | Replace temp/file guessing with typed status records or events owned by the server control plane. Keep local file output only as a local-dev adapter. |
| Model-selection Elo ratings | router runtime, `src/semantic-router/pkg/selection/elo.go`, `src/semantic-router/pkg/selection/storage.go` | JSON file autosave when configured; otherwise in-memory | `shared_durable_workflow_state` for learning systems | Defaults to local file or process memory; not replica-safe | Online learning state diverges across replicas and restarts | Persist ratings in a shared durable store when model selection is productized. Local JSON is acceptable only for experiments and local smoke paths. |
| RL-driven selector preferences and session context | router runtime, `src/semantic-router/pkg/selection/rl_driven.go` | In-memory maps plus optional file-backed Elo storage for part of the state | Mixed `shared_durable_workflow_state` and `ephemeral_request_state` | Global, user, category, and session preference state is not fully durable or shared | Learning quality and routing behavior drift between replicas; user/session affinity becomes nondeterministic | Define one shared persistence seam for user, category, and session learning state before treating RL selection as production-ready. |
| GMTRouter personalization state | router runtime, `src/semantic-router/pkg/selection/gmtrouter.go` | Local JSON file via `StoragePath` | `shared_durable_workflow_state` implemented as local file | Restart-safe only within one local path; not shared | Multi-instance deployments split personalization state and make recovery manual | Move to a shared durable store or explicitly keep this experimental/local-only. |
| Tools database | router integrations, `global.integrations.tools`, `config/tools_db.json` | JSON file path by default | `restart_safe_local_state` | Survives only as workspace file | Multi-user editing, audit, and HA are weak | If dashboard editing becomes first-class, add a durable metadata store or projection for tools; keep JSON as import/export and local-dev source. |
| Dashboard auth users, roles, permissions, audit logs | dashboard backend auth, `dashboard/backend/auth/store.go` | SQLite at `./data/auth.db` by default | `restart_safe_local_state` | Restart-safe in one workspace, not shared across replicas | Adequate for local stacks, weak for HA or multi-instance deployments | Keep SQLite for local dev. Browser auth now rides a server-owned `HttpOnly` session cookie on top of this store; add a relational production storage seam if multi-instance HA becomes a product requirement. |
| Browser auth session | dashboard frontend plus backend auth, `dashboard/frontend/src/utils/authFetch.ts`, `dashboard/backend/auth/{http_handlers.go,middleware.go}` | Server-set `HttpOnly` `vsr_session` cookie carrying a signed session token; no `localStorage` mirror or query-token fallback | `restart_safe_local_state` for the local dashboard control plane, with server-enforced browser access control | Session survives browser restarts until expiry and is revalidated against current user status or permissions on each request; session storage is still not replica-shared beyond the local dashboard auth store | JWT signing remains stateless and the underlying auth store still defaults to a local SQLite file | Keep the browser session server-owned and same-origin. If production HA or external identity federation becomes a requirement, layer OIDC and a shared auth DB or session store on top without returning to browser-owned token transport. |
| Evaluation task metadata | dashboard backend evaluation, `dashboard/backend/evaluation/db.go` | SQLite | `restart_safe_local_state` moving toward `shared_durable_workflow_state` | Tasks survive local restart in one workspace | HA and shared-operator workflows are limited to one mounted DB file | Keep task metadata in relational storage. SQLite is fine for local dev; add a production DB seam for shared deployments. |
| Evaluation progress fanout and cancellation | dashboard handlers, `dashboard/backend/handlers/evaluation.go` | In-memory `sync.Map` of SSE clients and cancel funcs | `ephemeral_request_state` | Active streams and cancel handles vanish on restart | Acceptable for live connections, but status history depends on the DB and in-flight runner state | Keep connection registries in memory, but ensure all progress and terminal states are durable and reconstructable from server-owned records. |
| ML pipeline jobs | dashboard backend ML pipeline, `dashboard/backend/mlpipeline/runner.go`, `dashboard/backend/handlers/mlpipeline.go` | In-memory job map and progress channel; output files on disk | Mixed `shared_durable_workflow_state` and `ephemeral_request_state` | Outputs may survive on disk, but job state and progress vanish on restart | Long-running jobs are not restart-aware; multi-node control is impossible | Move job metadata, lifecycle, and progress into durable workflow tables. Keep SSE client maps in memory only for delivery. |
| Model research campaigns | dashboard backend model research, `dashboard/backend/modelresearch/manager.go` | JSON state files plus in-memory map and event channel | `restart_safe_local_state` | Running campaigns are marked failed after dashboard restart | Campaign recovery semantics are lossy and single-node only | Persist campaign metadata and events in a durable workflow store. Keep artifacts in files/object storage, but make workflow state restart-aware. |
| OpenClaw container registry | dashboard backend OpenClaw, `dashboard/backend/handlers/openclaw.go`, CLI `src/vllm-sr/cli/docker_services.py` | JSON file under OpenClaw data dir | `restart_safe_local_state` | Registry survives in one workspace, not shared across dashboards | Container control depends on local file convention and workspace mounts | Keep this as local-dev adapter for `vllm-sr serve`, but introduce a server-owned registry for multi-user dashboard control. |
| OpenClaw teams and workers | dashboard backend OpenClaw, `dashboard/backend/handlers/openclaw.go`, `openclaw_teams.go`, `openclaw_workers.go` | JSON files under OpenClaw data dir | `shared_durable_workflow_state` implemented as local files | Restart-safe per workspace only | Collaboration semantics, permissions, and audits are weak | Persist team and worker entities in a database. Keep generated config files as derived runtime artifacts, not the only source of truth. |
| OpenClaw rooms and chat messages | dashboard backend OpenClaw, `dashboard/backend/handlers/openclaw_rooms.go` | JSON files rewritten on each append; SSE/WS client maps in memory | `shared_durable_workflow_state` implemented as local files | Restart-safe per workspace only; live client state is ephemeral | Message append path is O(n) per write and unsuitable for large rooms or long histories | Persist room and message metadata in a database. Keep live transport state in memory only. Consider object/blob storage only for large attachments, not message metadata. |
| Dashboard chat history in Playground and similar surfaces | dashboard frontend, `dashboard/frontend/src/hooks/useConversationStorage.ts`, `dashboard/frontend/src/components/ChatComponent.tsx` | Browser `localStorage` | `ephemeral_request_state` from the product-contract perspective, with browser-local convenience persistence only | Persists only in one browser and is intentionally not reconstructed by the server | No cross-device continuity, audit, or server recovery by design | Keep this as browser-local convenience state for demo and local operator workflows unless the product explicitly promotes chat history to a supported shared surface. If that happens later, move it into server-owned workflow storage instead of widening browser-local state. |
| Config backups, generated runtime config, DSL snapshots | CLI and dashboard, `src/vllm-sr/cli/commands/runtime_support.py`, `dashboard/backend/handlers/{config_backups.go,runtime_config_sync.go,config_projection.go}` | Files under `.vllm-sr/`, including backups, `config.dsl`, and `active-config-projection.json` | `restart_safe_local_state` and `derived_projection_state` | Survive in one workspace; the active projection is regenerated on apply or activate and hash-checked before setup-summary reads trust it | Workspace-local files still blur source-of-truth boundaries in multi-user deployments | Keep YAML and DSL as canonical intent and keep the projection derived. If a multi-replica dashboard needs shared queryability later, add shared audit or read-model storage without introducing a second mutable primary writer. |
| Active deployed models, signals, decisions, plugins, DSL parse results | shared config contract, dashboard topology/config APIs, `src/semantic-router/pkg/routerprojection/**`, `dashboard/backend/handlers/{config_projection.go,setup.go}` | Persisted JSON projection at `.vllm-sr/active-config-projection.json`, derived from canonical YAML plus archived DSL source when available | `derived_projection_state` | Survives restart inside one workspace and avoids ad hoc reparsing; the dashboard only trusts it when `validation.valid` is true and the stored config hash matches the current YAML | Projection storage is still workspace-local until lifted behind a shared control-plane store | Keep the projection derived from YAML or DSL rather than making it a mutable primary source. If multi-user topology or audit queries need shared access later, move the storage backend behind a server-owned read-model seam. |

## Default Memory-Backed Surfaces To Treat As High Risk

- `global.stores.semantic_cache.backend_type = memory`
- `global.stores.vector_store.backend_type = memory` in dashboard defaults when enabled
- RAG `cache_results` in `src/semantic-router/pkg/config/rag_plugin.go`
- model-selection learning state in `src/semantic-router/pkg/selection/{elo.go,rl_driven.go,gmtrouter.go}`

## What Should Go To A Database First

- User accounts, roles, permissions, and audit logs
- Evaluation, ML pipeline, and model-research job metadata and typed progress events
- OpenClaw teams, workers, rooms, and room messages
- Router-visible metadata for vector stores and uploaded files
- Response API conversations and response metadata if they are exposed as supported product features
- Config version history, active config projection, deployed model/signal/decision/plugin projection, and DSL snapshots

## What Should Prefer Shared Cache Or Specialized Storage Instead Of A Database

- Semantic cache entries: shared cache such as Redis or the existing hybrid cache path
- RAG retrieval cache: shared cache if retained at all
- Vector embeddings and memory embeddings: vector backends such as Milvus or Llama Stack
- Large binary artifacts: local file/object storage with durable metadata in a database

## Progressive Migration Order

1. Publish and keep this inventory current.
2. Keep router metadata and replay or response history restart-safe where the product already exposes those surfaces.
   Response API and replay now default to durable Redis or Postgres backends, while vector-store metadata now persists under `${file_storage_dir}/.vectorstore-metadata.json` and recovers interrupted ingestion with typed terminal status.
3. Keep at least one long-running dashboard workflow server-owned and restart-safe.
   Evaluation task metadata remains SQLite-backed and queryable after restart; live SSE fanout stays ephemeral by design.
4. Keep a persisted deployed-config projection so dashboard and future APIs do not need to reparse YAML and DSL on every query path.
   `.vllm-sr/active-config-projection.json` now stores models, signals, decisions, plugins, validation, and a DSL snapshot derived from canonical YAML and archived DSL source.
5. Keep browser auth server-owned and treat browser-local chat history as explicit convenience state rather than reopening another umbrella debt for those already-classified surfaces.
