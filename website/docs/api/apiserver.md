# Router management API

The router management API provides configuration, routing previews, plugin
inspection, model diagnostics, storage, and observability. It listens on port `8080`
by default and the local stack binds it to `127.0.0.1`.

For model traffic, use the configured Envoy listener described in
[Router API](./router).

## Start with the live schema

The running router generates its endpoint discovery and OpenAPI document from
the routes it has registered. Use these pages for registered methods, path and
query parameters, request-body fields, access policy, and response media:

| Path | Purpose |
| --- | --- |
| `GET /api/v1` | Endpoint discovery with permission and sensitivity metadata |
| `GET /openapi.json` | Complete OpenAPI 3.0 document |
| `GET /openapi.json?path=...&method=...` | One valid path or operation document |
| `GET /docs` | Interactive Swagger UI |

This page groups the API by user task. The live OpenAPI document is the
field-level source of truth for the version you are running.

```bash
curl -sS http://localhost:8080/health
curl -sS http://localhost:8080/openapi.json
curl -sS 'http://localhost:8080/openapi.json?path=/api/v1/config&method=PATCH'
```

Agents should call these Router endpoints directly; the Dashboard is not part
of the discovery path. The Website also renders the generated contract in the
[searchable OpenAPI reference](./openapi).

## Resource boundaries

| Capability | Responsibility |
| --- | --- |
| `config` | Canonical configuration, recipes, validation, planning, and activation |
| `routing` | Evaluate a request through the configured routing pipeline |
| `plugins` | Discover plugin types, inspect recipe/decision bindings, and preview or probe behavior |
| `inventory` | Inspect configured and prepared runtime resources |
| `diagnostics` | Invoke a specific prepared model or a routing classifier |
| `storage` | Manage stored data, response-cache partitions, and context-recovery scopes |
| `observability` | Read metrics, replays, and management audit; submit outcome evidence |
| `system` | Health, startup, readiness, and API discovery |

Plugin configuration belongs to its recipe decision in the canonical document.
Use `/api/v1/config/recipes/{name}` with `If-Match` to change it. There is no second
plugin configuration database or independent plugin enable/disable state.
`/config/router` is not registered; the canonical configuration API is
`/api/v1/config`.

## Access and authentication

The local CLI keeps the management port on loopback. For a remote router,
prefer a private network or an SSH tunnel instead of publishing the port:

```bash
ssh -N -L 8080:127.0.0.1:8080 router-host
```

Management authentication is disabled unless configured. To require bearer
tokens, set `global.services.management_api.auth.mode: bearer` and define roles
and token sources in the management API configuration. Then send:

```http
Authorization: Bearer <token>
```

`GET /health` remains public. Other routes enforce their assigned permission
when bearer authentication is enabled. Configuration and replay responses can
also redact sensitive fields unless the principal has the corresponding detail
permission.

## Health and discovery

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/health` | Process liveness |
| `GET` | `/ready` | Whether startup has completed |
| `GET` | `/startup-status` | Startup and model-download progress |
| `GET` | `/api/v1/status` | Versioned replica-local startup and configuration observations |
| `GET` | `/api/v1` | Registered endpoint discovery |
| `GET` | `/openapi.json` | Generated OpenAPI schema, optionally narrowed by exact `path` and `method` |
| `GET` | `/docs` | Swagger UI |

Use `/health` for liveness and `/ready` for readiness. During model download or
runtime preparation, a process can be healthy while `/ready` still returns
`503`.

### Replica-local status

`GET /api/v1/status` requires `ready.read`. It returns `schema_version: "v1"`,
an opaque `instance_id`, an observation timestamp, and conditions with `True`,
`False`, or `Unknown` status. The instance identity changes when the Router
registry is recreated.

The response reads in-process startup observations and the active configuration
registry. It does not treat a shared Redis value as this replica's state.
Configuration hashes identify the active snapshot and the most recent candidate
observed by the activation coordinator. They do not claim to describe an
unobserved external desired configuration.

| Condition | Meaning |
| --- | --- |
| `Live` | The process answered the request |
| `StartupComplete` | The local startup writer reported completion or failure |
| `ActiveConfigAvailable` | The registry has a classification runtime that can still be acquired |
| `ConfigConverged` | The active and latest observed configuration hashes match |
| `Ready` | Known startup/runtime failures are false; required dependency and quorum readiness is otherwise unknown |

The stable reason codes are `ProcessRunning`, `StartupUnobserved`,
`StartupIncomplete`, `StartupFailed`, `StartupComplete`, `NoActiveConfig`,
`ActiveConfigAvailable`, `ConfigIdentityUnobserved`, `ConfigHashesMatch`,
`ConfigActivationPending`, `ConfigActivationFailed`,
`ConfigActivationSuperseded`, and `RequiredDependenciesUnobserved`.
Activation stages are diagnostic metadata, not additional readiness conditions.

A failed reload reports `ConfigActivationFailed` without discarding the
last-known-good active hash. Successful publication changes the active hash and
reports `ConfigHashesMatch`. Raw startup messages, credentials, and activation
failure details are excluded.

This first contract does not evaluate every required dependency, recipe
capability, or quorum. `RequiredDependenciesUnobserved` must not be treated as
ready. HTTP `200` means the status document was retrieved, not that traffic can
be served. The existing `/health`, `/ready`, and `/startup-status` contracts are
unchanged; this endpoint does not replace their deployment probes. Fleet
aggregation, per-replica persistence/TTL, and probe wiring remain separate work.

## Inspect signals without an inference call

The classification endpoints are useful when tuning signals or diagnosing why
a decision did not match. They do not call a generation backend.

```bash
curl -sS http://localhost:8080/api/v1/diagnostics/classify/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a Python function that merges two sorted lists."}'
```

| Method | Path | Use |
| --- | --- | --- |
| `POST` | `/api/v1/diagnostics/classify/intent` | Evaluate intent/domain routing |
| `POST` | `/api/v1/diagnostics/classify/pii` | Detect configured PII types |
| `POST` | `/api/v1/diagnostics/classify/security` | Evaluate jailbreak and security classification |
| `POST` | `/api/v1/diagnostics/classify/fact-check` | Decide whether text needs fact checking |
| `POST` | `/api/v1/diagnostics/classify/user-feedback` | Classify user feedback |
| `POST` | `/api/v1/diagnostics/classify/combined` | Run intent, PII, and security classification |
| `POST` | `/api/v1/diagnostics/classify/batch` | Run a selected classifier over a batch |
| `POST` | `/api/v1/routing/preview` | Evaluate all configured signals |
| `POST` | `/api/v1/diagnostics/nli` | Evaluate a premise/hypothesis pair |
| `POST` | `/api/v1/diagnostics/embeddings` | Generate configured text or image embeddings |
| `POST` | `/api/v1/diagnostics/similarity` | Compare a text pair |
| `POST` | `/api/v1/diagnostics/similarity/batch` | Run batch similarity matching |

Names, scores, and matched rules depend on the active recipe. Use the live
schema for each endpoint's supported input forms.

When the matched decision uses `fast_response`, Preview reports
`selection_status: not_required` and `selection_method: fast_response`, with no
`selected_model`. This immediate response needs no model assignment or candidate
capability/context admission. The client-facing response model identifier does
not imply that a generation backend was selected or called.

Guard and PII report `input_limit` in `signal_errors` when input exceeds their
configured inference budget. Check the effective model and deployment limits
before retrying. Other inference failures retain their bounded signal error
codes; the configured unknown-signal policy determines the route outcome.

Standalone classification, embedding, and similarity diagnostics return
`400 INVALID_INPUT` when inference reports that a model's input budget was
exceeded. The error retains the model's limit details; other inference failures
remain server errors.

## Diagnose a prepared model binding

`GET /api/v1/diagnostics/models?recipe=<name>` lists the bindings actually
prepared for that recipe. Select a returned `name` and its `contract`; the
Router does not infer a model from a family name or silently load a fallback.
Both `recipe` and `binding` are required for the typed invocation endpoints.

| Operation under `/api/v1/diagnostics/models` | Prepared task | Vela use |
| --- | --- | --- |
| `POST /labels` | Label distribution, including configured window scanning | Domain, Guard, FactCheck, Feedback, Modality |
| `POST /label-scores` | Independent label scores and configured operating point | Safety, Hazard |
| `POST /tokens` | Token spans with original UTF-8 byte offsets | PII |
| `POST /embeddings` | The binding's configured embedding representation | Embedding |
| `POST /rerank` | Query/document relevance scores in input order | Reranker |

For example, take a label-distribution binding name from discovery:

```bash
curl -sS 'http://localhost:8080/api/v1/diagnostics/models?recipe=default'
curl -sS http://localhost:8080/api/v1/diagnostics/models/labels \
  -H 'Content-Type: application/json' \
  -d '{"recipe":"default","binding":"<prepared-binding-name>","text":"Explain this Python error."}'
```

Every invocation returns the actual binding identity, artifact revision,
provider, device, precision, and task limits alongside the result. Calls retain
the owning runtime generation until inference finishes, including during reload.
An unprepared or foreign-recipe binding fails explicitly. Windowed Guard and
PII keep their prepared scan geometry; Hazard keeps its operating-point
thresholds and policy digest. The shared Vela Encoder is an artifact used by
task bindings, so it does not need a separate inference endpoint.

The convenience classification, embedding, NLI, and similarity endpoints also
accept an optional `recipe`. Omitting it retains their default-recipe behavior;
a supplied recipe is resolved explicitly. The typed model endpoints are the
preferred way to verify exactly which loaded model produced a result. Rerank
batches obey `global.services.api.batch_classification.max_batch_size` (100 pairs when
unset). Typed calls share their prepared resource's admission gate and have a
two-minute request deadline; native inference retains its lease until it
actually returns, even after cancellation. Actual coverage follows active configuration: discovering a task does not load it.

## Inspect models and metrics

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/api/v1/inventory/models` | Loaded model inventory |
| `GET` | `/api/v1/inventory/classifier` | Classifier configuration and status |
| `GET` | `/api/v1/inventory/embedding-models` | Loaded embedding models |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `GET` | `/api/v1/observability/classification-metrics` | Classification counters and timing |

Secrets in classifier information are redacted unless the caller has
`secret_view`.

The model inventory reports successfully prepared task bindings in the active
runtime generation, including each binding's `recipe` and effective provider,
device, precision, and input limit in `metadata`. Shared artifacts may appear
under several recipe bindings. Configured but unused models are not marked ready.
During startup, the inventory can instead report pending artifact downloads.
`system.gpu_available` means an active prepared binding uses local GPU execution;
it does not indicate whether the host has unused GPU hardware.

## Read and change router configuration

Read the current canonical document and its `ETag` before making a change:

```bash
curl -i http://localhost:8080/api/v1/config \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}"
```

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/api/v1/config` | Read the active canonical configuration |
| `POST` | `/api/v1/config/validate` | Validate and normalize without writing |
| `POST` | `/api/v1/config/plan` | Plan the exact candidate and return the current/candidate ETags without writing |
| `PATCH` | `/api/v1/config` | Merge, validate, persist, and hot-reload an update |
| `PUT` | `/api/v1/config` | Replace, validate, persist, and hot-reload the document |
| `GET` | `/api/v1/config/versions` | List configuration backups |
| `POST` | `/api/v1/config/rollback` | Restore a backup |
| `GET` | `/api/v1/config/hash` | Compare persisted, generated, and active hashes |

Recipe operations use the same canonical document:

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/api/v1/config/recipes` | List default and named recipes and their entrypoints |
| `POST` | `/api/v1/config/recipes/validate` | Validate a recipe mutation without applying it |
| `GET` | `/api/v1/config/recipes/{name}` | Read one recipe |
| `PUT` | `/api/v1/config/recipes/{name}` | Create or replace one recipe |
| `DELETE` | `/api/v1/config/recipes/{name}` | Delete an unreferenced named recipe |

Every config mutation, including rollback and Recipe `PUT`/`DELETE`, requires
the exact current `ETag` in `If-Match`. The Router does not accept unguarded
writes. A mutation response and `GET /api/v1/config/hash` use the same explicit
runtime identity fields: `source_config_hash`, `generated_runtime_hash`,
`active_runtime_hash`, and `activation_status`. Config mutations validate,
create a backup, and trigger reload; an active config still does not prove that
upstream model backends are healthy. Check `/ready` and send a representative
request after a change.

`activation_status` is `active`, `pending`, `failed`, or `unknown`. When a
candidate has been attempted, `activation` includes its document hash, attempt,
stage, timestamps, and a redacted failure detail. A failed candidate leaves the
previous generation active. Config mutation responses return `503` with
`status: activation_failed` when that failure is observed during the activation
wait: the document was persisted, so read its current `ETag` and inspect or
roll back that candidate instead of blindly retrying the write. The status is
process-local and describes the latest attempted candidate. A newer file
supersedes an older attempt.

Tracing settings are initialized at process startup. Config plans, updates,
and rollbacks that change `global.services.observability.tracing` return
`409 RESTART_REQUIRED` without persisting the candidate. Apply those changes
through the deployment workflow and restart the Router.

## Manage knowledge bases and stored data

Knowledge-base configuration:

| Method | Path | Use |
| --- | --- | --- |
| `GET`, `POST` | `/api/v1/storage/knowledge-bases` | List or create managed knowledge bases |
| `GET`, `PUT`, `DELETE` | `/api/v1/storage/knowledge-bases/{name}` | Read, update, or delete one knowledge base |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/metadata` | Read generated map metadata |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/data.ndjson` | Stream map data as NDJSON |

In a router process, create, update, and delete persist a candidate and return
`202` with `activation_status: pending` and `generated_runtime_hash` while the
replacement generation prepares. Poll `/api/v1/config/hash` until
`active_runtime_hash` matches that candidate, or stop on `activation_status: failed` and inspect `activation`. A failure already observed at mutation time
returns `503` with the persisted candidate's activation detail. A second KB mutation while pending
returns `409` with `CONFIG_ACTIVATION_PENDING` and does not overwrite it.
Updates use independent asset revision paths; deletion removes the candidate
config entry while retaining files needed by old and rollback generations.
Old revisions require offline cleanup after their configuration references
have been retired. Standalone API servers report `activation_status: unknown`
with their normal success status because no router generation registry is present.

Router-managed storage and memory:

| Resource | Base path | Operations |
| --- | --- | --- |
| Long-term memory | `/api/v1/storage/memories` | List and delete by scope; read or delete by id |
| Vector stores | `/api/v1/storage/vector-stores` | Create, list, read, update, delete, and search |
| Vector-store files | `/api/v1/storage/vector-stores/{id}/files` | Attach, list, inspect, and detach files |
| Files | `/api/v1/storage/files` | Upload, list, inspect, download, and delete |

These routes return `503` when their required service is unavailable. File
upload uses multipart form data and accepts documents (`.txt`, `.md`, `.json`,
`.csv`, `.html`) for vector-store ingestion; upload an image (`.png`, `.jpg`,
`.jpeg`, `.gif`, `.webp`) with `purpose=vision` to reference it from a Response
API `input_image` part by `file_id`. Consult the live schema for limits and
fields. They exist only on the management listener. `/v1/files` and
`/v1/vector_stores` are not inference-listener aliases and are not registered
by the Router API.

## Discover and inspect plugins

`GET /api/v1/plugins` projects the canonical plugin registry, including each
plugin's schema URL, bindings URL, and supported operations. It includes all
registered types, including plugins without a separate runtime operation.
`GET /api/v1/plugins/{type}` selects one descriptor;
`GET /api/v1/plugins/{type}/bindings?recipe=<name>&decision=<name>` inspects active
bindings, declared enablement, reachability, and dependency availability.
Availability is not a network health probe; `runtime_inspected: false` means
that only configuration was available for inspection. Configuration links point back to
the owning recipe. An empty binding list means the plugin is not configured in
the selected scope.

Operation metadata distinguishes `read`, `preview`, `probe`, and `mutation`.
A descriptor links to shared storage or observability resources where applicable;
it does not duplicate their state under a plugin-specific resource tree.

| Plugin behavior | Operation |
| --- | --- |
| `system_prompt`, `request_params`, `header_mutation`, `fast_response` | `POST /api/v1/plugins/{type}/preview` using a typed candidate `configuration` or an active `binding` |
| `hallucination`, `response_jailbreak` | Preview an active policy and its supplied conditions; `mode: probe` invokes its configured detectors |
| `rag` | Preview context injection with `supplied_context`; `mode: probe` performs configured retrieval without the RAG result cache |
| `tools`, `tool_selection` | Preview static tool policy; semantic selection requires explicit `mode: probe` |
| `context_compression` | Preview retained targets and budgets using the existing compression contract |
| `response_cache`, `memory` | Operate their resources under `/api/v1/storage` |
| `router_replay`, `shadow_dispatch` | Inspect recorded outcomes under `/api/v1/observability/replays` |

The guard, RAG, and tool previews require an explicit
`binding: {"recipe":"<name>","decision":"<name>"}`. Their default `mode` is
`preview`; they do not automatically escalate to a probe. A probe may invoke a
configured local or remote classifier, embedding service, or retrieval backend.
RAG previews require `data.read`, tool-policy previews require `config.read`,
and guard probes require `classify.invoke`. These operations never execute a
selected tool or generate a provider response. Returned
`mode`, `persisted`, and `backend_calls` describe the operation: `persisted: false` means it does not write plugin data or conversation state; a probe may
update runtime metrics or embedding memoization, and a configured external
retrieval service may have its own effects. Header values in previews require
`secret_view`.

## Operate the response cache

Response-cache endpoints are separate from inference-time cache lookup. They
let operators inspect the backend, test a candidate configuration, and perform
audited invalidation.

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/api/v1/storage/response-cache/capabilities` | Backend capabilities |
| `GET` | `/api/v1/storage/response-cache/health` | Backend health |
| `GET` | `/api/v1/storage/response-cache/stats` | Redacted statistics |
| `POST` | `/api/v1/storage/response-cache/test` | Validate and probe a candidate configuration |
| `POST` | `/api/v1/storage/response-cache/invalidate` | Dry-run or invalidate a scoped partition |
| `POST` | `/api/v1/storage/response-cache/flush` | Advance a scoped or global cache epoch |

Prefer scoped invalidation and a dry run before a destructive cache mutation.
Bearer roles distinguish read, invalidate, and broader cache-management
permissions.

## Inspect management audit

`GET /api/v1/observability/audit` requires `audit.read` and covers audited
management operations across config, recipes, data, cache, and compression.
Use an exact `action` filter, `limit` (1–1000, default 100), and `after_sequence`
for pagination. Continue with the returned `next_sequence` and keep the same
filter. `has_more` indicates more matching entries; `truncated` means the
requested cursor predates retained data.

The hash-chained ring retains the latest 10,000 entries in this process.
`oldest_sequence` identifies its retention boundary; the oldest entry can refer
to a predecessor outside the ring. Restart resets the log and its sequence, so
this endpoint is not a durable compliance archive. Audit records contain route
and authorization metadata, not request bodies or plugin payloads.

## Inspect context compression

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/api/v1/plugins/context_compression/capabilities` | Runtime capabilities |
| `GET` | `/api/v1/plugins/context_compression/health` | Runtime health |
| `GET` | `/api/v1/observability/plugins/context_compression/stats` | Redacted statistics |
| `POST` | `/api/v1/plugins/context_compression/preview` | Preview compression without persistence |
| `POST` | `/api/v1/storage/context-recovery/invalidate` | Invalidate a trusted recovery scope |

Use `preview` to evaluate what would be retained before enabling compression on
important traffic.

## Inspect replay and submit outcomes

Router Replay is management-only. Its query endpoints and redaction model are
described in [Router API](./router#router-replay).

Router Learning can ingest an outcome linked to an owned replay record:

```bash
curl -sS http://localhost:8080/api/v1/observability/outcomes \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: feedback-123' \
  -d '{
    "replay_id": "replay_...",
    "target": "model",
    "verdict": "good_fit",
    "score": 0.9
  }'
```

`replay_id`, `target`, and `verdict` are required. The authenticated principal,
not the optional `source` body field, determines provenance. Use a stable
`Idempotency-Key` when a client may retry. Ingestion also requires an active
Router Learning runtime and the `learning.ingest` permission when bearer auth
is enabled.

## API boundaries

- The management API is an operational surface, not the public inference
  gateway.
- Endpoint availability can depend on compiled features and enabled services.
- The OpenAPI document describes shape, not the behavior of a particular
  model, store, or external backend.
- Keep bearer tokens out of URLs and logs. Give automation only the permissions
  it needs.

## Complete endpoint index

The following reference is generated from the Router's registered route
catalog. Use it to scan every endpoint; use the task-oriented sections above
for guidance and the running `/openapi.json` for exact schemas.

The former `/api/v1/response-cache/*` and
`/api/v1/context-compression/*` routes are retired. Clients should rediscover
operations from `/api/v1` or `/api/v1/plugins`: cache operations now use
`/api/v1/storage/response-cache/*`; compression capabilities, health, and preview
use `/api/v1/plugins/context_compression/*`; statistics, recovery invalidation,
and audit use the observability/storage paths above. No redirect or legacy
handler changes a mutation's meaning.

<!-- BEGIN-GENERATED-ENDPOINT-INDEX -->
### system

Health, readiness, and API contract discovery.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/ready` | Readiness endpoint that turns green only after startup completes |
| `GET` | `/startup-status` | Detailed router startup and model-download status |
| `GET` | `/api/v1/status` | Versioned replica-local startup and configuration status; not a deployment probe |
| `GET` | `/api/v1` | Progressive API capability discovery |
| `GET` | `/openapi.json` | OpenAPI 3.0 specification; optionally narrowed to one path or operation |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### config

Validate, inspect, apply, version, and roll back Router configuration and Recipes.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/config/recipes` | List the default and named routing recipes with their entrypoints |
| `POST` | `/api/v1/config/recipes/validate` | Validate a recipe mutation without writing or reloading config |
| `GET` | `/api/v1/config/recipes/{name}` | Read one routing recipe and its entrypoints |
| `PUT` | `/api/v1/config/recipes/{name}` | Atomically create or replace one routing recipe; requires If-Match |
| `DELETE` | `/api/v1/config/recipes/{name}` | Delete an unreferenced named routing recipe; requires If-Match |
| `GET` | `/api/v1/config/schema` | Discover the canonical Router configuration contract progressively or return the complete JSON Schema |
| `GET` | `/api/v1/config` | Get the current router config as JSON (secrets redacted without secret_view) |
| `POST` | `/api/v1/config/validate` | Validate and normalize a router config without writing it |
| `POST` | `/api/v1/config/plan` | Plan an exact merge or replace mutation, including hot-reload compatibility, without writing it |
| `PATCH` | `/api/v1/config` | Compare-and-swap merge of a router config update (validates, backs up, writes, triggers hot-reload) |
| `PUT` | `/api/v1/config` | Compare-and-swap replacement of the router config (validates, backs up, writes, triggers hot-reload) |
| `POST` | `/api/v1/config/rollback` | Compare-and-swap rollback to a previous router config version |
| `GET` | `/api/v1/config/versions` | List available router config backup versions |
| `GET` | `/api/v1/config/hash` | Compare persisted source, generated runtime, and active router config hashes |

### routing

Preview routing behavior without invoking a generation backend.

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/routing/preview` | Preview all configured signals and the resulting route without invoking a generation backend. global.services.api.routing_preview controls the request deadline and concurrent worker bound. |

### inventory

Inspect configured and loaded model and classifier resources.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/inventory/models` | Get information about loaded models |
| `GET` | `/api/v1/inventory/classifier` | Get classifier information and status (secrets redacted without secret_view) |
| `GET` | `/api/v1/inventory/embedding-models` | Get information about loaded embedding models |
| `GET` | `/v1/models` | OpenAI-compatible public model and Entrypoint listing |

### observability

Inspect routing replays, metrics, and management audit; submit outcome evidence.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/observability/classification-metrics` | Get classification metrics and statistics |
| `POST` | `/api/v1/observability/outcomes` | Submit Router Learning outcome feedback linked to a replay record |
| `GET` | `/api/v1/observability/replays` | List Router Replay records |
| `GET` | `/api/v1/observability/replays/aggregate` | Aggregate Router Replay routing and cost metadata |
| `GET` | `/api/v1/observability/replays/trajectory` | Build a recipe-scoped session trajectory with each recorded routing result |
| `GET` | `/api/v1/observability/replays/{id}` | Read one Router Replay record |
| `GET` | `/api/v1/observability/audit` | Page through this Router process's bounded management mutation audit; filter by action and resume after a sequence |
| `GET` | `/api/v1/observability/plugins/context_compression/stats` | Get redacted context-compression statistics |

### storage

Manage Router-owned knowledge bases, memories, files, vector stores, cache partitions, and context recovery.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/storage/response-cache/capabilities` | Get response-cache backend capabilities |
| `GET` | `/api/v1/storage/response-cache/health` | Check response-cache backend health |
| `GET` | `/api/v1/storage/response-cache/stats` | Get redacted response-cache statistics |
| `POST` | `/api/v1/storage/response-cache/test` | Validate and probe a response-cache candidate configuration |
| `POST` | `/api/v1/storage/response-cache/invalidate` | Dry-run or invalidate a scoped response-cache partition |
| `POST` | `/api/v1/storage/response-cache/flush` | Advance a scoped or global response-cache epoch |
| `POST` | `/api/v1/storage/context-recovery/invalidate` | Invalidate a trusted context-recovery request scope |
| `GET` | `/api/v1/storage/knowledge-bases` | List configured knowledge bases |
| `POST` | `/api/v1/storage/knowledge-bases` | Create a managed knowledge base |
| `GET` | `/api/v1/storage/knowledge-bases/{name}` | Read a knowledge base |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/metadata` | Read generated knowledge-base map metadata |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/data.ndjson` | Stream generated knowledge-base map data as NDJSON |
| `PUT` | `/api/v1/storage/knowledge-bases/{name}` | Update a managed knowledge base |
| `DELETE` | `/api/v1/storage/knowledge-bases/{name}` | Delete a managed knowledge base |
| `GET` | `/api/v1/storage/memories` | List long-term memories |
| `DELETE` | `/api/v1/storage/memories` | Delete memories by scope |
| `GET` | `/api/v1/storage/memories/{id}` | Read one long-term memory |
| `DELETE` | `/api/v1/storage/memories/{id}` | Delete one long-term memory |
| `POST` | `/api/v1/storage/vector-stores` | Create a vector store |
| `GET` | `/api/v1/storage/vector-stores` | List vector stores |
| `GET` | `/api/v1/storage/vector-stores/{id}` | Read a vector store |
| `POST` | `/api/v1/storage/vector-stores/{id}` | Update a vector store |
| `DELETE` | `/api/v1/storage/vector-stores/{id}` | Delete a vector store |
| `POST` | `/api/v1/storage/vector-stores/{id}/search` | Search a vector store |
| `POST` | `/api/v1/storage/vector-stores/{id}/files` | Attach a file to a vector store |
| `GET` | `/api/v1/storage/vector-stores/{id}/files` | List files attached to a vector store |
| `DELETE` | `/api/v1/storage/vector-stores/{id}/files/{file_id}` | Detach a file from a vector store |
| `POST` | `/api/v1/storage/files` | Upload a file |
| `GET` | `/api/v1/storage/files` | List uploaded files |
| `GET` | `/api/v1/storage/files/{id}` | Read uploaded-file metadata |
| `DELETE` | `/api/v1/storage/files/{id}` | Delete an uploaded file |
| `GET` | `/api/v1/storage/files/{id}/content` | Download uploaded-file content |

### plugins

Discover recipe-scoped plugin bindings, dependencies, and typed behavior previews.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/plugins/context_compression/capabilities` | Get context-compression capabilities |
| `GET` | `/api/v1/plugins/context_compression/health` | Check context-compression runtime health |
| `POST` | `/api/v1/plugins/context_compression/preview` | Preview context compression without persistence |
| `GET` | `/api/v1/plugins` | Discover every registered recipe-scoped plugin, its schema, bindings, and supported operations |
| `GET` | `/api/v1/plugins/{type}` | Describe one canonical plugin type and its supported management operations |
| `GET` | `/api/v1/plugins/{type}/bindings` | Inspect active recipe and decision plugin bindings and published dependency availability; does not probe network health |
| `POST` | `/api/v1/plugins/system_prompt/preview` | Preview system instruction changes using the dispatch protocol codec; no persistence or backend calls |
| `POST` | `/api/v1/plugins/request_params/preview` | Preview the dispatch parameter policy, including blocked fields, defaults, and caps; no persistence or backend calls |
| `POST` | `/api/v1/plugins/header_mutation/preview` | Preview ordered Envoy header operations; values require secret_view; no persistence or backend calls |
| `POST` | `/api/v1/plugins/fast_response/preview` | Preview fixed assistant text before transport encoding; no persistence or backend calls |
| `POST` | `/api/v1/plugins/response_jailbreak/preview` | Preview an active binding's response-jailbreak policy; mode=probe explicitly invokes its configured classifier without generation or persistence |
| `POST` | `/api/v1/plugins/hallucination/preview` | Preview an active binding's hallucination policy and supplied fact-check/context conditions; mode=probe explicitly invokes configured detectors without generation or persistence |
| `POST` | `/api/v1/plugins/rag/preview` | Preview configured RAG context injection using supplied_context; mode=probe retrieves from the configured backend without the RAG result cache or generation |
| `POST` | `/api/v1/plugins/tools/preview` | Preview the configured tools policy; mode=probe permits semantic tool retrieval but never executes tools |
| `POST` | `/api/v1/plugins/tool_selection/preview` | Preview the configured tool_selection policy; mode=probe permits configured retrieval and embedding calls but never executes tools |

### diagnostics

Inspect and invoke recipe-scoped prepared models, classifiers, embeddings, NLI, and rerankers.

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/diagnostics/classify/intent` | Classify user queries into routing categories |
| `POST` | `/api/v1/diagnostics/classify/pii` | Detect personally identifiable information in text |
| `POST` | `/api/v1/diagnostics/classify/security` | Detect jailbreak attempts and security threats |
| `POST` | `/api/v1/diagnostics/classify/fact-check` | Classify if text needs fact-checking |
| `POST` | `/api/v1/diagnostics/classify/user-feedback` | Classify user feedback type (satisfied, need_clarification, wrong_answer, want_different) |
| `POST` | `/api/v1/diagnostics/classify/combined` | Perform combined classification (intent, PII, and security) |
| `POST` | `/api/v1/diagnostics/classify/batch` | Batch classification with configurable task_type parameter |
| `POST` | `/api/v1/diagnostics/nli` | Natural language inference classification for premise and hypothesis pairs |
| `POST` | `/api/v1/diagnostics/embeddings` | Generate text and image embeddings |
| `POST` | `/api/v1/diagnostics/similarity` | Calculate pairwise text similarity |
| `POST` | `/api/v1/diagnostics/similarity/batch` | Calculate batch text-similarity matches |
| `GET` | `/api/v1/diagnostics/models` | List prepared model bindings in an explicitly selected recipe |
| `POST` | `/api/v1/diagnostics/models/labels` | Inspect a prepared label distribution; windowed bindings preserve their configured scan |
| `POST` | `/api/v1/diagnostics/models/label-scores` | Inspect independent label scores using the prepared operating point when configured |
| `POST` | `/api/v1/diagnostics/models/tokens` | Inspect prepared token spans with original UTF-8 byte offsets and complete configured window scanning |
| `POST` | `/api/v1/diagnostics/models/embeddings` | Run the explicitly selected prepared embedding binding at its published representation |
| `POST` | `/api/v1/diagnostics/models/rerank` | Score query-document pairs using the selected prepared relevance binding without running a RAG request; max_batch_size applies (default 100 pairs) |
<!-- END-GENERATED-ENDPOINT-INDEX -->
