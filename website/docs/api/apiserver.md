# Router management API

The router management API provides health, classification, configuration,
storage, cache, compression, and replay operations. It listens on port `8080`
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
| `GET` | `/api/v1` | Registered endpoint discovery |
| `GET` | `/openapi.json` | Generated OpenAPI schema, optionally narrowed by exact `path` and `method` |
| `GET` | `/docs` | Swagger UI |

Use `/health` for liveness and `/ready` for readiness. During model download or
runtime preparation, a process can be healthy while `/ready` still returns
`503`.

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

## Manage knowledge bases and stored data

Knowledge-base configuration:

| Method | Path | Use |
| --- | --- | --- |
| `GET`, `POST` | `/api/v1/storage/knowledge-bases` | List or create managed knowledge bases |
| `GET`, `PUT`, `DELETE` | `/api/v1/storage/knowledge-bases/{name}` | Read, update, or delete one knowledge base |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/metadata` | Read generated map metadata |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/data.ndjson` | Stream map data as NDJSON |

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

## Operate the response cache

Response-cache endpoints are separate from inference-time cache lookup. They
let operators inspect the backend, test a candidate configuration, and perform
audited invalidation.

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/api/v1/response-cache/capabilities` | Backend capabilities |
| `GET` | `/api/v1/response-cache/health` | Backend health |
| `GET` | `/api/v1/response-cache/stats` | Redacted statistics |
| `GET` | `/api/v1/response-cache/audit` | Redacted mutation audit entries |
| `POST` | `/api/v1/response-cache/test` | Validate and probe a candidate configuration |
| `POST` | `/api/v1/response-cache/invalidate` | Dry-run or invalidate a scoped partition |
| `POST` | `/api/v1/response-cache/flush` | Advance a scoped or global cache epoch |

Prefer scoped invalidation and a dry run before a destructive cache mutation.
Bearer roles distinguish read, invalidate, and broader cache-management
permissions.

## Inspect context compression

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/api/v1/context-compression/capabilities` | Runtime capabilities |
| `GET` | `/api/v1/context-compression/health` | Runtime health |
| `GET` | `/api/v1/context-compression/stats` | Redacted statistics |
| `POST` | `/api/v1/context-compression/preview` | Preview compression without persistence |
| `POST` | `/api/v1/context-compression/recovery/invalidate` | Invalidate a trusted recovery scope |

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

<!-- BEGIN-GENERATED-ENDPOINT-INDEX -->
### system

Health, readiness, and API contract discovery.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/ready` | Readiness endpoint that turns green only after startup completes |
| `GET` | `/startup-status` | Detailed router startup and model-download status |
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
| `POST` | `/api/v1/routing/preview` | Preview all configured signals and the resulting route without invoking a generation backend |

### inventory

Inspect configured and loaded model and classifier resources.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/inventory/models` | Get information about loaded models |
| `GET` | `/api/v1/inventory/classifier` | Get classifier information and status (secrets redacted without secret_view) |
| `GET` | `/api/v1/inventory/embedding-models` | Get information about loaded embedding models |
| `GET` | `/v1/models` | OpenAI-compatible public model and Entrypoint listing |

### observability

Inspect routing replays and metrics, and submit outcome evidence.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/observability/classification-metrics` | Get classification metrics and statistics |
| `POST` | `/api/v1/observability/outcomes` | Submit Router Learning outcome feedback linked to a replay record |
| `GET` | `/api/v1/observability/replays` | List Router Replay records |
| `GET` | `/api/v1/observability/replays/aggregate` | Aggregate Router Replay routing and cost metadata |
| `GET` | `/api/v1/observability/replays/trajectory` | Build a Router Replay session trajectory |
| `GET` | `/api/v1/observability/replays/{id}` | Read one Router Replay record |

### storage

Manage Router-owned knowledge bases, memories, files, and vector stores.

| Method | Path | Description |
| --- | --- | --- |
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

### response-cache

Inspect and manage the response-cache service.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/response-cache/capabilities` | Get response-cache backend capabilities |
| `GET` | `/api/v1/response-cache/health` | Check response-cache backend health |
| `GET` | `/api/v1/response-cache/stats` | Get redacted response-cache statistics |
| `GET` | `/api/v1/response-cache/audit` | Get redacted response-cache mutation audit entries |
| `POST` | `/api/v1/response-cache/test` | Validate and probe a response-cache candidate configuration |
| `POST` | `/api/v1/response-cache/invalidate` | Dry-run or invalidate a scoped response-cache partition |
| `POST` | `/api/v1/response-cache/flush` | Advance a scoped or global response-cache epoch |

### context-compression

Inspect, preview, and manage context compression.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/context-compression/capabilities` | Get context-compression capabilities |
| `GET` | `/api/v1/context-compression/health` | Check context-compression runtime health |
| `GET` | `/api/v1/context-compression/stats` | Get redacted context-compression statistics |
| `POST` | `/api/v1/context-compression/preview` | Preview context compression without persistence |
| `POST` | `/api/v1/context-compression/recovery/invalidate` | Invalidate a trusted context-recovery request scope |

### diagnostics

Invoke low-level classifiers, embeddings, NLI, and similarity diagnostics.

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
<!-- END-GENERATED-ENDPOINT-INDEX -->
