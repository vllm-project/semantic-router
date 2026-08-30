# Router management API

The router management API provides health, classification, configuration,
storage, cache, compression, and replay operations. It listens on port `8080`
by default and the local stack binds it to `127.0.0.1`.

For model traffic, use the configured Envoy listener described in
[Router API](./router).

## Start with the live schema

The running router generates its endpoint discovery and OpenAPI document from
the routes it has registered. Use these pages for exact request and response
fields:

| Path | Purpose |
| --- | --- |
| `GET /api/v1` | Endpoint discovery |
| `GET /openapi.json` | OpenAPI 3.0 document |
| `GET /docs` | Interactive Swagger UI |

This page groups the API by user task. The live OpenAPI document is the
field-level source of truth for the version you are running.

```bash
curl -sS http://localhost:8080/health
curl -sS http://localhost:8080/openapi.json
```

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
| `GET` | `/openapi.json` | Generated OpenAPI schema |
| `GET` | `/docs` | Swagger UI |

Use `/health` for liveness and `/ready` for readiness. During model download or
runtime preparation, a process can be healthy while `/ready` still returns
`503`.

## Inspect signals without an inference call

The classification endpoints are useful when tuning signals or diagnosing why
a decision did not match. They do not call a generation backend.

```bash
curl -sS http://localhost:8080/api/v1/classify/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a Python function that merges two sorted lists."}'
```

| Method | Path | Use |
| --- | --- | --- |
| `POST` | `/api/v1/classify/intent` | Evaluate intent/domain routing |
| `POST` | `/api/v1/classify/pii` | Detect configured PII types |
| `POST` | `/api/v1/classify/security` | Evaluate jailbreak and security classification |
| `POST` | `/api/v1/classify/fact-check` | Decide whether text needs fact checking |
| `POST` | `/api/v1/classify/user-feedback` | Classify user feedback |
| `POST` | `/api/v1/classify/combined` | Run intent, PII, and security classification |
| `POST` | `/api/v1/classify/batch` | Run a selected classifier over a batch |
| `POST` | `/api/v1/eval` | Evaluate all configured signals |
| `POST` | `/api/v1/nli` | Evaluate a premise/hypothesis pair |
| `POST` | `/api/v1/embeddings` | Generate configured text or image embeddings |
| `POST` | `/api/v1/similarity` | Compare a text pair |
| `POST` | `/api/v1/similarity/batch` | Run batch similarity matching |

Names, scores, and matched rules depend on the active recipe. Use the live
schema for each endpoint's supported input forms.

## Inspect models and metrics

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/info/models` | Loaded model inventory |
| `GET` | `/info/classifier` | Classifier configuration and status |
| `GET` | `/api/v1/embeddings/models` | Loaded embedding models |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `GET` | `/metrics/classification` | Classification counters and timing |

Secrets in classifier information are redacted unless the caller has
`secret_view`.

## Read and change router configuration

Read the current canonical document and its `ETag` before making a change:

```bash
curl -i http://localhost:8080/config/router \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}"
```

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/config/router` | Read the active canonical configuration |
| `POST` | `/config/router/validate` | Validate and normalize without writing |
| `PATCH` | `/config/router` | Merge, validate, persist, and hot-reload an update |
| `PUT` | `/config/router` | Replace, validate, persist, and hot-reload the document |
| `GET` | `/config/router/versions` | List configuration backups |
| `POST` | `/config/router/rollback` | Restore a backup |
| `GET` | `/config/hash` | Compare persisted, generated, and active hashes |

Recipe operations use the same canonical document:

| Method | Path | Use |
| --- | --- | --- |
| `GET` | `/config/router/recipes` | List default and named recipes and their entrypoints |
| `POST` | `/config/router/recipes/validate` | Validate a recipe mutation without applying it |
| `GET` | `/config/router/recipes/{name}` | Read one recipe |
| `PUT` | `/config/router/recipes/{name}` | Create or replace one recipe |
| `DELETE` | `/config/router/recipes/{name}` | Delete an unreferenced named recipe |

Recipe `PUT` and `DELETE` require `If-Match`. Config mutations validate before
writing, create a backup, and trigger reload; a successful HTTP response does
not mean upstream model backends themselves are healthy. Check `/ready` and
send a representative request after a change.

## Manage knowledge bases and stored data

Knowledge-base configuration:

| Method | Path | Use |
| --- | --- | --- |
| `GET`, `POST` | `/config/kbs` | List or create managed knowledge bases |
| `GET`, `PUT`, `DELETE` | `/config/kbs/{name}` | Read, update, or delete one knowledge base |
| `GET` | `/config/kbs/{name}/map/metadata` | Read generated map metadata |
| `GET` | `/config/kbs/{name}/map/data.ndjson` | Stream map data as NDJSON |

OpenAI-compatible storage and router memory:

| Resource | Base path | Operations |
| --- | --- | --- |
| Long-term memory | `/v1/memory` | List and delete by scope; read or delete by id |
| Vector stores | `/v1/vector_stores` | Create, list, read, update, delete, and search |
| Vector-store files | `/v1/vector_stores/{id}/files` | Attach, list, inspect, and detach files |
| Files | `/v1/files` | Upload, list, inspect, download, and delete |

These routes return `503` when their required service is unavailable. File
upload uses multipart form data; consult the live schema for limits and fields.

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
curl -sS http://localhost:8080/v1/router/outcomes \
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
### Discovery and health

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/ready` | Readiness endpoint that turns green only after startup completes |
| `GET` | `/startup-status` | Detailed router startup and model-download status |
| `GET` | `/api/v1` | API discovery and documentation |
| `GET` | `/openapi.json` | OpenAPI 3.0 specification |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### Classification and signals

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/classify/intent` | Classify user queries into routing categories |
| `POST` | `/api/v1/classify/pii` | Detect personally identifiable information in text |
| `POST` | `/api/v1/classify/security` | Detect jailbreak attempts and security threats |
| `POST` | `/api/v1/classify/fact-check` | Classify if text needs fact-checking |
| `POST` | `/api/v1/classify/user-feedback` | Classify user feedback type (satisfied, need_clarification, wrong_answer, want_different) |
| `POST` | `/api/v1/classify/combined` | Perform combined classification (intent, PII, and security) |
| `POST` | `/api/v1/classify/batch` | Batch classification with configurable task_type parameter |
| `POST` | `/api/v1/eval` | Evaluate all configured signals regardless of decision usage |
| `POST` | `/api/v1/nli` | Natural language inference classification for premise and hypothesis pairs |
| `POST` | `/api/v1/embeddings` | Generate text and image embeddings |
| `POST` | `/api/v1/similarity` | Calculate pairwise text similarity |
| `POST` | `/api/v1/similarity/batch` | Calculate batch text-similarity matches |

### Models and metrics

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/info/models` | Get information about loaded models |
| `GET` | `/info/classifier` | Get classifier information and status (secrets redacted without secret_view) |
| `GET` | `/api/v1/embeddings/models` | Get information about loaded embedding models |
| `GET` | `/v1/models` | OpenAI-compatible model listing |
| `GET` | `/metrics/classification` | Get classification metrics and statistics |
| `POST` | `/v1/router/outcomes` | Submit Router Learning outcome feedback linked to a replay record |

### Router config and recipes

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/config/router/recipes` | List the default and named routing recipes with their entrypoints |
| `POST` | `/config/router/recipes/validate` | Validate a recipe mutation without writing or reloading config |
| `GET` | `/config/router/recipes/{name}` | Read one routing recipe and its entrypoints |
| `PUT` | `/config/router/recipes/{name}` | Atomically create or replace one routing recipe; requires If-Match |
| `DELETE` | `/config/router/recipes/{name}` | Delete an unreferenced named routing recipe; requires If-Match |
| `GET` | `/config/router` | Get the current router config as JSON (secrets redacted without secret_view) |
| `POST` | `/config/router/validate` | Validate and normalize a router config without writing it |
| `PATCH` | `/config/router` | Merge a router config update (validates, backs up, writes, triggers hot-reload) |
| `PUT` | `/config/router` | Replace the router config (validates, backs up, writes, triggers hot-reload) |
| `POST` | `/config/router/rollback` | Rollback to a previous router config version |
| `GET` | `/config/router/versions` | List available router config backup versions |
| `GET` | `/config/hash` | Compare persisted source, generated runtime, and active router config hashes |

### Knowledge bases

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/config/kbs` | List configured knowledge bases |
| `POST` | `/config/kbs` | Create a managed knowledge base |
| `GET` | `/config/kbs/{name}` | Read a knowledge base |
| `GET` | `/config/kbs/{name}/map/metadata` | Read generated knowledge-base map metadata |
| `GET` | `/config/kbs/{name}/map/data.ndjson` | Stream generated knowledge-base map data as NDJSON |
| `PUT` | `/config/kbs/{name}` | Update a managed knowledge base |
| `DELETE` | `/config/kbs/{name}` | Delete a managed knowledge base |

### Memory, vector stores, and files

These require the corresponding service to be enabled; otherwise the API returns `503`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/v1/memory` | List long-term memories |
| `DELETE` | `/v1/memory` | Delete memories by scope |
| `GET` | `/v1/memory/{id}` | Read one long-term memory |
| `DELETE` | `/v1/memory/{id}` | Delete one long-term memory |
| `POST` | `/v1/vector_stores` | Create a vector store |
| `GET` | `/v1/vector_stores` | List vector stores |
| `GET` | `/v1/vector_stores/{id}` | Read a vector store |
| `POST` | `/v1/vector_stores/{id}` | Update a vector store |
| `DELETE` | `/v1/vector_stores/{id}` | Delete a vector store |
| `POST` | `/v1/vector_stores/{id}/search` | Search a vector store |
| `POST` | `/v1/vector_stores/{id}/files` | Attach a file to a vector store |
| `GET` | `/v1/vector_stores/{id}/files` | List files attached to a vector store |
| `DELETE` | `/v1/vector_stores/{id}/files/{file_id}` | Detach a file from a vector store |
| `POST` | `/v1/files` | Upload a file |
| `GET` | `/v1/files` | List uploaded files |
| `GET` | `/v1/files/{id}` | Read uploaded-file metadata |
| `DELETE` | `/v1/files/{id}` | Delete an uploaded file |
| `GET` | `/v1/files/{id}/content` | Download uploaded-file content |

File uploads accept documents (`.txt`, `.md`, `.json`, `.csv`, `.html`) for
vector-store ingestion. Upload an image (`.png`, `.jpg`, `.jpeg`, `.gif`,
`.webp`) with `purpose=vision` to use it as model input: a Response API request
can then reference it with `{"type": "input_image", "file_id": "<id>"}`, and
the router inlines the image for the selected backend.

### Other endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/v1/router_replay` | List Router Replay records |
| `GET` | `/v1/router_replay/` | List Router Replay records (trailing-slash compatibility) |
| `GET` | `/v1/router_replay/aggregate` | Aggregate Router Replay routing and cost metadata |
| `GET` | `/v1/router_replay/trajectory` | Build a Router Replay session trajectory |
| `GET` | `/v1/router_replay/{id}` | Read one Router Replay record |
| `GET` | `/api/v1/response-cache/capabilities` | Get response-cache backend capabilities |
| `GET` | `/api/v1/response-cache/health` | Check response-cache backend health |
| `GET` | `/api/v1/response-cache/stats` | Get redacted response-cache statistics |
| `GET` | `/api/v1/response-cache/audit` | Get redacted response-cache mutation audit entries |
| `POST` | `/api/v1/response-cache/test` | Validate and probe a response-cache candidate configuration |
| `POST` | `/api/v1/response-cache/invalidate` | Dry-run or invalidate a scoped response-cache partition |
| `POST` | `/api/v1/response-cache/flush` | Advance a scoped or global response-cache epoch |
| `GET` | `/api/v1/context-compression/capabilities` | Get context-compression capabilities |
| `GET` | `/api/v1/context-compression/health` | Check context-compression runtime health |
| `GET` | `/api/v1/context-compression/stats` | Get redacted context-compression statistics |
| `POST` | `/api/v1/context-compression/preview` | Preview context compression without persistence |
| `POST` | `/api/v1/context-compression/recovery/invalidate` | Invalidate a trusted context-recovery request scope |
<!-- END-GENERATED-ENDPOINT-INDEX -->
