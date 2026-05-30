# Router API, Vector Store, and Harness Ratchet

## Goal

- Converge the router service platform API layer, vector-store ingestion path, and
  agent harness toward smaller, safer, easier-to-resume modules.
- Remove avoidable memory and concurrency hazards in API request handling, file
  uploads, metadata publication, external vector-store backend responses, and
  ingestion lifecycle management.
- Keep the workstream resumable from repository state alone by recording the
  completed ratchets, validation state, and remaining local-smoke blocker.

## Scope

- `src/semantic-router/pkg/apiserver/**`
- `src/semantic-router/pkg/selection/**`
- `src/semantic-router/pkg/vectorstore/**`
- `tools/agent/scripts/**`
- `tools/agent/scripts/tests/**`
- `tools/make/rust.mk`
- `docs/agent/testing-strategy.md`
- `docs/agent/plans/**`

## Exit Criteria

- API request decoding uses bounded helpers and reports oversized payloads with
  explicit client-facing errors.
- API route registration and public documentation derive from one route catalog
  instead of hand-maintained duplicate lists.
- Vector-store file upload, metadata registry, manager, and ingestion pipeline
  avoid unnecessary full-buffer copies and do not leak mutable internal state.
- Ingestion lifecycle semantics are explicit: stopped pipelines reject new work,
  queued jobs are failed during shutdown, and stopped pipelines can be restarted
  without dead workers.
- External vector-store backend HTTP responses are bounded so unhealthy
  dependencies cannot force unbounded router memory growth.
- Default vector-store chunking cannot emit unbounded chunks when an input file
  has no paragraph or heading boundaries.
- Llama Stack backend code is split by responsibility so collection lifecycle,
  search adaptation, and HTTP transport can evolve independently.
- Knowledge-base API upserts reject invalid scores and empty label groups before
  staging managed assets or patching config files.
- Knowledge-base API helpers are split into request normalization, document
  projection, and YAML persistence seams instead of one mixed hotspot file.
- Model-info API helpers are split into HTTP response assembly, classifier
  inventory, embedding discovery, and runtime-state enrichment seams.
- API documentation helpers are split into route documentation handlers,
  OpenAPI contract types, and OpenAPI spec generation.
- API documentation tests are split by route catalog, OpenAPI contract, and
  HTTP documentation endpoint behavior.
- Memory API helpers are split into request parsing, response projection, and
  handler orchestration, with list-limit validation handled at the API boundary.
- Vector-store list pagination validates limit, order, and mutually exclusive
  cursors at the API boundary, and manager pagination honors both `after` and
  `before` without sorting under the store read lock.
- Valkey backend search result parsing is isolated from the backend lifecycle
  adapter and skips malformed documents instead of silently publishing invalid
  zero-score or zero-index hits.
- Feedback API requests and selector update paths reject ambiguous or corrupting
  model-comparison contracts before selector state is mutated.
- Agent changed-file handling and local validation commands can consume
  whitespace, comma, newline, or path-file inputs consistently.
- The changed-file validation ladder passes, and any local-smoke gap that cannot
  run in the current environment is recorded with concrete blocker evidence.

## Task List

- [x] `AVH001` Add bounded JSON request helpers and route-specific body limits.
- [x] `AVH002` Replace duplicate API route registration and documentation lists
  with one route catalog.
- [x] `AVH003` Split oversized API handler files along endpoint ownership seams
  without changing public behavior.
- [x] `AVH004` Stream multipart file uploads into `FileStore` and clean temporary
  multipart state after upload handling.
- [x] `AVH005` Make vector-store metadata publication defensive by cloning
  records at manager, registry, file-store, and pipeline boundaries.
- [x] `AVH006` Fix ingestion pipeline lifecycle so stopped pipelines reject new
  attachments, fail queued shutdown work, and restarted pipelines launch live
  workers.
- [x] `AVH007` Bound Llama Stack backend response reads and keep diagnostic
  error bodies capped.
- [x] `AVH008` Add a fixed-size fallback inside auto chunking for oversized
  paragraph blocks.
- [x] `AVH009` Split Llama Stack backend search and HTTP transport helpers out
  of the collection-lifecycle adapter.
- [x] `AVH010` Split agent changed-file and context-resolution helpers out of the
  monolithic resolution module and cover changed-file parsing.
- [x] `AVH011` Make Rust build targets remove stale native library artifacts
  before rebuilding release libraries.
- [x] `AVH012` Fix make-level `AGENT_CHANGED_FILES_PATH` handling when
  `CHANGED_FILES` is passed through as an empty string.
- [x] `AVH013` Run focused Go, race, diff, and repo-native agent gates for the
  affected surfaces.
- [x] `AVH014` Split vector-store legacy test closures and file layout so
  structure checks no longer need relaxed hotspot warnings for touched tests.
- [x] `AVH015` Split the oversized Llama Stack backend test suite into
  collection, chunk, search, request, flow, and integration contract files.
- [x] `AVH016` Reject invalid knowledge-base thresholds and empty groups at the
  API normalization boundary, and split KB document/persistence helpers out of
  the request normalization file.
- [x] `AVH017` Split model-info API helpers by classifier inventory,
  embedding discovery, and runtime-state enrichment seams.
- [x] `AVH018` Split API documentation handlers from OpenAPI contract types and
  spec generation, and split related tests by contract.
- [x] `AVH019` Split memory API request/response helpers and validate list
  limits before store calls.
- [x] `AVH020` Validate vector-store list query params and implement the
  previously declared `before` cursor in manager pagination.
- [x] `AVH021` Split Valkey backend search-result parsing and reject malformed
  search documents at the backend boundary.
- [x] `AVH022` Validate feedback API model-comparison, category-alias, and
  confidence contracts before selector updates.
- [x] `AVH023` Move common feedback validation into the selection layer so
  direct selector updates reject self-comparisons and invalid confidence before
  mutating policy state.
- [x] `AVH024` Make Hybrid selector feedback fanout preflight component
  requirements before any child selector mutates policy state.
- [x] `AVH025` Add a shared selection-context contract so selector entrypoints
  reject nil or empty request state instead of panicking.
- [x] `AVH026` Split the touched latency-aware selector hotspot into collection,
  scoring, confidence, and fallback helpers without changing selection behavior.
- [x] `AVH027` Align direct selector API validation with config validation by
  rejecting blank candidate model names and out-of-range latency percentiles.
- [x] `AVH028` Add a shared selection-result contract and make extproc model
  selection reject nil, blank, or non-candidate selector outputs before routing.
- [ ] `AVH029` Run local smoke and manual E2E profiles once Docker is available
  in the local environment.

## Current Loop

- Loop status: implementation loop active as of 2026-05-24; code-level ratchets
  passed the repo-native gates, but local smoke remains blocked by the local
  Docker daemon being unavailable.
- Completed in this loop:
  - added bounded request body parsing and consistent JSON error mapping for
    API handlers
  - moved API route registration, overview, and OpenAPI generation onto a shared
    route catalog
  - split vector-store API handler surfaces into CRUD, search, and file-attach
    modules
  - split classification and embedding handler helpers enough to keep request
    validation and response assembly testable
  - streamed uploaded files through `SaveFromReader` instead of buffering the
    whole file in API memory
  - added defensive metadata cloning across vector-store registry, manager,
    file-store, and pipeline status APIs
  - fixed the race exposed by `go test -race ./pkg/vectorstore` around pipeline
    status mutation
  - fixed ingestion pipeline restart semantics, stopped-pipeline attachment
    rejection, and queued-job shutdown cleanup
  - bounded Llama Stack backend HTTP response reads so successful payloads have
    a hard cap and error diagnostics are read as a small preview
  - capped default auto chunking for long paragraph blocks so a file without
    blank lines cannot create one oversized embedding request
  - split the Llama Stack backend into collection lifecycle, search adapter, and
    HTTP transport files so the adapter no longer grows as one hotspot
  - split agent changed-file parsing and context-resolution helpers out of the
    compatibility resolution module
  - fixed make-level `AGENT_CHANGED_FILES_PATH` report handling so empty
    `CHANGED_FILES` no longer hides path-file input and under-reports the
    working tree
  - split `FileStore`, `Manager`, and `IngestionPipeline` tests into smaller
    responsibility-scoped `Describe` blocks and moved file-store concurrency
    coverage to its own file
  - split the monolithic Llama Stack backend test file into smaller
    contract-scoped files, including a helper-backed mock end-to-end flow
  - split the oversized memory API tests into list, get/delete, validation, and
    lifecycle files while keeping shared mock state isolated
  - rejected invalid knowledge-base top-level thresholds, label thresholds, and
    empty groups before managed assets are staged or config YAML is patched
  - split knowledge-base request normalization, document projection/path
    handling, and YAML persistence helpers into separate API-layer files
  - split the model-info API from one mixed handler file into HTTP response
    assembly, classifier model inventory, embedding model discovery, and
    runtime-state enrichment files
  - split API documentation code so route overview handlers, OpenAPI contract
    structs, and OpenAPI spec generation can evolve independently while staying
    backed by the shared route catalog
  - split API documentation tests into route-catalog, OpenAPI contract, and
    endpoint contract files
  - split memory API request parsing and response projection out of the handler
    file, and rejected invalid `limit` query values before store calls
  - added vector-store list request parsing so invalid `limit`, invalid
    `order`, and ambiguous `after` plus `before` cursors fail before manager
    calls
  - implemented vector-store `before` cursor pagination and moved list sorting
    outside the manager read lock
  - split Valkey search result parsing out of the backend lifecycle file and
    skipped malformed result documents with missing or invalid score, chunk
    index, or required returned fields
  - rejected feedback API requests where `winner_model` and `loser_model` refer
    to the same model, `confidence` falls outside `0.0..1.0`, or category
    aliases disagree
  - moved common feedback validation into `pkg/selection` so Elo, RL-driven,
    GMTRouter, RouterDC, and AutoMix reject invalid feedback before mutating
    selector state
  - made Hybrid selector feedback fanout preflight shared feedback contracts and
    winner-required component requirements before updating any child selector,
    avoiding partial Elo mutations when RouterDC or AutoMix would reject the
    same event
  - added a shared selection-context validation seam and wired it through
    selector `Select` entrypoints, global selection fallback, AutoMix cascade
    initialization, and RL-driven multi-round selection
  - aligned direct selector API validation with the config validator by rejecting
    blank candidate model names before any selector fallback can return an empty
    model
  - rejected out-of-range latency-aware percentile values at the selector
    boundary instead of silently degrading them into no-stats fallback behavior
  - added a shared selection-result validation seam so global selector calls and
    extproc routing reject nil, blank, or non-candidate selector outputs before
    routing decisions are recorded
  - made extproc model selection fall back to the first valid candidate, or the
    configured default model when no valid decision model references remain
  - split `LatencyAwareSelector.Select` into smaller latency candidate
    collection, normalized scoring, and confidence helpers after the structure
    harness flagged the touched function as a new non-legacy hotspot
  - updated the testing strategy docs for `CHANGED_FILES` and
    `AGENT_CHANGED_FILES_PATH`
  - removed stale native library artifacts before release Rust rebuild targets
- Commands run:
  - `make agent-report ENV=cpu CHANGED_FILES="..."`
  - `go test ./pkg/apiserver`
  - `go test ./pkg/vectorstore`
  - `make agent-lint CHANGED_FILES="..."`
  - `go test -race ./pkg/vectorstore`
  - `go test -race ./pkg/apiserver`
  - `git diff --check`
  - `make agent-ci-gate CHANGED_FILES="..."`
  - `python -m unittest tools.agent.scripts.tests.test_changed_files_path`
  - `make agent-report ENV=cpu AGENT_CHANGED_FILES_PATH="..."`
  - `python tools/agent/scripts/structure_check.py <vectorstore test files>`
  - `python tools/agent/scripts/structure_check.py src/semantic-router/pkg/vectorstore/llama_stack_backend*_test.go`
  - `python tools/agent/scripts/structure_check.py src/semantic-router/pkg/apiserver/kb_store.go src/semantic-router/pkg/apiserver/kb_document.go src/semantic-router/pkg/apiserver/kb_persistence.go src/semantic-router/pkg/apiserver/route_taxonomy_classifiers_test.go`
  - `go test ./pkg/apiserver -run 'TestBuildModelsInfoResponse|TestNormalizeEmbeddingModelPath'`
  - `go test ./pkg/apiserver -run 'TestAPIRouteCatalog|TestOpenAPISpec|TestAPIOverview|TestSwaggerUI'`
  - `go test ./pkg/apiserver -run 'TestHandleListMemories|TestHandleDeleteMemoriesByScope|TestHandleGetMemory|TestHandleDeleteMemory'`
  - `go test ./pkg/apiserver -run 'TestParseVectorStoreListParams'`
  - `go test ./pkg/vectorstore -run 'TestVectorStore/Manager_store_listing'`
  - `go test ./pkg/vectorstore -run 'TestVectorStore/ValkeyBackend'`
  - `go test ./pkg/apiserver -run 'TestHandleFeedback|TestHandleGetRatings'`
  - `go test ./pkg/selection -run 'TestNormalizeFeedback|TestValidateFeedback|TestEloSelectorRejectsInvalidFeedback|TestRLDrivenSelectorRejectsInvalidFeedback|TestEloSelector_UpdateFeedback|TestRLDrivenSelector_UpdateFeedback'`
  - `go test ./pkg/selection -run 'TestHybridSelectorRejectsFeedbackBeforeFanoutWhenWinnerRequired|TestNormalizeFeedback|TestValidateFeedback|TestEloSelectorRejectsInvalidFeedback|TestRLDrivenSelectorRejectsInvalidFeedback'`
  - `go test ./pkg/selection -run 'TestValidateSelectionContext|TestSelectorsRejectInvalidSelectionContextWithoutPanic|TestGlobalSelectRejectsInvalidSelectionContextBeforeFallback|TestSelectorCascadeEntrypointsRejectInvalidSelectionContext'`
  - `go test ./pkg/selection -run 'TestLatencyAware|TestValidateSelectionContext|TestSelectorsRejectInvalidSelectionContextWithoutPanic'`
  - `go test ./pkg/selection -run 'TestLatencyAwareSelector|TestValidateSelectionContext|TestSelectorsRejectInvalidSelectionContextWithoutPanic|TestGlobalSelectRejectsInvalidSelectionContextBeforeFallback|TestSelectorCascadeEntrypointsRejectInvalidSelectionContext'`
  - `go test ./pkg/selection -run 'TestValidateSelectionResult|TestGlobalSelectRejectsInvalidSelectorResult|TestGlobalSelectRejectsNilSelectorResult|TestValidateSelectionContext'`
  - `go test ./pkg/extproc -run 'TestSelectModelFromCandidates'`
  - `go test ./pkg/selection ./pkg/extproc`
  - `go test ./pkg/selection`
  - `go test -race ./pkg/selection`
  - `python tools/agent/scripts/structure_check.py src/semantic-router/pkg/apiserver/route_feedback.go src/semantic-router/pkg/apiserver/route_feedback_request.go src/semantic-router/pkg/apiserver/route_feedback_test.go`
  - `python tools/agent/scripts/structure_check.py src/semantic-router/pkg/selection/feedback.go src/semantic-router/pkg/selection/feedback_test.go src/semantic-router/pkg/selection/elo.go src/semantic-router/pkg/selection/rl_driven.go src/semantic-router/pkg/selection/gmtrouter.go src/semantic-router/pkg/selection/router_dc.go src/semantic-router/pkg/selection/automix.go src/semantic-router/pkg/apiserver/route_feedback_request.go src/semantic-router/pkg/apiserver/route_feedback.go src/semantic-router/pkg/apiserver/route_feedback_test.go`
  - `docker info`
- Remaining blocker:
  - `docker info` cannot connect to the local Docker socket at
    `/Users/bitliu/.docker/run/docker.sock`, so `make agent-dev ENV=cpu`,
    `make agent-serve-local ENV=cpu`, `make agent-smoke-local`, and the manual
    `kubernetes` plus `vectorstore-registry` local E2E profiles cannot be run in
    this environment yet.

## Decision Log

- 2026-05-24: Treat request-size limits as API infrastructure rather than
  handler-specific ad hoc checks, so future JSON handlers inherit the same
  status-code and error-shape behavior.
- 2026-05-24: Keep route metadata close to registration because the API overview
  and OpenAPI document were drifting from the actual mux setup.
- 2026-05-24: Prefer defensive copies at vector-store metadata boundaries over
  relying on caller discipline; this removes data races and prevents external
  mutation of registry-owned state.
- 2026-05-24: Keep stopped ingestion pipelines closed to new work and fail
  queued shutdown work. Restart is supported, but accepting or retaining
  attachments without live workers creates permanent `in_progress` states.
- 2026-05-24: Bound external Llama Stack response reads in the backend adapter;
  router memory should be controlled at dependency boundaries, not only at
  inbound API handlers.
- 2026-05-24: Preserve auto chunking's paragraph-first semantics, but use the
  static chunker as a safety fallback for oversized blocks before embedding.
- 2026-05-24: Split Llama Stack backend responsibilities by API seam; transport
  limits belong with the HTTP helper, while search request/response adaptation
  belongs with the search method.
- 2026-05-24: Keep `agent_resolution.py` as a compatibility re-export layer
  while moving new harness logic into narrower modules.
- 2026-05-24: Treat an empty `CHANGED_FILES` argument as absent when
  `AGENT_CHANGED_FILES_PATH` is provided, because make targets pass empty
  variables explicitly and the path-file source must still drive validation
  scope.
- 2026-05-24: Keep high-churn vector-store tests organized by behavioral
  contract instead of one large package-level closure; the structure harness
  should report real regressions, not warn every time these tests are touched.
- 2026-05-24: Treat backend adapter tests as contract documentation. A single
  monolithic test file hides which behavior belongs to collection lifecycle,
  chunk mutation, search adaptation, HTTP request behavior, or integration
  coverage.
- 2026-05-24: Validate knowledge-base API payload scores and group membership at
  the request boundary. Relying on later config parsing is insufficient because
  unreferenced KBs intentionally skip manifest-backed validation during startup.
- 2026-05-24: Keep knowledge-base API helpers separated by seam: request
  normalization owns client payload rules, document helpers own API projection
  and managed paths, and persistence helpers own YAML AST patching plus runtime
  sync.
- 2026-05-24: Keep model-info API handlers as transport orchestration only;
  classifier inventory, embedding FFI discovery, and startup-state enrichment
  change on different clocks and should not share one growing handler file.
- 2026-05-24: Keep API documentation endpoints separate from OpenAPI contract
  construction. Both should derive from `apiRoutes()`, but handler code should
  not own schema-generation internals.
- 2026-05-24: Treat memory list `limit` as an API contract, not only a backend
  default. Invalid limits fail before store calls; oversized values are capped
  consistently at the documented maximum.
- 2026-05-24: Treat vector-store list cursors as a real API contract. `before`
  was already declared in `ListStoresParams`, so manager pagination now honors
  it, while ambiguous external `after` plus `before` requests fail at the
  handler boundary.
- 2026-05-24: Treat Valkey search result parsing as a dependency-boundary
  contract. A malformed Valkey document should be skipped, not coerced into a
  zero-score or zero-index search hit that downstream ranking could mistake for
  valid data.
- 2026-05-24: Treat feedback API input as selector state mutation, not generic
  telemetry. Self-comparisons, conflicting category aliases, and out-of-range
  confidence values must fail before Elo, RL-driven, or GMTRouter selectors see
  the event.
- 2026-05-24: Keep feedback validation in the selection layer as the source of
  truth. HTTP handlers may map errors to API codes, but direct selector callers
  must get the same self-comparison and confidence guards before policy state is
  updated.
- 2026-05-24: Treat Hybrid selector feedback fanout as a single policy-state
  mutation boundary. Component-specific requirements must be checked before any
  child selector updates so one accepted Elo update cannot be followed by a
  RouterDC or AutoMix rejection for the same event.
- 2026-05-24: Treat `SelectionContext` as a public selector API contract rather
  than an internal convention. Selector entrypoints should reject nil context or
  empty candidates with explicit errors before fallback or algorithm-specific
  logic can panic.
- 2026-05-24: Keep direct selector API validation at parity with config
  validation for routing-policy inputs. Blank model refs and invalid
  latency-aware percentiles should fail explicitly rather than selecting an empty
  model or falling back for the wrong reason.
- 2026-05-24: Treat selector outputs as public API contracts too. A selector may
  fail, but it must not route to nil, blank, or non-candidate results; extproc
  should recover to a valid candidate or configured default rather than panic or
  record an impossible routing decision.
- 2026-05-24: When a touched non-legacy selector trips the structure harness,
  split the helper boundary immediately instead of widening the exception set.
  Latency-aware collection, scoring, and confidence now evolve independently.

## Follow-up Debt / ADR Links

- [TD006 Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots](../tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md)
- [TD031 Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [TD034 Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract](../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- [TD037 Dev Integration Environment Ownership and Shared-Suite Topology Still Diverge Across CLI, Kind, and CI](../tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md)
