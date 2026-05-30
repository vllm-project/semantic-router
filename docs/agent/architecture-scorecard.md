# Architecture Scorecard

This document is the canonical architecture scorecard for whole-repository
quality review and staged improvement. It turns broad architecture audit
findings into a repeatable 100-point ratchet that future PRs can update.

## Purpose

- keep architecture scores evidence-backed instead of chat-only
- make each score update traceable to source, tests, docs, or harness output
- coordinate staged PRs toward a 100-point target without a broad rewrite
- link score movement to the existing technical debt and execution-plan system

## Scoring Model

Each dimension starts at 100. Findings deduct from the score by severity:

- open blocker: -20
- major: -10
- medium: -5
- minor: -2

Positive score movement is allowed only when the improving PR records a
validation command that passed. Score updates must include:

- the changed subsystem or contract
- the evidence that changed
- the validation command and result
- any debt entry closed, narrowed, or left open

## Current Baseline

Current score date: 2026-05-25.

| Dimension | Score | Main deductions |
| --- | ---: | --- |
| System completeness | 93 | Broad product surface exists, K8s reconcile shares config-family validation with file config, the DSL tree-authoring round-trip contract is explicit, the Python CLI now has a shared Docker/K8s deployment target family and TD004 is closed as stale, Response API storage defaults to Redis with restart-recovery E2E coverage, local CLI/runtime wiring now covers vector-store metadata Postgres defaults, router replay runtime backend selection now recognizes configured durable store blocks when `store_backend` is omitted, RL-driven user/session learning state now survives restart when `storage_path` is configured, decision-scoped RLDriven and GMTRouter persistence/config knobs now reach the runtime selector factory, Dashboard auth now has a server-issued session cookie, logout endpoint, cookie-backed reload path, bounded backend token intake, explicit cookie expiry, and persisted session revocation alongside the existing token response, Helm can now mount dashboard-local auth/session/workflow SQLite state on a PVC while blocking unsupported multi-replica dashboard settings, Helm now rejects multi-replica router renders that would run local learning selector state for Elo, RL-driven, or GMTRouter without an explicit opt-out, and Helm now rejects invalid HPA replica bounds before producing Kubernetes resources; production-readiness still varies across runtime, dashboard, operator, and stateful workflows. |
| Maintainability, iteration, extensibility | 98 | The K8s/file config validator dispatch is shared, TD036 is resolved by a narrowed contract, DSL compiler payload lowering and selected decompile formatting now use named DSL value seams, Dashboard tools config and OpenClaw listener discovery go through local contract adapters, Dashboard security-policy apply and validation logic now live in focused sibling helpers instead of growing the handler file, Dashboard authenticated route registration now lives in a sibling route registry plus testable route manifest instead of the setup/auth shell, Dashboard overview row shaping plus Playground conversation, queue persistence, and auth-session refresh have tested sibling support modules, Dashboard auth public handlers, admin user handlers, admin metadata handlers, session-user helpers, session storage, role/permission storage, and audit storage now live in focused sibling modules instead of two broad auth hotspot files, Dashboard lint now distinguishes support modules from Fast Refresh component boundaries, Operator canonical config construction now separates base builder, backend discovery, and CRD spec-family translation, CLI storage default injection separates semantic-cache Milvus from vector metadata Postgres, CLI runtime support now splits command help, runtime paths, KB bootstrap state, and runtime config mutation into focused support modules with KB tests separated from the broad runtime-support suite, legacy classifier mapping bootstrap, legacy unified-classifier label loading, classifier model-discovery scanning/selection, model path validation, discovery-info reporting, unified auto-initialization, built-in category/jailbreak/PII classifier behavior, native category/jailbreak/PII init and inference adapters, vLLM jailbreak request/client and parser ownership, contrastive preference embedding preload and scoring ownership, preference external/contrastive dispatch ownership, complexity candidate preload/query/scoring ownership, request-time jailbreak/PII/embedding/preference signal evaluators, generic keyword/domain/fact-check/feedback/reask/context/complexity/modality evaluator ownership, signal group application/resolution/trace/output-policy ownership, projection ordering/input/output/trace ownership, signal result/usage/authz/decision bridge helpers, embedding backend initialization, embedding candidate-preload fanout, embedding scoring/prototype maintenance, embedding text/multimodal request-path ownership, shared prototype-bank construction/clustering/scoring ownership, keyword rule preprocessing, keyword dispatch cache ownership, keyword regex/fuzzy matching, MCP client/protocol handling, MCP category mapping bootstrap, MCP entropy/metrics runtime ownership, hallucination detector basic/NLI runtime ownership, classifier option construction/backend selection/rule option building, KB manifest/classify orchestration, KB embedding preload/prototype rebuild, KB label/group/metric scoring, modality detection, category entropy reasoning/metrics, family-owned fact-check/hallucination/feedback/preference/language lifecycle files, services-side PII, security, recommendation, unified-batch, signal DTO, signal response, and matched-signal helpers, plus unified-classifier public types/statistics, LoRA lifecycle guards, and native result decoding now live in focused seams, Fleet Sim threshold Pareto logic has a sibling module, native backend capabilities have a typed router-side seam, MLP FFI device selection now has one explicit helper, GMTRouter local persistence now has a sibling storage helper, decision-scoped learning-selector config mapping is explicit and tested, RL-driven looper selection now uses the runtime selector registry through a focused factory seam, API-side selection state now prefers the runtime registry over a process-wide global, classification runtime refresh now separates service-local config refresh from legacy global config publication, ExtProc request-time plugin config precedence now has a small router-owned helper instead of duplicated `config.Get()` reads, ExtProc server construction now has a runtime-registry-first config resolver instead of always entering the legacy `NewOpenAIRouter` path, and ExtProc config watch/reload ownership now lives in a focused watcher module instead of the core server file; multiple legacy hotspots still collapse orchestration, schema, validation, and UI responsibilities. |
| API stability, clarity, consistency | 94 | Canonical v0.3 config and OpenAPI surfaces exist, TD036/TD038 are resolved-contract guards, K8s reconcile validates through a shared config-contract seam, two TD015 DSL weak-typing seams are narrowed, two TD039 Dashboard slices consume contract adapters, Dashboard security-policy apply now validates merged canonical configs against endpoint contract rules and the router parser before writing, the Operator controller translation path now has narrower config-family seams, vector-store `metadata_store: postgres` now has matching CLI/docs behavior, RLDriven and GMTRouter documented algorithm fields are represented across the router config schema, CLI schema, Dashboard DSL field schema, and selector runtime mapping, Helm values now have a narrow schema for public router/dashboard deployment controls and safety-guard types, Dashboard auth now applies one backend token-normalization contract across bearer, cookie, and query transports plus a persisted session-id revocation seam for newly issued tokens, API startup plus feedback/ratings/RL-state plus vector/file API helpers resolve state from the active runtime registry without stale global fallback once a registry is present, ExtProc initial router construction, gRPC startup, config-source watcher selection, and vectorstore RAG manager/embedder lookup resolve from active router/runtime config before compatibility paths, request-time modality and system-prompt plugin config now resolve from the router instance or runtime registry before the legacy global fallback, and registry-backed config refresh no longer rewrites process-wide config through the classification service; remaining control planes still depend on router internals. |
| Performance, availability, stability | 96 | Observability and E2E coverage exist, CRD-loaded configs now fail fast through the same family validators as file configs, Response API Redis persistence has current restart-recovery E2E coverage and no longer leaks Redis integration tests into default package tests, ML pipeline restart recovery closes pending/running jobs with durable progress evidence, local vector-store metadata Postgres config now starts with usable connection defaults, router replay no longer silently falls back to memory when a durable backend block is present but `store_backend` is omitted, RL-driven user/session learning state reloads from configured storage, decision-scoped RLDriven/GMTRouter storage paths are no longer silently ignored by selector construction, GMTRouter local state uses atomic save/load helpers with corruption backup, `FileEloStorage.Close` no longer blocks when auto-save was not started, reloaded router selector state is published into the runtime registry, RL-driven looper execution now uses the router's runtime selector registry instead of a process-wide selector, ExtProc server construction no longer reparses or republishes legacy process-wide config when a runtime registry already has current config, ExtProc server startup no longer reads gRPC message limits or config-source watcher mode from stale process-wide config when router/runtime config is present, ExtProc vectorstore RAG now has a regression guard that manager/embedder lookup is runtime-registry-only, ExtProc modality and system-prompt request filters no longer adopt another runtime's process-wide config while handling a router-owned request or a pre-publication runtime registry, registry-backed config refresh no longer emits process-wide config notifications through the classification service, `/ready` now uses the same startup-state resolver as `/startup-status`, Dashboard security-policy apply rejects invalid merged router configs before write/runtime apply, Dashboard auth login/logout, cookie-backed reload, bounded token intake, explicit cookie expiry, logout-time session revocation, and inactive-session pruning have server-session coverage, Helm now blocks multi-replica router renders when active learning selectors would keep local replica-divergent state and rejects invalid HPA min/max replica bounds, the Dashboard route manifest now has a frontend unit-test gate inside `make dashboard-check`, and Dashboard frontend lint is clean; broader state recovery and classification seams still carry reliability risk. |
| Memory and compute management | 95 | Response API history is durable by default through Redis and its memory backend is documented as local-dev only; Dashboard ML pipeline workflow state has a durable recovery event path, vector-store metadata has a restart-safe local Postgres configuration path, router replay backend resolution now prefers configured durable stores over implicit memory fallback, API-side model-selection access is no longer hard-wired to the process-wide registry, `InitWithRuntime` config and memory API startup now use only the shared runtime registry instead of adopting stale process-wide config or memory state, runtime-registry-backed vector/file/selector helpers no longer adopt process-wide globals before publication, ExtProc server construction now uses the runtime registry config directly instead of re-entering the legacy global config publication path, runtime-registry-backed config refresh no longer publishes refreshed service config into `config.Get()`, RL-driven looper construction on ExtProc uses the runtime selector registry instead of adopting the process-wide selector, and ExtProc startup sizes gRPC buffers from active router/runtime config instead of the process-wide config cache. RL-driven selector user preferences plus session context now persist through the existing `storage_path` seam, decision-scoped RLDriven and GMTRouter storage paths now wire into runtime selector construction, GMTRouter personalization local state now creates its storage directory, writes atomically, backs up corrupted state, and normalizes loaded state, Helm now rejects unsafe multi-replica use of replica-local Elo/RL-driven/GMTRouter learning state by default, native unified/LoRA result decoding no longer carries the old fixed 1000-result unsafe slice cap, router readiness now honors the shared Redis/file startup status contract instead of forcing a file-only read path, Playground conversation plus queued-task browser-local persistence is bounded and sanitized before hydration, browser auth-token storage now rejects malformed or oversized local values before protected transports, backend auth-token extraction now rejects malformed or oversized token material before JWT parsing, Dashboard auth can now recover the current user from the server-owned HttpOnly session cookie without a readable local token, revoke newly issued session ids from the SQLite-backed auth store, and prune old inactive session records on auth-store open, and CLI runtime KB bootstrap state now ignores corrupted YAML plus writes atomically from a dedicated state seam instead of the broad runtime-support orchestrator. Several product-facing state or learning surfaces still default to memory, local files, or compatibility globals when durable shared storage is not configured. |
| Release and upgrade completeness | 94 | Release workflows exist for Python, Rust, Docker, Helm, and docs, native backend feature support is now exposed through a typed runtime seam instead of being inferred only from build tags and backend stubs and is documented in the installation guide, the minimal native binding build no longer emits the repeated stale MLP Metal cfg warnings or generative unused-import/rand-deprecation warnings, generative raw-pointer FFI exports now carry an explicit unsafe Rust contract without changing the C ABI and the touched Guard/Qwen3Guard native files no longer breach structure-rule errors, `make release` now updates the same Python and Candle crate versions that the tag workflow validates, and `make release-check` shares one checker with local release prep plus tag validation for Python/Candle version alignment, Helm release packaging semantics, Docker plus Operator image coverage in unified release notes, upgrade/rollback runbook coverage for full tagged release image refs plus pinned Helm/Docker/Python runbook fixture commands, Candle crate publish/test/artifact attachment ownership, `vllm-sr-sim` independent PyPI release workflow/runbook ownership, Dashboard Helm state now renders with a dedicated PVC plus a template-time guard against unsupported dashboard multi-replica auth/session deployments, Helm release rendering now rejects unsafe multi-replica router configs with local learning selector state unless explicitly overridden, Helm values schema validation now catches invalid public deployment-control types before rendering, and `make helm-safety-validate` now locks those schema and local-state guard regressions in a repo-native target; live upgrade/rollback execution, shared dashboard session-store HA, and native lifecycle parity are still incomplete. |
| Potential bottlenecks and bugs | 99 | TD040's false-Ready K8s validation path is closed, TD036 has an explicit regression test, DSL compiler field-bag conversion is centralized, DSL decompile formatting for structure/conversation/tools dynamic retrieval now avoids raw maps on typed paths, Dashboard tools DB path and OpenClaw model-gateway listener discovery have direct adapter coverage, Dashboard security-policy apply no longer writes an invalid merged router config before validation, Dashboard authenticated route registration no longer duplicates route-shell layout wrappers inline and its manifest invariants are unit-tested, Dashboard overview signal and decision row calculations are unit-tested outside the page render tree, Dashboard projection filters no longer depend on recreated fallback arrays, Playground conversation and queued-task persistence now drops malformed restored entries and enforces localStorage bounds, Dashboard auth-token persistence now sanitizes local browser state before reusing it in protected transports, Dashboard backend token extraction now rejects malformed or oversized header/cookie/query token material before JWT parsing, Dashboard auth reload no longer requires localStorage token material when the server session cookie is valid, Dashboard logout now revokes newly issued persisted session ids instead of only clearing the browser cookie, ColorBends no longer risks WebGL renderer churn from prop-driven init effects, Operator CRD-to-canonical config families have focused conversion coverage, ML pipeline restart recovery no longer leaves pending jobs stuck, CLI local runtime no longer provisions vector metadata Postgres without injecting its config, Response API Redis integration tests no longer run in plain package tests without Redis, router readiness no longer false-negatives when startup status is configured for Redis/shared status, API config and memory-store resolution in `InitWithRuntime` cannot bind to stale global state before runtime publication, runtime-registry-backed vector/file/selector API helpers cannot bind to stale process-wide globals before runtime publication, registry-backed classification refresh no longer rewrites the process-wide config snapshot, RL-driven looper construction is tested against runtime-registry/global selector bleed, ExtProc gRPC max-message-size resolution is tested against router/runtime/global config precedence, ExtProc modality/system-prompt request filters are tested against router/global config bleed, system-prompt injection no longer panics when the matched decision is absent, service-level classification refresh no longer owns mapping-file bootstrap, request-time jailbreak and PII signals now own their cached model-inference fanout outside the central signal orchestrator, public complexity image classification no longer depends on a non-nil request image cache, native unified/LoRA classifier calls fail early when the selected backend does not advertise support, unified classifier initialization now rejects empty label sets before unsafe native pointer setup, LoRA result decoding no longer panics at batches above the old 1000-result cap, native MLP/generative binding warnings are removed without adding broken feature flags, generative raw-pointer FFI exports are explicitly unsafe and ownership-documented, Fleet Sim threshold Pareto exports are covered through the package seam, selector feedback/ratings/RL-state handlers are tested against runtime-registry state, RL-driven user/session learning state now has restart coverage, decision-scoped RLDriven/GMTRouter persistence config has runtime mapping coverage, GMTRouter state files are atomically replaced and corrupted files are backed up instead of silently discarded, and file-backed selector storage close is non-blocking without auto-save; dashboard, operator, and broader vector/state surfaces still retain hotspot and restart-risk debt. |
| System debt governance | 98 | Debt is well recorded, TD004/TD036/TD040 have moved from open to resolved, TD023/TD029 were reclassified as historical rather than active ratchet targets, TD015/TD020/TD027/TD028/TD030/TD031/TD033/TD039 have recorded narrowed slices, TD039 now includes Dashboard tools, OpenClaw listener, and security-policy apply-validation evidence, TD030 now has route-shell code, route and overview frontend unit-test evidence, and a clean frontend lint gate instead of only prose, TD033 now records native capability/build-hygiene, generative FFI safety/structure hygiene, release-version, shared release-contract checker, simulator release-workflow, native backend docs, upgrade-runbook fixture checks, and Candle crate workflow ownership slices, and TD034 records verified state-contract slices including Response API Redis durability, RL-driven learning-state persistence, decision-scoped RLDriven/GMTRouter config-to-runtime wiring, runtime-registry-only config and memory-store resolution for `InitWithRuntime`, runtime-registry-only vector/file/selector API helper behavior, registry-only classification refresh, RL-driven looper runtime selector ownership, ExtProc initial router config precedence, ExtProc gRPC startup config precedence, ExtProc request-time plugin config precedence including pre-publication runtime registry behavior, ExtProc vectorstore RAG runtime dependency ownership, GMTRouter local-state hardening, shared startup-status readiness, bounded browser-local playground state, browser auth-token storage hardening, server-issued auth session cookies, backend auth-token intake normalization, persisted auth session revocation, cookie-backed reload semantics, CLI KB bootstrap-state recovery, and CLI runtime state/mutation seam extraction; open debt and plan volume still mean governance is ahead of implementation. |
| Repo harness and agent friendliness | 96 | Harness validates and routes well, Helm chart and values-schema changes now route through a non-doc `helm-chart` rule with `make helm-lint` plus local reproducible `helm-ci-validate HELM_REPO_UPDATE=false` and `helm-safety-validate HELM_REPO_UPDATE=false` feature gates, Response API Redis integration coverage is now build-tag isolated from default package tests, `make dashboard-check` now runs frontend Vitest unit tests in addition to lint/type/backend checks, Dashboard frontend lint is warning-free, the CLI runtime-support changed set no longer trips structure warnings after ownership and test extraction, the Dashboard auth changed set no longer trips store or handler structure warnings after ownership extraction, and shellcheck now excludes local `.venv-*` environments so agent-created Python virtualenvs cannot poison changed-set gates; the main gap is execution-plan volume, remaining broader hotspot inventory, and scorecard-driven prioritization. |
| Documentation clarity and freshness | 95 | DSL tree-authoring docs now match the implemented flat-route contract, vector-store docs now use the current `backend_type` / `metadata_store` contract, current troubleshooting no longer claims Redis Response API storage is unimplemented, startup-status guidance and runtime warnings now use `startup_status.store_backend`, stale active model-research campaign references were removed from TD034 state scope, RLDriven/GMTRouter docs, reference config, CLI schema, Dashboard DSL schema, and runtime selector mapping now agree on persistence and tuning fields, GMTRouter local-state hardening is now reflected in the state taxonomy, browser auth-token storage, backend auth-token normalization, server-issued session cookies, persisted auth session revocation, cookie-backed session reload, Dashboard Helm local-state persistence, Helm local learning selector multi-replica guards, CLI KB bootstrap-state recovery, and runtime-registry-only config/vector/file/selector helper behavior are now reflected in the state taxonomy or user-facing docs, TD030 now reflects the current `AppRouter`/route-manifest state instead of stale `App.tsx` wording, TD033 records the current native capability seam and user-facing native backend capability guide, and upgrade/rollback docs now have mechanically checked full release image refs plus pinned upgrade and rollback fixture markers; state/API guidance still needs tighter score-linked freshness checks. |

## Score Movement Log

2026-05-24 ASR005 / TD040:

- Current-source recheck found TD038 already resolved, so it should no longer be treated as an open Contract/API deduction.
- `src/semantic-router/pkg/config/validator.go` now owns one `sharedConfigContractValidators` dispatch table for file and K8s post-conversion validation.
- `src/semantic-router/pkg/k8s/reconciler.go` now calls `config.ValidateKubernetesConfigContracts` after CRD conversion instead of mirroring individual family validators.
- `src/semantic-router/pkg/config/validator_test.go` covers shared dispatch wiring and K8s post-conversion rejection.
- `src/semantic-router/pkg/k8s/reconciler_embedding_modality_test.go` covers the original embedding-modality reconcile bug plus a non-embedding dispatch canary.
- Validation passed: `go test ./pkg/config ./pkg/k8s`; `make test-semantic-router`.
- Local smoke was not run because Docker daemon was unavailable in the desktop environment: `docker info` failed to connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-24 ASR005 / TD036:

- Current-source recheck found the prior TD036 scope partially stale: `spec/dsl.md` no longer exists, and the active DSL workstream already decided that `DECISION_TREE` is authoring sugar rather than a lossless runtime-config shape.
- `src/semantic-router/pkg/dsl/dsl_test.go` now locks the supported contract: a `DECISION_TREE` source compiles, `DecompileRouting()` emits flat `ROUTE` blocks, and that output recompiles without changing the decision set.
- `deploy/recipes/README.md` now explains paired `.dsl` / `.yaml` recipes and the flat runtime representation.
- `website/docs/tutorials/decision/retention.md` now states that retention fields round-trip through DSL/config while the `DECISION_TREE` shape does not.
- TD036 is resolved by the narrowed contract instead of adding tree metadata to canonical config.
- Validation passed: `go test ./pkg/dsl`; `make test-semantic-router`.
- Local smoke was still blocked by the unavailable Docker daemon.

2026-05-24 ASR005 / TD015 partial:

- Current-source recheck confirmed TD015 remains open, but the Go compiler/validator part was still carrying a concrete weak-typing seam through `fieldsToMap` / `valueToInterface`.
- `src/semantic-router/pkg/dsl/field_payload.go` now owns a named `dslFieldObject` / `dslFieldValue` conversion layer for DSL AST fields crossing into YAML and `StructuredPayload`.
- `src/semantic-router/pkg/dsl/compiler.go` and `src/semantic-router/pkg/dsl/validator.go` now use that helper for structure signals, conversation signals, and RAG `backend_config` payloads instead of local raw field-bag conversion.
- TD015 remains open for DSL decompile/export raw maps and Dashboard config/setup adapter record bags.
- Validation passed: `go test ./pkg/dsl`; `make test-semantic-router`.
- Local smoke remains blocked by the unavailable Docker daemon.

2026-05-24 ASR005 / TD015 decompiler typed-value slice:

- Current-source recheck confirmed the compiler/validator portion had already moved to named `dslFieldObject` helpers, but `src/semantic-router/pkg/dsl/decompiler.go` still converted structure/conversation feature objects and tools `dynamic_retrieval` back through raw maps for typed decompile paths.
- `decompiler.go` now formats these typed decompile paths from AST `Value` / `ObjectValue` helpers through `formatDSLValue`, leaving raw map normalization only for compatibility-preserving unknown plugin payload fields.
- Removed the now-unused `dynamicRetrievalConfigMap`, `dynamicRetrievalWeightsMap`, `structureFeatureToMap`, `structureSourceToMap`, `conversationFeatureToMap`, `conversationSourceToMap`, and `structurePredicateToMap` helpers.
- TD015 remains open for `emitter_yaml.go` raw infra/CRD maps and Dashboard config/setup/canonicalization record-bag adapters.
- Validation passed: `go test ./pkg/dsl -run 'TestDecompileRoutingPreservesStructureSequenceLists|TestDecompileRoutingOmitsRemovedStructureNormalizer|TestDecompileRoutingRoundTripsToolsDynamicRetrievalPluginConfig|TestDecompileRoutingPreservesRawPluginConfigMaps'` and `go test ./pkg/dsl`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/dsl/decompiler.go"`, including `make test-semantic-router`.
- Local smoke remains blocked by environment because `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-24 ASR005 / TD039 partial:

- Current-source recheck confirmed TD039 remains open, but `dashboard/backend/router/core_routes.go` had a narrow, current cross-boundary import used only to parse `tools_db_path`.
- `dashboard/backend/routercontract/tools.go` now owns the Dashboard backend contract adapter for reading tool-selection config from the canonical router config.
- `dashboard/backend/router/core_routes.go` now consumes `routercontract.ReadToolSelection` instead of importing `src/semantic-router/pkg/config` directly.
- `dashboard/backend/router/core_routes_test.go` and `dashboard/backend/routercontract/tools_test.go` cover custom tools DB path resolution and parse-error fallback.
- TD039 remains open for Dashboard config/deploy/setup handlers, Operator canonical translation, CLI contract/runtime ownership, and runtime global publication.
- Validation passed: `go test ./router ./routercontract` from `dashboard/backend`.
- Validation passed: `make dashboard-check`.
- Local smoke remains blocked by the unavailable Docker daemon.

2026-05-24 ASR006 / TD034 partial:

- Current-source recheck found TD034 partially stale for ML pipeline jobs: they are already server-owned in `dashboard/backend/workflowstore`, but restart recovery only failed `running` jobs.
- `dashboard/backend/workflowstore/mlpipeline.go` now treats `pending` and `running` ML jobs as interrupted on dashboard restart, marks them failed, and appends a durable progress event with the recovery reason.
- `dashboard/backend/workflowstore/store_test.go` now covers pending/running recovery, completed-job preservation, and the persisted recovery event.
- TD034 remains open for router-side response/replay/vector metadata durability, dashboard/browser-local state, model-selection learning state, and a shared cross-stack state taxonomy beyond this workflow slice.
- Validation passed: `go test ./workflowstore` from `dashboard/backend`.
- Validation passed: `make dashboard-check`.

2026-05-24 ASR006 / TD034 local vector metadata slice:

- Current-source recheck found the vector-store portion narrower than the original TD034 text: router runtime already has a Postgres metadata registry and startup reload path, but the Python CLI only provisioned Postgres when `vector_store.metadata_store: postgres` was set and did not inject the required `metadata_postgres` connection block.
- `src/vllm-sr/cli/service_defaults.py` now injects local Postgres connection defaults for vector-store metadata separately from semantic-cache Milvus defaults, and it skips the vector metadata backend when `vector_store.enabled: false`.
- `src/vllm-sr/tests/test_storage_backends.py` covers backend detection, disabled-store skip behavior, new metadata Postgres injection, and blank-field backfill.
- `website/docs/tutorials/global/stores-and-tools.md` and the zh-Hans translation now use `backend_type` / `metadata_store` instead of the stale `provider` field and document restart-safe metadata behavior.
- TD034 remains open for router-side state taxonomy completion, model-selection learning state, dashboard/browser-local state, and shared durable projections.
- Validation passed: `python3 -m pytest src/vllm-sr/tests/test_storage_backends.py -q`.
- Validation passed: `make vllm-sr-test` with integration tests disabled by default; Docker daemon remained unavailable for integration/local smoke.

2026-05-24 ASR006 / TD034 router replay backend-selection slice:

- Current-source recheck found another narrower replay durability gap: canonical defaults and docs now prefer Postgres for router replay, but the extproc runtime still treated an omitted `store_backend` as memory even when the config already included a durable backend block.
- `src/semantic-router/pkg/extproc/router_replay_setup.go` now resolves replay storage from the full `RouterReplayConfig`: an explicit `store_backend` wins; otherwise Postgres, Redis, Milvus, or Qdrant config blocks imply their durable backend; only an otherwise empty config keeps the compatibility memory fallback.
- `src/semantic-router/pkg/extproc/router_replay_enablement_test.go` covers explicit backend precedence, durable-backend inference, and the empty-config memory fallback.
- TD034 remained open at this point for Response API durability verification, replay restart E2E with real services, model-selection learning state, dashboard/browser-local state, and shared durable projections.
- Validation passed: `go test ./pkg/extproc -run 'TestResolveReplayStoreBackend|TestInitializeReplayRecordersUsesGlobalReplayDefault|TestApplyDecisionResultToContextUsesEffectiveRouterReplayConfig|TestBuildReplayPostgresConfigUsesUnifiedTableName' -v`.

2026-05-24 governance stale-debt recheck:

- Current-source recheck found TD023 and TD029 are already `Status: Closed` with resolution evidence and validation commands.
- The scorecard PR chain, execution plan, and debt coverage map now keep TD023/TD029 as historical guards instead of active Router Pipeline ratchet targets.

2026-05-24 ASR007 / TD020 partial:

- Current-source recheck confirmed TD020 remains open, but `src/semantic-router/pkg/services/classification.go` still owned legacy mapping bootstrap for category, PII, and jailbreak mappings.
- `src/semantic-router/pkg/classification/legacy_factory.go` now owns legacy classifier mapping bootstrap and construction through `classification.NewLegacyClassifierFromConfig`.
- `src/semantic-router/pkg/services/classification.go` and `src/semantic-router/pkg/services/classification_runtime_config.go` now assemble or refresh the legacy classifier through the classification-owned seam instead of duplicating mapping ownership.
- `src/semantic-router/pkg/classification/legacy_factory_test.go` carries the mapping gate tests that previously lived under `pkg/services`, plus nil-config coverage.
- TD020 remains open for `classifier.go`, `unified_classifier.go`, model discovery, family adapters, and remaining service-level assembly boundaries.
- Validation passed: `go test ./pkg/classification ./pkg/services`.

2026-05-25 ASR007 / TD020 legacy unified-label loading slice:

- Current-source recheck found the older TD020 scope still active in `model_discovery.go`, but narrower than a broad discovery rewrite: the file still owned directory normalization, model scanning, LoRA/legacy classification, preferred architecture selection, plus legacy unified-classifier category, PII, and security mapping reads after discovery/validation had already completed.
- `src/semantic-router/pkg/classification/model_discovery_scan.go` now owns model directory normalization, scanning, LoRA registry/name classification, legacy fallback collection, and preferred architecture selection.
- `src/semantic-router/pkg/classification/unified_legacy_labels.go` now owns legacy unified label loading, ordered mapping validation, and support for both jailbreak mapping field conventions.
- `src/semantic-router/pkg/classification/model_discovery.go` now delegates discovery scanning and legacy label loading through focused helpers, keeping the public auto-discovery API unchanged while dropping the file from 500 to 316 lines.
- `src/semantic-router/pkg/classification/unified_legacy_labels_test.go` covers successful ordered label loading, alternate jailbreak `label_to_id` / `id_to_label` mappings, and sparse mapping rejection.
- TD020 remains open for broader `classifier.go`, `unified_classifier.go`, remaining discovery/bootstrap contracts, family adapter, and request-time orchestration splits.
- Validation passed: `go test ./pkg/classification -run 'TestLoadLegacyUnifiedLabels|TestAutoDiscoverModels|TestValidateModelPaths|TestModelPathsIsComplete'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/model_discovery.go,src/semantic-router/pkg/classification/model_discovery_scan.go,src/semantic-router/pkg/classification/unified_legacy_labels.go,src/semantic-router/pkg/classification/unified_legacy_labels_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 services-side classification API slices:

- Current-source recheck found `src/semantic-router/pkg/services/classification.go` still mixed core service lifecycle, PII and security request orchestration, PII response shaping, model recommendation, routing-decision helpers, and unified batch classification wrappers.
- `src/semantic-router/pkg/services/classification_pii.go`, `classification_pii_response.go`, `classification_security.go`, `classification_recommendation.go`, and `classification_unified_batch.go` now own those service API slices. `classification.go` keeps service construction, auto-discovery assembly, intent classification, and config accessors, dropping from 627 to 219 lines and clearing the changed-file structure warning for that file.
- TD020 remains open for broader `classifier.go`, `unified_classifier.go`, remaining discovery/bootstrap contracts, family adapter, native batch-adapter internals, and request-time orchestration splits.
- Validation passed: `go test ./pkg/services -run 'TestDetectPII|TestBuildPIIResponse|TestClassificationService_ClassifyBatchUnified|TestGetRecommendedModel'`.

2026-05-25 ASR007 / TD020 unified-classifier native adapter slice:

- Current-source recheck found `src/semantic-router/pkg/classification/unified_classifier.go` still mixed public result/data types, global singleton access, stats, convenience wrappers, native batch invocation, C result decoding, initialization, mode selection, and LoRA lazy-init in one cgo file while `unified_classifier_stub.go` duplicated the public type surface.
- `unified_classifier_types.go` now owns shared public types, global access, stats, and compatibility convenience methods; `unified_classifier_cgo_results.go` owns native batch invocation plus C result decoding; `unified_classifier_stub.go` keeps only stub native-capability behavior. The cgo orchestrator dropped from 660 to 309 lines, and the stub dropped from 179 to 84 lines.
- The slice also fixes two current robustness risks: empty label sets are rejected before native pointer setup, and LoRA/native result decoding no longer uses the old fixed 1000-result unsafe array cap.
- TD020 remains open for `classifier.go` family-adapter boundaries and remaining request-time orchestration splits.
- Validation passed: `go test ./pkg/classification -run 'TestUnifiedClassifier|TestCurrentNativeBackendCapabilitiesShape|TestUnifiedClassifierRejectsUnsupportedNativeCapabilities|TestUnifiedClassifierStatsIncludeNativeCapabilities'`; `go test ./pkg/classification ./pkg/services`.

2026-05-25 ASR007 / TD020 unified-classifier LoRA lifecycle slice:

- Current-source recheck found the narrowed `unified_classifier.go` still owned LoRA C binding declarations, lazy-init locking, mode/capability guards, and logging alongside legacy initialization and batch dispatch.
- `unified_classifier_lora.go` now owns LoRA binding initialization, native capability checks, and lazy-init concurrency guards. `unified_classifier.go` keeps legacy native initialization plus `ClassifyBatch` dispatch only, dropping from 309 to 175 lines; the LoRA lifecycle helper is 116 lines.
- TD020 remains open for broader classification boundaries, but the unified classifier no longer mixes LoRA lifecycle policy into the main cgo batch dispatcher.
- Validation passed: `go test ./pkg/classification -run 'TestUnifiedClassifier|TestCurrentNativeBackendCapabilitiesShape|TestUnifiedClassifierRejectsUnsupportedNativeCapabilities|TestUnifiedClassifierStatsIncludeNativeCapabilities'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/embedding_classifier.go,src/semantic-router/pkg/classification/embedding_classifier_backend.go,src/semantic-router/pkg/classification/embedding_classifier_preload_support.go,src/semantic-router/pkg/classification/embedding_classifier_scoring.go,src/semantic-router/pkg/classification/unified_classifier.go,src/semantic-router/pkg/classification/unified_classifier_lora.go,src/semantic-router/pkg/classification/unified_classifier_cgo_results.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 built-in classifier family slice:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier.go` still mixed constructor/state-table ownership with built-in category, jailbreak, and PII model enablement, initialization, and jailbreak request helpers.
- `src/semantic-router/pkg/classification/classifier_builtin_models.go` now owns those built-in family behaviors. `classifier.go` keeps the dependency table, option wiring, and constructor, dropping from 383 to 187 lines.
- TD020 remains open for broader request-time signal orchestration and any remaining family-specific adapters outside the built-in classifier slice.
- Validation passed: `go test ./pkg/classification ./pkg/services`.

2026-05-25 ASR007 / TD020 request-time signal orchestration slice:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_signal_context.go` had become the live request-time signal hotspot at 783 lines, bundling central dispatcher setup with jailbreak, PII, embedding, and preference signal evaluators.
- `src/semantic-router/pkg/classification/classifier_signal_jailbreak.go` now owns jailbreak signal content deduplication, cached classifier inference, BERT/contrastive rule evaluation, and result mutation.
- `src/semantic-router/pkg/classification/classifier_signal_pii.go` now owns request-time PII signal content fanout, cached token-classification inference, per-rule allow-list checks, and result mutation.
- Existing embedding and preference support files now own their request-time evaluators, so the central `classifier_signal_context.go` keeps dispatcher/readiness/text-selection plus remaining small generic signal evaluators and drops below the 400-line structure warning threshold at 391 lines.
- TD020 remains open for any remaining request-time family adapters and for a future check that `classifier_signal_context.go` can stay below hotspot thresholds as new signals are added.
- Validation passed: `go test ./pkg/classification ./pkg/services`.
- Focused validation passed: `go test ./pkg/classification -run 'TestPII|TestJailbreak|TestSignal|TestEmbedding|TestPreference|TestContext|TestClassifier'`.

2026-05-25 ASR007 / TD020 signal model-family lifecycle slice:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_model_select.go` was still misnamed and over-broad: model-selection helpers shared the file with fact-check, hallucination, feedback, preference, and language model enablement, initialization, runtime wrappers, and accessors.
- `src/semantic-router/pkg/classification/classifier_signal_model_families.go` initially carried those auxiliary signal model-family lifecycle and runtime methods before the follow-up family split below deleted that holding file.
- `src/semantic-router/pkg/classification/classifier_model_select.go` now keeps only decision lookup, category prompt/description helpers, and candidate model selection; it dropped from 423 to 146 lines.
- TD020 remains open for other signal-family hotspots such as large embedding, keyword, MCP, and evaluation/result-contract files, but this slice removes another changed-file structure warning from the main classifier boundary.
- Validation passed: `go test ./pkg/classification ./pkg/services`.

2026-05-25 ASR007 / TD020 model-family lifecycle family split:

- Current-source recheck found the extracted `classifier_signal_model_families.go` had become a second-order hotspot by mixing fact-check, hallucination, feedback, preference, and language lifecycle/readiness/public wrapper methods.
- `classifier_fact_hallucination_lifecycle.go` now owns fact-check and hallucination readiness, initialization, public calls, NLI fallback, and getters.
- `classifier_feedback_lifecycle.go`, `classifier_preference_lifecycle.go`, and `classifier_language_lifecycle.go` now own their respective family readiness, initialization, public calls where applicable, and logging helpers.
- `classifier_signal_model_families.go` was deleted; the replacement files are 158, 55, 62, and 28 lines.
- TD020 remains open for remaining classifier hotspots and dispatcher growth protection, but model-family lifecycle ownership no longer collapses into a cross-family holding file.
- Focused validation passed: `go test ./pkg/classification -run 'Test.*Hallucination|Test.*FactCheck|Test.*Feedback|Test.*Preference|Test.*Language|TestClassifier'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.

2026-05-25 ASR007 / TD020 signal evaluation bridge slice:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_signal_eval.go` still bundled four separate concerns after the evaluator-family split: public signal result types, used-signal dependency expansion, authz header evaluation, and decision-engine bridging.
- `src/semantic-router/pkg/classification/classifier_signal_results.go`, `classifier_signal_usage.go`, `classifier_signal_authz.go`, and `classifier_signal_decision.go` now own those seams separately.
- `classifier_signal_eval.go` now keeps only the two public convenience entrypoints for whole-signal evaluation and dropped from 431 to 13 lines; the extracted bridge files are each under 153 lines.
- TD020 remains open for larger family implementations such as embedding, keyword, MCP, and classifier initialization/constructor boundaries, but the central signal evaluation file no longer acts as a mixed result/authz/decision orchestrator.
- Validation passed: `go test ./pkg/classification ./pkg/services`.
- Focused validation passed: `go test ./pkg/classification -run 'Test.*Signal|Test.*Authz|Test.*Decision|Test.*Projection|Test.*PII|Test.*Jailbreak'`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_signal_eval.go,src/semantic-router/pkg/classification/classifier_signal_authz.go,src/semantic-router/pkg/classification/classifier_signal_decision.go,src/semantic-router/pkg/classification/classifier_signal_results.go,src/semantic-router/pkg/classification/classifier_signal_usage.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 native family init/inference adapter slice:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_init.go` still bundled category, jailbreak, and PII native initializer/inference adapters, mmBERT variants, and API result structs into one cross-family file.
- `src/semantic-router/pkg/classification/classifier_category_init.go`, `classifier_jailbreak_init.go`, and `classifier_pii_init.go` now own those family-specific native adapter seams separately, and the mixed `classifier_init.go` file is removed.
- The deleted 403-line mixed file is replaced by 126-line category, 164-line jailbreak, and 129-line PII files.
- TD020 remains open for larger classifier family implementations and constructor boundaries, but native adapter initialization no longer depends on a shared category/jailbreak/PII hotspot.
- Validation passed: `go test ./pkg/classification ./pkg/services`.
- Focused validation passed: `go test ./pkg/classification -run 'Test.*Classifier|Test.*Jailbreak|Test.*PII|Test.*Native|Test.*Unified'`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_init.go,src/semantic-router/pkg/classification/classifier_category_init.go,src/semantic-router/pkg/classification/classifier_jailbreak_init.go,src/semantic-router/pkg/classification/classifier_pii_init.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 services signal contract slice:

- Current-source recheck found `src/semantic-router/pkg/services/classification_signal_contract.go` still bundled API DTOs, eval execution, intent response shaping, matched-signal extraction, and unmatched-signal collection into one service contract file.
- `classification_signal_types.go`, `classification_signal_response.go`, and `classification_signal_matched.go` now own the DTO surface, response shaping, and signal-set helpers separately.
- `classification_signal_contract.go` now keeps eval request execution and decision-trace evaluation only, dropping from 451 to 69 lines; the extracted helpers are 90, 131, and 180 lines.
- TD020 remains open for larger classification family implementations and constructor boundaries, but service-level signal response shaping is no longer tied to the eval execution path.
- Validation passed: `go test ./pkg/classification ./pkg/services`.
- Focused validation passed: `go test ./pkg/services -run 'Test.*Intent|Test.*Classification|Test.*Signal|Test.*Eval|Test.*Recommended|Test.*Routing'`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/services/classification_signal_contract.go,src/semantic-router/pkg/services/classification_signal_matched.go,src/semantic-router/pkg/services/classification_signal_response.go,src/semantic-router/pkg/services/classification_signal_types.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 embedding classifier backend/scoring/preload slice:

- Current-source recheck found `src/semantic-router/pkg/classification/embedding_classifier.go` still bundled backend FFI variables and model initialization, candidate preload orchestration, text/image request orchestration, matched-rule scoring, top-k shaping, cosine similarity, and prototype-bank maintenance.
- `embedding_classifier_backend.go` now owns embedding FFI indirection, multimodal image/text encode helpers, and keyword-embedding backend initialization. `embedding_classifier_preload_support.go` owns candidate collection, worker fanout, preload result collection, and prototype-bank rebuild triggering. `embedding_classifier_scoring.go` owns rule scoring, hard/soft match selection, output ordering/limiting, cosine similarity, preload stats, and prototype-bank rebuilds.
- `embedding_classifier.go` now keeps classifier state, construction, and text/multimodal classification flow only, dropping from 655 to 323 lines; the backend, preload, and scoring helpers are 130, 137, and 166 lines.
- TD020 remains open for larger embedding, keyword, MCP, hallucination, and constructor-family hotspots, but this slice removes another broad mixed responsibility from the signal runtime path without changing public APIs.
- Validation passed: `go test ./pkg/classification -run 'Test.*Embedding|Test.*Prototype|Test.*Classifier|Test.*Signal|Test.*Multimodal'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/embedding_classifier.go,src/semantic-router/pkg/classification/embedding_classifier_backend.go,src/semantic-router/pkg/classification/embedding_classifier_preload_support.go,src/semantic-router/pkg/classification/embedding_classifier_scoring.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 embedding classifier request-path split:

- Current-source recheck found the narrowed `embedding_classifier.go` still mixed classifier construction/state with public text classification and multimodal request orchestration.
- `embedding_classifier_text.go` now owns text `Classify`, `ClassifyAll`, and detailed text scoring orchestration.
- `embedding_classifier_multimodal.go` now owns multimodal request validation, image-embedding cache use, multimodal scoring orchestration, and unsupported-audio errors.
- `embedding_classifier.go` now keeps construction, state, model-type resolution, rules-by-modality indexing, and result DTOs only, dropping from 323 to 122 lines; the two request-path files are 90 and 91 lines.
- TD020 remains open for residual medium classifier files and dispatcher growth protection, but embedding classification no longer has a single file that mixes constructor/state ownership with request-time execution paths.
- Focused validation passed: `go test ./pkg/classification -run 'TestEmbeddingClassifier|TestSignal.*Embedding|TestClassifier'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.

2026-05-25 ASR007 / TD020 keyword classifier dispatch/matching slice:

- Current-source recheck found `src/semantic-router/pkg/classification/keyword_classifier.go` still bundled classifier construction, Rust-backed method dispatch, per-call BM25/N-gram cache state, regex rule preprocessing, regex/fuzzy matching, Levenshtein scoring, and public wrapper methods.
- `keyword_classifier_regex.go` now owns regex rule preprocessing and pattern construction, `keyword_classifier_dispatch.go` owns ordered first-match dispatch and per-call BM25/N-gram cache state, and `keyword_classifier_match.go` owns regex/fuzzy matching plus Levenshtein distance helpers.
- `keyword_classifier.go` now keeps classifier construction, resource release, and public classify wrappers, dropping from 569 to 165 lines; the extracted helpers are 99, 133, and 190 lines.
- TD020 remains open for MCP, hallucination, and constructor-family hotspots, but keyword matching no longer forces every rule-shape or fuzzy-match edit through the classifier constructor file.
- Validation passed: `go test ./pkg/classification -run 'Test.*Keyword|Test.*BM25|Test.*Ngram|Test.*Fuzzy|Test.*Levenshtein'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/keyword_classifier.go,src/semantic-router/pkg/classification/keyword_classifier_dispatch.go,src/semantic-router/pkg/classification/keyword_classifier_match.go,src/semantic-router/pkg/classification/keyword_classifier_regex.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 MCP classifier client/runtime slice:

- Current-source recheck found `src/semantic-router/pkg/classification/mcp_classifier.go` still bundled MCP client initialization, tool discovery, tool-call response parsing, category mapping construction, Classifier-level initialization, and entropy/metrics runtime classification in one file.
- `mcp_classifier_client.go` now owns the MCP transport/client lifecycle, tool discovery, shared text-response parsing, classification tool calls, probability calls, and `list_categories` mapping conversion.
- `mcp_classifier_runtime.go` now owns Classifier-level MCP enablement, category mapping bootstrap, timeout-scoped probability inference, category-name translation, threshold handling, entropy reasoning, and probability-quality metrics.
- `mcp_classifier.go` now keeps only the public MCP category contracts, classifier struct, factory helpers, and option wiring, dropping from 593 to 80 lines; the client and runtime helpers are 230 and 223 lines.
- TD020 remains open for hallucination, constructor-family, and remaining classification hotspot slices, but MCP protocol and runtime entropy logic no longer share a single edit point.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/mcp_classifier.go,src/semantic-router/pkg/classification/mcp_classifier_client.go,src/semantic-router/pkg/classification/mcp_classifier_runtime.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 hallucination detector NLI slice:

- Current-source recheck found `src/semantic-router/pkg/classification/hallucination_detector.go` still mixed basic hallucination detector lifecycle and span filtering with NLI config, NLI classification, enhanced hallucination result contracts, NLI-based span filtering, severity adjustment, and explanations.
- `hallucination_detector_nli.go` now owns the NLI labels/results, enhanced hallucination result contracts, NLI initialization/classification, enhanced detection, span filtering, threshold defaults, and explanation/severity adjustment.
- `hallucination_detector.go` now keeps detector construction, Candle hallucination-model initialization, basic detection, basic span filtering, and basic readiness only, dropping from 400 to 160 lines; the NLI helper is 248 lines.
- TD020 remains open for constructor-family, KB/category, signal grouping, projection, and remaining classifier hotspots, but hallucination basic detection and enhanced NLI explanation no longer share a single file.
- Focused validation passed: `go test ./pkg/classification -run 'TestHallucination|Test.*Hallucination'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/hallucination_detector.go,src/semantic-router/pkg/classification/hallucination_detector_nli.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 classifier construction option-builder slice:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_construction.go` still mixed classifier-option orchestration, rule-family option construction, category/MCP backend wiring, and native category/jailbreak/PII dependency selection.
- `classifier_option_rules.go` now owns keyword, embedding, context, structure, event-context, reask, complexity, contrastive jailbreak, authz, and KB option builders.
- `classifier_option_backends.go` now owns category/MCP option wiring plus native jailbreak and PII dependency selection.
- `classifier_construction.go` now keeps construction orchestration, parallel option assembly, multimodal initialization, default embedding model selection, and heuristic initialization logging only, dropping from 398 to 135 lines.
- TD020 remains open for KB/category scoring, signal grouping, projection, discovery/bootstrap contracts, and long-term dispatcher-growth protection.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_construction.go,src/semantic-router/pkg/classification/classifier_option_rules.go,src/semantic-router/pkg/classification/classifier_option_backends.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 KB classifier embedding/scoring slice:

- Current-source recheck found `src/semantic-router/pkg/classification/category_kb_classifier.go` still mixed KB manifest loading, exemplar embedding preload, prototype-bank rebuilds, query classification, label/group matching, metric calculation, and result shaping.
- `category_kb_embeddings.go` now owns exemplar ref collection, parallel embedding preload, preload telemetry, and per-label prototype-bank rebuilds.
- `category_kb_scoring.go` now owns prototype label scoring, thresholded label matching, group scoring/matching, metric calculation, and `KBClassifyResult` shaping.
- `category_kb_classifier.go` now keeps the public KB classifier type, manifest loading, constructor, query embedding/classify orchestration, and label count only, dropping from 384 to 121 lines.
- TD020 remains open for category entropy, signal grouping, projection, discovery/bootstrap contracts, and long-term dispatcher-growth protection.
- Focused validation passed: `go test ./pkg/classification -run 'TestKnowledgeBase|Test.*KB|Test.*Prototype'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/category_kb_classifier.go,src/semantic-router/pkg/classification/category_kb_embeddings.go,src/semantic-router/pkg/classification/category_kb_scoring.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 modality/category entropy split:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_category_entropy.go` still mixed modality detection with category entropy routing, probability-distribution reasoning, fallback handling, and metrics.
- `classifier_modality.go` now owns modality result contracts plus classifier, keyword, and hybrid modality detection.
- `classifier_category_entropy.go` now keeps category-with-entropy orchestration, keyword/embedding category fallback, in-tree probability classification, reasoning-decision construction, fallback category selection, and entropy metrics only, dropping from 381 to 239 lines.
- TD020 remains open for signal grouping, projection, discovery/bootstrap contracts, and long-term dispatcher-growth protection.
- Focused validation passed: `go test ./pkg/classification -run 'Test.*Modality|Test.*Entropy|Test.*Category|Test.*Reasoning'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_category_entropy.go,src/semantic-router/pkg/classification/classifier_modality.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 request-time generic signal evaluator split:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_signal_context.go` had regrown past a narrow context role: it still owned signal readiness, request orchestration, and keyword/domain/fact-check/user-feedback/reask/context/complexity/modality evaluator implementations.
- `classifier_signal_rule_evaluators.go` now owns keyword, domain, fact-check, user-feedback, reask, and context evaluator implementations.
- `classifier_signal_complexity.go` owns complexity result mutation and metric emission, while `classifier_signal_modality_eval.go` owns request-time modality signal result mutation and metric emission.
- `classifier_signal_context.go` now keeps readiness, text selection, image-cache setup, dispatcher execution, and post-processing only, dropping from 391 to 102 lines.
- TD020 remains open for signal grouping, projection, discovery/bootstrap contracts, and long-term dispatcher-growth protection, but the central request-time evaluator file no longer has to absorb unrelated signal families.
- Focused validation passed: `go test ./pkg/classification -run 'Test.*Signal|Test.*Modality|Test.*Complexity|Test.*Feedback|Test.*Domain|Test.*Context|Test.*Reask'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_signal_context.go,src/semantic-router/pkg/classification/classifier_signal_rule_evaluators.go,src/semantic-router/pkg/classification/classifier_signal_complexity.go,src/semantic-router/pkg/classification/classifier_signal_modality_eval.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 signal group resolution/trace/output-policy split:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_signal_groups.go` still mixed partition application, member resolution, winner/default selection, softmax math, projection trace construction, and embedding top-k output policy.
- `classifier_signal_group_resolution.go` now owns group winner/default selection and softmax scoring helpers.
- `classifier_signal_group_trace.go` now owns partition trace entry construction, raw winner scores, and margin calculation.
- `classifier_signal_output_policy.go` now owns matched-signal output limiting, including embedding top-k pruning.
- `classifier_signal_groups.go` now keeps signal-group entrypoints and per-signal member resolution only, dropping from 371 to 68 lines.
- TD020 remains open for projection, discovery/bootstrap contracts, and long-term dispatcher/group-growth protection, but partition resolution and trace behavior no longer share the group application entrypoint file.
- Focused validation passed: `go test ./pkg/classification -run 'TestSignalGroup|TestApplySignalGroups|TestAnalyzeSoftmaxSignalGroupCentroids|Test.*Projection|Test.*Partition|Test.*TopK'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_signal_groups.go,src/semantic-router/pkg/classification/classifier_signal_group_resolution.go,src/semantic-router/pkg/classification/classifier_signal_group_trace.go,src/semantic-router/pkg/classification/classifier_signal_output_policy.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 projection ordering/input/output/trace split:

- Current-source recheck found `src/semantic-router/pkg/classification/classifier_projections.go` still mixed projection execution, score dependency ordering, input accessor/value matching, output matching/confidence math, boundary distance calculation, and projection trace merging.
- `classifier_projection_order.go` now owns score dependency detection and topological ordering.
- `classifier_projection_inputs.go` now owns projection input accessors, typed value reads, match-set construction, and input matching.
- `classifier_projection_outputs.go` now owns output rule matching, confidence selection, and boundary-distance calculation.
- `classifier_projection_trace.go` now owns projection trace merging.
- `classifier_projections.go` now keeps the projection execution entrypoint only, dropping from 366 to 43 lines.
- TD020 remains open for discovery/bootstrap contracts and long-term dispatcher/group-growth protection, but projection behavior no longer lives in one cross-cutting helper.
- Focused validation passed: `go test ./pkg/classification -run 'TestApplyProjections|TestProjection|Test.*Projection|Test.*Partition'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/classifier_projections.go,src/semantic-router/pkg/classification/classifier_projection_order.go,src/semantic-router/pkg/classification/classifier_projection_inputs.go,src/semantic-router/pkg/classification/classifier_projection_outputs.go,src/semantic-router/pkg/classification/classifier_projection_trace.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 model discovery validation/info/auto-init split:

- Current-source recheck found `src/semantic-router/pkg/classification/model_discovery.go` still mixed model path types, directory discovery, architecture detection, path validation, discovery-info response shaping, and unified classifier auto-initialization.
- `model_discovery_validation.go` now owns discovered model path validation and model-directory file checks.
- `model_discovery_info.go` now owns discovery-info response shaping and missing-model reporting.
- `unified_auto_init.go` now owns auto-discover plus LoRA/legacy unified classifier initialization.
- `model_discovery.go` now keeps model path contracts, architecture models, public discovery entrypoints, and architecture detection only, dropping from 316 to 87 lines.
- TD020 remains open for long-term dispatcher/growth protection and remaining family hotspots, but model discovery no longer owns validation/reporting/bootstrap side effects.
- Focused validation passed: `go test ./pkg/classification -run 'TestAutoDiscoverModels|TestValidateModelPaths|TestModelPathsIsComplete|TestGetModelDiscoveryInfo|TestLoadLegacyUnifiedLabels|TestUnifiedClassifier'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/model_discovery.go,src/semantic-router/pkg/classification/model_discovery_validation.go,src/semantic-router/pkg/classification/model_discovery_info.go,src/semantic-router/pkg/classification/unified_auto_init.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 vLLM jailbreak parser split:

- Current-source recheck found `src/semantic-router/pkg/classification/vllm_classifier.go` still mixed vLLM request construction, timeout/client handling, response-to-`ClassResult` mapping, parser selection, Qwen3Guard parsing, JSON parsing, simple keyword parsing, and category extraction.
- `vllm_jailbreak_parser.go` now owns parser selection, auto fallback, Qwen3Guard safety/severity/category parsing, JSON parsing, simple parsing, and category extraction.
- `vllm_classifier.go` now keeps vLLM client setup and remote classification orchestration only, dropping from 325 to 116 lines.
- `vllm_jailbreak_parser_test.go` covers parser-type selection, Qwen3Guard safety/severity/category parsing, JSON parsing, simple parsing, and auto fallback.
- TD020 remains open for remaining family hotspots, but remote inference transport no longer shares one file with safety-output parser policy.
- Focused validation passed: `go test ./pkg/classification -run 'TestVLLMJailbreakParser|TestNewVLLMJailbreakInference'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/vllm_classifier.go,src/semantic-router/pkg/classification/vllm_jailbreak_parser.go,src/semantic-router/pkg/classification/vllm_jailbreak_parser_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 contrastive preference split:

- Current-source recheck found `src/semantic-router/pkg/classification/contrastive_preference_classifier.go` still mixed public preference classifier construction, concurrent rule embedding preload, query scoring, threshold/margin decisioning, example collection, and prototype-bank rebuilds.
- `contrastive_preference_embeddings.go` now owns concurrent rule-example embedding tasks, worker fanout, preload telemetry, and loaded embedding collection.
- `contrastive_preference_scoring.go` now owns detailed query scoring, deterministic score ordering, example collection, and prototype-bank rebuilds.
- `contrastive_preference_classifier.go` now keeps public types, construction/configuration, and `Classify` threshold/margin decisioning only, dropping from 356 to 126 lines.
- TD020 remains open for remaining family hotspots, but preference embedding preload and scoring no longer share the public classifier entrypoint file.
- Focused validation passed: `go test ./pkg/classification -run 'Test.*Preference|TestContrastivePreference'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/contrastive_preference_classifier.go,src/semantic-router/pkg/classification/contrastive_preference_embeddings.go,src/semantic-router/pkg/classification/contrastive_preference_scoring.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 preference external/contrastive wrapper split:

- Current-source recheck found `src/semantic-router/pkg/classification/preference_classifier.go` still mixed public preference classifier construction, external LLM config/client/prompt setup, external response parsing, contrastive conversation-text extraction, and runtime dispatch.
- `preference_classifier_external.go` now owns external LLM preference construction, default prompt contract, route JSON shaping, LLM call orchestration, and tolerant route-output parsing.
- `preference_classifier_contrastive.go` now owns contrastive construction and conversation-message text extraction before delegating into the contrastive scorer.
- `preference_classifier.go` now keeps only the public result/classifier type, top-level constructor dispatch, runtime dispatch, and readiness check, dropping from 266 to 65 lines; parser compatibility now has focused coverage for plain JSON, prose-wrapped JSON, single-quoted JSON, missing JSON, and empty route output.
- TD020 remains open for remaining classifier hotspots, but preference routing no longer forces external LLM prompt/parser edits through the public classifier wrapper.
- Focused validation passed: `go test ./pkg/classification -run 'Test.*Preference|TestContrastivePreference'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.

2026-05-25 ASR007 / TD020 complexity classifier support split:

- Current-source recheck found `src/semantic-router/pkg/classification/complexity_classifier_support.go` still mixed candidate embedding preload, query embedding loading, text/image rule scoring, signal fusion, difficulty labeling, and logging, while `complexity_classifier.go` still owned preload orchestration and prototype-bank rebuilds.
- `complexity_candidate_embeddings.go` now owns candidate task construction, worker fanout, embedding result collection, and text/image prototype-bank rebuilds.
- `complexity_query_embeddings.go` now owns text, multimodal text, and optional request-image embedding loading; it also handles public image classification without requiring a non-nil request cache.
- `complexity_rule_scoring.go` now owns per-rule text/image scoring, signal fusion, difficulty labeling, and result logging.
- `complexity_classifier.go` now keeps public types, construction, and request-level classify orchestration only, dropping from 222 to 148 lines; the deleted 330-line mixed support file is replaced by focused 250/60/116-line modules.
- `complexity_classifier_test.go` now covers public image classification without an injected request image cache.
- TD020 remains open for remaining family hotspots, but complexity candidate preload, query embedding, and scoring no longer share one support file.
- Focused validation passed: `go test ./pkg/classification -run 'Test.*Complexity|TestEvaluateEmbeddingSignal_SharedCacheDedupsAcrossDivergentTargetDims|TestEvaluateEmbeddingAndComplexitySignalsShareRequestImageCache'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/classification/complexity_classifier.go,src/semantic-router/pkg/classification/complexity_classifier_support.go,src/semantic-router/pkg/classification/complexity_candidate_embeddings.go,src/semantic-router/pkg/classification/complexity_query_embeddings.go,src/semantic-router/pkg/classification/complexity_rule_scoring.go,src/semantic-router/pkg/classification/complexity_classifier_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR007 / TD020 shared prototype-bank split:

- Current-source recheck found `src/semantic-router/pkg/classification/prototype_scoring.go` still mixed shared prototype-bank contracts, example dedupe, similarity-matrix construction, clustering/medoid selection, representative copying, and runtime score aggregation for embedding, preference, complexity, and KB classifiers.
- `prototype_bank.go` now owns prototype example/representative/bank contracts, bank construction, uncompressed representatives, ordering, and safe representative copies.
- `prototype_clustering.go` now owns example dedupe, similarity-matrix construction, greedy clustering, cluster tie-breaking, member removal, and medoid selection.
- `prototype_scoring.go` now owns only score options and runtime query-to-bank aggregation, dropping from 281 to 66 lines; the extracted bank and clustering helpers are 90 and 145 lines.
- TD020 remains open for remaining classifier hotspots and long-term growth protection, but shared prototype construction/compression no longer shares an edit point with request-time scoring.
- Focused validation passed: `go test ./pkg/classification -run 'TestPrototypeBank|TestEmbeddingClassifier|TestContrastivePreference|TestComplexityClassifier|TestKnowledgeBase'`.
- Package validation passed: `go test ./pkg/classification ./pkg/services`.

2026-05-24 ASR008 / TD027 partial:

- Current-source recheck confirmed TD027 remains open, but the `threshold_pareto` dataclass, sweep, Pareto marking, and report helper were a self-contained slice still living in `src/fleet-sim/fleet_sim/optimizer/base.py`.
- `src/fleet-sim/fleet_sim/optimizer/threshold.py` now owns threshold Pareto analysis, while `optimizer/__init__.py` exports the public optimizer seam directly from that module.
- `src/fleet-sim/fleet_sim/optimizer/base.py` keeps compatibility exports for callers that still import threshold helpers from `fleet_sim.optimizer.base`.
- TD027 remains open for DES/reporting boundaries, root-package export curation, and further optimizer hotspot reduction.
- Validation passed: `make vllm-sr-sim-test`.

2026-05-24 ASR008 / TD028 partial:

- Current-source recheck confirmed TD028 remains open, but the controller-side `canonical_config_builder.go` slice was still collapsing base config assembly, backend discovery fan-in, CRD spec-family translation, and generic YAML conversion helpers.
- `deploy/operator/controllers/canonical_config_builder.go` now keeps only base canonical config construction and orchestration, dropping from 314 to 58 lines.
- `deploy/operator/controllers/canonical_config_backends.go` now owns discovered vLLM backend fan-in and LoRA adapter conversion, while `deploy/operator/controllers/canonical_config_spec.go` owns CRD spec-family translation into canonical config.
- `deploy/operator/controllers/canonical_config_spec_test.go` covers strategy, reasoning family defaults, tools, classifier modules, and complexity composer translation through the controller-owned canonical config seam.
- TD028 remains open for CRD schema hotspot size, webhook validation-family extraction, and sample-fixture ownership.
- Validation passed: `go test ./controllers -run 'TestBuildCanonicalConfigAppliesOperatorSpecFamilies|TestGenerateConfigYAMLIncludesLoRACatalogFromVLLMEndpoints'` and `go test ./...` from `deploy/operator`.
- Validation passed: `make generate-crd`, with no CRD or Helm CRD drift.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="deploy/operator/controllers/canonical_config_builder.go,deploy/operator/controllers/canonical_config_backends.go,deploy/operator/controllers/canonical_config_spec.go,deploy/operator/controllers/canonical_config_spec_test.go"`.

2026-05-24 ASR006 / TD031 model-selector registry slice:

- Current-source recheck found the older TD031 wording partly stale because steady-state classification, memory, vector-store, file-store, and config fanout already flow through `pkg/routerruntime`; the live gap in this slice was API handlers still resolving model-selection state directly from `selection.GlobalRegistry`.
- A follow-up config-update fanout recheck found the old single-consumer wording stale as well: `config.SubscribeConfigUpdates` owns per-subscriber buffered channels, `config.Replace` fans out to all current subscribers, and `WatchConfigUpdates` is retained as a compatibility wrapper.
- `src/semantic-router/pkg/routerruntime/registry.go` now owns an optional model-selection registry alongside config, classification, memory, and vector-store runtime dependencies.
- `src/semantic-router/pkg/extproc/server.go` publishes `OpenAIRouter.ModelSelector` into the runtime registry on startup and reload.
- `src/semantic-router/pkg/apiserver/runtime_dependencies.go`, `route_feedback.go`, and `server.go` now make feedback, ratings, and RL-state handlers prefer the active runtime registry before falling back to the legacy global registry.
- `src/semantic-router/pkg/routerruntime/registry_test.go`, `src/semantic-router/pkg/extproc/server_reload_test.go`, and `src/semantic-router/pkg/apiserver/route_feedback_test.go` cover model-selector publication and API reads from runtime-registry state.
- TD031 remains open for `config.Get` / `config.Replace`, compatibility global service accessors, selection package global fallback, and broad bootstrap lifecycle ownership.
- Validation passed: `go test ./pkg/config -run TestSubscribeConfigUpdatesFanout -count=1`; `go test ./pkg/routerruntime`; `go test ./pkg/apiserver -run 'TestHandleFeedback|TestHandleGetRatings|TestRuntimeRegistryResolvesSharedDependencies'`; `go test ./pkg/extproc -run TestReloadRouterFromConfigPublishesRuntimeRegistryAfterSwap`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="..."`, including `make agent-validate`, `make vllm-sr-test`, `make vllm-sr-sim-test`, `make test-semantic-router`, and `make dashboard-check`.
- Feature integration and local smoke remain blocked by environment because `docker info` still cannot connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-24 ASR006 / TD031-TD034 memory-store runtime boundary slice:

- Current-source recheck found the memory-store portion narrower than broad shared-state debt: steady-state ExtProc startup already publishes `OpenAIRouter.MemoryStore` into `routerruntime.Registry`, but `apiserver.InitWithRuntime` still fell back to `memory.GetGlobalMemoryStore()` when the registry had not yet published a store.
- `src/semantic-router/pkg/apiserver/server.go` now keeps the runtime-owned path strict: when a registry is provided, memory API startup reads only `runtimeRegistry.MemoryStore()` and waits for later request-time lookup through `currentMemoryStore()`; only the legacy nil-registry `Init` path keeps the process-wide global fallback.
- `src/semantic-router/pkg/apiserver/server_memory_test.go` covers both the runtime-registry-only path and the preserved legacy global fallback.
- TD031 and TD034 remain open for broader compatibility globals, production shared-store policy, selector shared-store parity beyond local files, and durable config projections.
- Validation passed: `go test ./pkg/apiserver -run 'TestResolveMemoryStoreUsesRuntimeRegistryOnly|TestResolveMemoryStorePreservesLegacyGlobalFallback|TestRuntimeRegistryResolvesSharedDependencies'`; `go test ./pkg/routerruntime`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/apiserver/server.go,src/semantic-router/pkg/apiserver/server_memory_test.go,docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md"`, including `make agent-validate` and `make test-semantic-router`.
- Validation passed: `make agent-scorecard`; `git diff --check`.
- Local dev/smoke remains blocked by environment: `docker info` and `make agent-dev ENV=cpu` both failed because Docker cannot connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-25 ASR006 / TD031-TD034 API runtime dependency boundary slice:

- Current-source recheck found the memory-store runtime-boundary fix had not been applied consistently to API vector/file/selector helpers: an API server with a `routerruntime.Registry` could still fall back to legacy process-wide vector manager, embedder, ingestion pipeline, file store, or `selection.GlobalRegistry` before runtime publication.
- `src/semantic-router/pkg/apiserver/runtime_dependencies.go` now treats a present runtime registry as authoritative for these helpers. It returns registry-published dependencies when present and nil before publication; legacy process-wide fallback remains only for nil-registry callers.
- `src/semantic-router/pkg/apiserver/runtime_globals_test.go` covers both compatibility paths: nil-registry servers still resolve legacy globals, while registry-backed servers suppress legacy vector/file/selector globals until runtime publication.
- TD031 and TD034 remain open for `config.Get` / `config.Replace`, broad bootstrap lifecycle ownership, shared selector-store parity beyond local files, production session-store policy, and durable config projections.
- Validation passed: `go test ./pkg/apiserver -run 'TestLegacyRuntimeGlobalsResolveThroughSynchronizedAccessors|TestRuntimeRegistrySuppressesLegacyVectorGlobalsUntilPublished|TestRuntimeRegistrySuppressesLegacySelectionGlobalUntilPublished|TestHandleFeedbackUsesRuntimeSelectionRegistry|TestHandleGetRatingsUsesRuntimeSelectionRegistry|TestResolveMemoryStoreUsesRuntimeRegistryOnly'`; `go test ./pkg/routerruntime`.
- Package validation passed: `go test ./pkg/apiserver`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/apiserver/runtime_dependencies.go,src/semantic-router/pkg/apiserver/runtime_globals_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
- Validation passed: `make agent-scorecard`; `git diff --check`.
- Local smoke remains blocked by environment: `docker info` still cannot connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-25 ASR006 / TD031-TD034 API startup config resolver slice:

- Current-source recheck found one remaining `InitWithRuntime` startup-state leak: `resolveAPIServerConfig` could still fall back to `config.Get()` when a runtime registry existed but had not published its current config.
- `src/semantic-router/pkg/apiserver/server.go` now treats a present runtime registry as authoritative for API startup config, returning only `runtimeRegistry.CurrentConfig()` on the registry path while preserving `config.Get()` for the legacy nil-registry entrypoint.
- `src/semantic-router/pkg/apiserver/server_memory_test.go` covers registry-only config resolution before and after runtime publication plus the preserved legacy global fallback.
- TD031 and TD034 remain open for fully retiring compatibility `config.Get` / `config.Replace` callers, broad bootstrap lifecycle ownership, production shared-store policy, shared selector-store parity beyond local files, and durable deployed-config projections.
- Validation passed: `go test ./pkg/apiserver -run 'TestResolveAPIServerConfigUsesRuntimeRegistryOnly|TestResolveAPIServerConfigPreservesLegacyGlobalFallback|TestResolveMemoryStoreUsesRuntimeRegistryOnly|TestResolveMemoryStorePreservesLegacyGlobalFallback'`.
- Package validation passed: `go test ./pkg/apiserver`; `go test ./pkg/routerruntime`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/apiserver/server.go,src/semantic-router/pkg/apiserver/server_memory_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
- Validation passed: `make agent-scorecard`; `git diff --check`.
- Local smoke remains blocked by environment: `docker info` still cannot connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-25 ASR006 / TD031-TD034 classification refresh global-config slice:

- Current-source recheck found a separate live global-state leak in the API runtime config updater: registry-backed `liveRuntimeConfig.Update()` called `routerruntime.Registry.RefreshRuntimeConfig`, which refreshed the classification service and indirectly called `config.Replace()` through `ClassificationService.RefreshRuntimeConfig`.
- `src/semantic-router/pkg/services/classification_runtime_config.go` now keeps `RefreshRuntimeConfig` service-local. The preserved legacy global publication path is explicit in `buildConfigUpdater(nil, ...)`, and `ClassificationService.UpdateConfig` still keeps its compatibility behavior.
- `src/semantic-router/pkg/services/classification_update_test.go` covers service-local refresh without replacing `config.Get()`, while `src/semantic-router/pkg/apiserver/server_memory_test.go` covers both legacy nil-registry global publication and registry-backed updates that refresh registry/service state without rewriting the process-wide config.
- TD031 and TD034 remain open for fully retiring compatibility globals, broad bootstrap lifecycle ownership, production shared-store policy, shared selector-store parity beyond local files, and durable deployed-config projections.
- Validation passed: `go test ./pkg/services -run 'TestClassificationServiceRefreshRuntimeConfig'`; `go test ./pkg/apiserver -run 'TestBuildConfigUpdater|TestResolveAPIServerConfig|TestResolveMemoryStore'`.
- Package validation passed: `go test ./pkg/services ./pkg/apiserver ./pkg/routerruntime`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/services/classification_runtime_config.go,src/semantic-router/pkg/services/classification_update_test.go,src/semantic-router/pkg/apiserver/server.go,src/semantic-router/pkg/apiserver/server_memory_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.
- Validation passed: `make agent-scorecard`; `git diff --check`.
- Local smoke remains blocked by environment: `docker info` still cannot connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-25 ASR006 / TD031-TD034 RL-driven looper runtime selector slice:

- Current-source recheck found another stale global-state path outside API handlers: `looper.NewRLDrivenLooper` read `selection.GlobalRegistry`, while the ExtProc runtime already owns the current router selector registry through `OpenAIRouter.ModelSelector`.
- `src/semantic-router/pkg/looper/looper.go` now exposes `FactoryWithSelectionRegistry`, and `src/semantic-router/pkg/looper/rl_driven.go` resolves RL-driven selectors from the provided runtime registry. The existing `Factory` / `NewRLDrivenLooper` path keeps global-registry compatibility for direct legacy callers.
- `src/semantic-router/pkg/extproc/req_filter_looper.go` now creates loopers with `r.ModelSelector`, so RL-driven looper execution cannot accidentally bind to selector state from another router/runtime in the same process.
- `src/semantic-router/pkg/looper/rl_driven_test.go` covers the runtime-registry path choosing the provided RL-driven selector even when `selection.GlobalRegistry` contains a different selector, plus the preserved global compatibility path.
- TD031 and TD034 remain open for fully retiring compatibility globals, broad bootstrap lifecycle ownership, production shared-store policy, shared selector-store parity beyond local files, and durable deployed-config projections.
- Validation passed: `go test ./pkg/looper -run 'TestFactory_RLDriven|TestFactoryWithSelectionRegistryUsesRuntimeRLDrivenSelector|TestNewRLDrivenLooperPreservesGlobalRegistryCompatibility|TestRLDrivenLooper_SelectorIntegration'`; `go test ./pkg/extproc -run 'TestShouldUseLooper|TestCreateLooperResponseIncludesTrackedHeaders'`.
- Package validation passed: `go test ./pkg/looper`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/looper/looper.go,src/semantic-router/pkg/looper/rl_driven.go,src/semantic-router/pkg/looper/rl_driven_test.go,src/semantic-router/pkg/extproc/req_filter_looper.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR006 / TD031-TD034 ExtProc startup config precedence slice:

- Current-source recheck found another live `config.Get()` fallback in the main runtime path: `Server.Start()` sized gRPC receive/send buffers from the process-wide config even though the server already had a current router and optional runtime registry.
- `src/semantic-router/pkg/extproc/server.go` now resolves gRPC max-message-size config from the active `RouterService` router config first, then from the runtime registry, and only then from the legacy process-wide config fallback.
- `src/semantic-router/pkg/extproc/server_reload_test.go` covers router-config precedence, runtime-registry precedence, and the preserved legacy global fallback by setting the process-wide config to a different value in each test.
- TD031 and TD034 remain open for fully retiring compatibility globals, broad bootstrap lifecycle ownership, production shared-store policy, shared selector-store parity beyond local files, and durable deployed-config projections.
- Validation passed: `go test ./pkg/extproc -run 'TestConfiguredGRPCMaxMessageSize'`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_reload_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR006 / TD031-TD034 ExtProc request-time plugin config precedence slice:

- Current-source recheck found two request-time config reads that were still live: `handleModalityFromDecision` read `config.Get()` directly, and system-prompt injection preferred the process-wide config before the router instance while also dereferencing a missing decision before checking nil.
- `src/semantic-router/pkg/extproc/router_config.go` now centralizes router-owned config and decision resolution. Request-time modality and system-prompt paths use the router instance first, fall back to classifier-local config, and keep the process-wide config only for legacy nil-router-config callers.
- `src/semantic-router/pkg/extproc/req_filter_modality_test.go` covers router config winning over a conflicting global detector setting and global detector settings not leaking into a router with modality disabled.
- `src/semantic-router/pkg/extproc/req_filter_sys_prompt_test.go` covers router-owned system prompt config winning over a conflicting global decision, no global fallback when the router config is present but missing the decision, and the missing-decision no-panic path.
- TD031 and TD034 remain open for compatibility-global retirement beyond legacy fallbacks, broad bootstrap lifecycle ownership, production shared-store policy, and durable deployed-config projections.
- Validation passed: `go test ./pkg/extproc -run 'TestHandleModalityFromDecision|TestAddSystemPrompt'`.
- Changed-set validation passed after request-time plugin and config-source watcher slices: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_reload_test.go,src/semantic-router/pkg/extproc/req_filter_modality.go,src/semantic-router/pkg/extproc/req_filter_modality_test.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR006 / TD031-TD034 ExtProc config-source watcher precedence slice:

- Current-source recheck found one more startup-time `config.Get()` read in `watchConfigAndReload`: the server chose file watching versus Kubernetes update watching from the process-wide config even when an active router/runtime config existed.
- `src/semantic-router/pkg/extproc/server.go` now uses `usesKubernetesConfigSource`, which shares the same router-config, runtime-registry, legacy-global precedence as gRPC max-message-size resolution.
- `src/semantic-router/pkg/extproc/server_reload_test.go` covers router config overriding conflicting runtime/global source, runtime registry source winning over global fallback, and preserved legacy global fallback.
- TD031 and TD034 remain open for compatibility-global retirement beyond legacy fallbacks, broad bootstrap lifecycle ownership, production shared-store policy, and durable deployed-config projections.
- Validation passed: `go test ./pkg/extproc -run 'TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource'`.
- Changed-set validation passed after request-time plugin and config-source watcher slices: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_reload_test.go,src/semantic-router/pkg/extproc/req_filter_modality.go,src/semantic-router/pkg/extproc/req_filter_modality_test.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR006 / TD031-TD034 ExtProc initial router config precedence slice:

- Current-source recheck found the same runtime-boundary issue one step earlier in `NewServer`: even when a `routerruntime.Registry` already carried the active config, server construction still entered `NewOpenAIRouter(configPath)`, which parsed through the legacy path and republished `config.Replace(cfg)`.
- `src/semantic-router/pkg/extproc/router_build.go` now has `resolveInitialRouterConfig` and `newOpenAIRouterForServer`. Server construction uses the runtime registry config first; if a registry exists but has not published config yet, it parses the explicit config file without entering the legacy Kubernetes-global fallback. Only legacy nil-registry callers publish the process-wide config.
- A follow-up server-config resolver recheck tightened the same edge at runtime: if a server has a runtime registry but that registry has no config yet, `resolveServerConfig` no longer falls through to `config.Get()`. gRPC startup uses the explicit default message size in that pre-publication state, and config-source watcher selection remains non-Kubernetes instead of adopting a conflicting global config.
- A follow-up reload recheck found the same process-wide publication risk after file reload: `reloadRouterFromConfig("file", ...)` still called `replaceReloadConfig` even when `Server.runtime` was present. Registry-backed reload now publishes the new router state through `routerruntime.Registry` only; the `config.Replace` compatibility publication remains reserved for legacy nil-registry reloads.
- A request-time helper recheck found `OpenAIRouter.routerConfig()` and `decisionByName()` could still fall through to `config.Get()` when a router carried a runtime registry but no local `Config` pointer yet. The helper now treats a present runtime registry as authoritative: it uses the registry config when available and returns nil before publication instead of adopting a conflicting process-wide config.
- `src/semantic-router/pkg/extproc/server_reload_test.go` covers registry config winning over a conflicting Kubernetes global without parsing a missing file or changing `config.Get()` plus the preserved legacy Kubernetes global fallback, `src/semantic-router/pkg/extproc/router_build_runtime_config_test.go` covers empty-registry file parsing without adopting a conflicting global, and `src/semantic-router/pkg/extproc/router_config_test.go` covers request-time helper registry/global precedence.
- TD031 and TD034 remain open for full compatibility-global retirement, broad bootstrap lifecycle ownership, production shared-store policy, and durable deployed-config projections.
- Validation passed: `go test ./pkg/extproc -run 'TestResolveInitialRouterConfig|TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource'`; follow-up focused validation passed: `go test ./pkg/extproc -run 'TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource|TestResolveInitialRouterConfig' -count=1`; reload focused validation passed: `go test ./pkg/extproc -run 'TestReloadRouterFromFile|TestReloadRouterFromConfig|TestResolveInitialRouterConfig|TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource' -count=1`; request-time helper validation passed: `go test ./pkg/extproc -run 'TestRouterConfig|TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource|TestResolveInitialRouterConfig|TestReloadRouterFromConfig' -count=1`.
- Changed-set validation passed after the ExtProc initial config precedence slice: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_build.go,src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_reload_test.go,src/semantic-router/pkg/extproc/req_filter_modality.go,src/semantic-router/pkg/extproc/req_filter_modality_test.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt.go,src/semantic-router/pkg/extproc/req_filter_sys_prompt_test.go,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`, including `make agent-validate` and `make test-semantic-router`.

2026-05-25 ASR006 / TD031-TD006 ExtProc watcher ownership split:

- Current-source recheck found the changed-set structure gate still reported `src/semantic-router/pkg/extproc/server.go` as a 542-line hotspot and flagged the 132-line, deeply nested `watchConfigAndReload` function.
- `src/semantic-router/pkg/extproc/server_config_watch.go` now owns file watcher setup, event filtering, debounce scheduling, file-stat logging, reload success/failure logging, and Kubernetes config-update handling. `server.go` keeps core server lifecycle, startup config resolution, reload composition, and runtime publication, dropping to 347 lines.
- `src/semantic-router/pkg/extproc/server_watch_config_test.go` still covers config-event filtering, and focused validation passed: `go test ./pkg/extproc -run 'TestShouldReloadForConfigEvent|TestRouterConfig|TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource|TestResolveInitialRouterConfig|TestReloadRouterFromFile|TestReloadRouterFromConfig' -count=1`.
- Changed-set validation passed after the watcher ownership split: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/router_config_test.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_config_watch.go,src/semantic-router/pkg/extproc/server_reload_test.go,docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md"`, including `make agent-validate` and `make test-semantic-router`.
- TD006 remains open for other legacy hotspots, and TD031 remains open for broader compatibility-global retirement, but ExtProc server lifecycle no longer carries the watch/reload event loop hotspot.

2026-05-25 ASR006 / TD006 ExtProc runtime/reload test ownership split:

- Current-source recheck found the changed-set structure gate still reported `src/semantic-router/pkg/extproc/server_reload_test.go` as a 683-line regression-suite hotspot after the production watcher loop was extracted.
- `src/semantic-router/pkg/extproc/server_config_runtime_test.go` now owns gRPC message-size, config-source, and initial-router config precedence tests; `src/semantic-router/pkg/extproc/server_reload_runtime_test.go` owns reload-source, runtime-preparation failure, and runtime-registry publication tests. `src/semantic-router/pkg/extproc/server_reload_test.go` now stays focused on file reload order, AMD model preflight, and reload seam helpers, dropping to 283 lines.
- Focused validation passed: `go test ./pkg/extproc -run 'TestConfiguredGRPCMaxMessageSize|TestUsesKubernetesConfigSource|TestResolveInitialRouterConfig|TestReloadRouterFromFile|TestReloadRouterFromConfig|TestShouldReloadForConfigEvent|TestRouterConfig' -count=1`.
- Changed-set validation passed after the test ownership split: `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/router_config.go,src/semantic-router/pkg/extproc/router_config_test.go,src/semantic-router/pkg/extproc/server.go,src/semantic-router/pkg/extproc/server_config_watch.go,src/semantic-router/pkg/extproc/server_config_runtime_test.go,src/semantic-router/pkg/extproc/server_reload_test.go,src/semantic-router/pkg/extproc/server_reload_runtime_test.go,docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md,docs/agent/tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/state-taxonomy-and-inventory.md"`, including `make agent-validate`, changed-file lint, Go structural lint, structure checks, and `make test-semantic-router`.

2026-05-25 ASR006 / TD031-TD034 ExtProc vectorstore RAG runtime dependency guard:

- Current-source recheck found the older RAG vectorstore wording partly stale: request-time retrieval already calls `OpenAIRouter.currentVectorStoreManager()` / `currentVectorStoreEmbedder()`, which read from `routerruntime.Registry` rather than API-server globals.
- `router_runtime_services_test.go` now locks that boundary: nil routers or routers without a runtime registry return nil, registry-backed routers return only registry-published manager/embedder dependencies, and no compatibility global is consulted.
- `req_filter_rag_vectorstore.go` and response-memory naming were updated to say router-owned/runtime-owned dependencies rather than "global" state where the code is already router-local.
- TD031 and TD034 remain open for compatibility-global retirement, broad bootstrap lifecycle ownership, production shared-store policy, and durable deployed-config projections, but vectorstore RAG is no longer an unverified process-global dependency path.
- Focused validation passed: `go test ./pkg/extproc -run 'TestRouterVectorStoreRuntime|TestScheduleResponseMemoryStore|TestResolveVectorStoreRetrievalParams|TestVectorStoreRAG'`.

2026-05-25 ASR006 / TD031-TD034 API knowledge-base config mutation boundary:

- Current-source recheck found a narrow API-side mutation leak: managed knowledge-base create/update/delete used `persistConfigAndSync`, which could call `config.Replace(newCfg)` when `runtimeConfig` was absent even if the API server already carried a `routerruntime.Registry`.
- `src/semantic-router/pkg/apiserver/runtime_config.go` now owns `publishConfigMutation`, so config mutation paths update server-local state, prefer `liveRuntimeConfig` when present, update `routerruntime.Registry` when present, and reserve `config.Replace` for legacy nil-registry callers.
- `src/semantic-router/pkg/apiserver/route_taxonomy_runtime_config_test.go` now covers managed KB mutation with a runtime registry and a conflicting process-wide config, proving the registry and server current config update while `config.Get()` remains unchanged.
- TD031 and TD034 remain open for full compatibility-global retirement, production shared-store policy, server-owned dashboard session/product-state decisions, shared selector-store parity, and durable deployed-config projections.
- Focused validation passed: `go test ./pkg/apiserver -run 'TestHandleKnowledgeBaseMutationWithRuntimeRegistryDoesNotReplaceGlobalConfig|TestHandleKnowledgeBaseLifecycle|TestBuildConfigUpdater|TestResolveAPIServerConfig' -count=1`.

2026-05-25 ASR006 / TD031-TD034 API classification-service runtime boundary:

- Current-source recheck found the same stale-global pattern in API classification service resolution: `resolveClassificationService` and the live service resolver could still adopt `services.GetGlobalClassificationService()` when a runtime registry existed but had not published a classification service yet.
- `src/semantic-router/pkg/apiserver/server.go` now treats a present runtime registry as authoritative for classification service lookup. Registry-backed startup returns only `runtimeRegistry.ClassificationService()`, while the legacy nil-registry API path keeps the retrying global-service fallback.
- `src/semantic-router/pkg/apiserver/server_memory_test.go` covers both sides of the boundary: no global fallback before runtime service publication when a registry exists, runtime service after publication, and preserved legacy nil-registry global fallback.
- TD031 and TD034 remain open for full compatibility-global retirement, broad bootstrap lifecycle ownership, production shared-store policy, server-owned dashboard state decisions, shared selector-store parity, and durable deployed-config projections.
- Focused validation passed: `go test ./pkg/apiserver -run 'TestResolveClassificationService|TestResolveAPIServerConfig|TestResolveMemoryStore|TestBuildConfigUpdater' -count=1`.

2026-05-24 ASR009 / TD033 native capability slice:

- Current-source recheck confirmed TD033 remains open, but the live router-side gap was narrower than a full binding rewrite: higher layers still inferred unified/LoRA classifier support from build tags and backend-compatible stubs.
- `src/semantic-router/pkg/classification/native_capabilities.go` now defines the router-owned native backend capability contract.
- `src/semantic-router/pkg/classification/unified_classifier_cgo_candle.go`, `unified_classifier_cgo_onnx.go`, and `unified_classifier_stub.go` now publish the current build's backend capabilities.
- `src/semantic-router/pkg/classification/unified_classifier.go` now rejects unsupported unified and LoRA batch classification before calling backend FFI stubs, and exposes the current native backend capabilities in classifier stats.
- TD033 remains open for binding lifecycle/reset ownership, ONNX feature parity, backend docs, and release artifact version consistency.
- Validation passed: `go test ./pkg/classification -run 'TestCurrentNativeBackendCapabilitiesShape|TestUnifiedClassifierRejectsUnsupportedNativeCapabilities|TestUnifiedClassifierStatsIncludeNativeCapabilities|TestUnifiedClassifier_GetStats|TestUnifiedClassifier_ClassifyBatch' -v`.

2026-05-24 ASR009 / TD033 native MLP build-hygiene slice:

- Current-source recheck found repeated native Rust build warnings in the standard binding build from unsupported `feature = "metal"` cfg branches in `mlp.rs`.
- The MLP FFI path now uses one `device_from_type` helper for CPU/CUDA selection and keeps `device_type: 2` as the existing CPU fallback because this crate does not currently expose a working Metal feature.
- A direct attempt to add a Metal feature was rejected during validation because it pulled Candle Metal dependency resolution into the current lockfile and conflicted on `once_cell`; the resulting code keeps the current runtime behavior while making the unsupported feature state explicit.
- TD033 remains open for real Metal/ONNX parity decisions, binding lifecycle/reset ownership, backend docs, and release artifact version consistency.
- Validation passed: `cargo fmt --check`; `cargo check --release --no-default-features`; `make test-binding-minimal`.
- Feature validation attempted but did not pass locally: `make test-binding-lora` failed because the local environment lacks the LoRA/PII/Gemma model assets expected by those tests.

2026-05-25 ASR009 / TD033 native generative Rust warning-hygiene slice:

- Current validation output still emitted warning-only debt from the generative Candle binding: unused imports in `candle-binding/src/ffi/generative_classifier.rs` and deprecated rand API usage in `candle-binding/src/model_architectures/generative/qwen3_guard.rs`.
- The C ABI surface and sampling behavior were kept unchanged: the unused imports were removed, top-p sampling now uses `rand::rng().random()` through the current `rand::Rng` API, and generative raw-pointer exports that dereference caller-owned memory are now explicit `unsafe extern "C"` functions with pointer ownership documented in their `# Safety` sections.
- The touched legacy hotspots were also split instead of waived: `generative_guard.rs` owns Qwen3Guard FFI exports, while `qwen3_guard_generation.rs`, `qwen3_guard_loading.rs`, and `qwen3_guard_sampling.rs` own prefix-cache generation, weight loading, and sampling helpers. `generative_classifier.rs` is now 579 lines and `qwen3_guard.rs` is 466 lines, with no changed-file structure-rule errors.
- Direct default `cargo check --manifest-path candle-binding/Cargo.toml --lib` is not the right local verification path in this environment because it enters the CUDA default-feature build and fails before crate checking when `nvcc` is absent. The repository CPU path is `--no-default-features` / `make rust-ci`.
- Validation passed: `cargo fmt --manifest-path candle-binding/Cargo.toml`; changed-file Rust lint through `tools/agent/scripts/agent_gate.py run-rust-lint`; `cargo check --release --no-default-features --lib` from `candle-binding`; `make rust-ci`; `make test-binding-minimal`; `make agent-ci-gate CHANGED_FILES="candle-binding/src/ffi/generative_classifier.rs,candle-binding/src/ffi/generative_guard.rs,candle-binding/src/ffi/mod.rs,candle-binding/src/model_architectures/generative/qwen3_guard.rs,candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_generation.rs,candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_loading.rs,candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_sampling.rs,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`.
- Feature validation attempted but did not pass locally: `make test-binding-lora` failed in existing restricted-model paths, including missing `../models/mom-pii-classifier` / LoRA classifier assets and Gemma fallback dimension expectations. The native smoke path above remains green.

2026-05-24 ASR009 / TD033 release version contract slice:

- Current-source recheck found a release contract drift: `.github/workflows/release.yml` validates both `src/vllm-sr/pyproject.toml` and `candle-binding/Cargo.toml` against `v*` tags, but `src/vllm-sr/scripts/release.sh` only updated the Python package version.
- `src/vllm-sr/scripts/release.sh` now updates the vLLM-SR Python package and the `candle-semantic-router` Cargo manifest/lockfile together for both the release commit and next development bump.
- `candle-binding/Cargo.toml` and `candle-binding/Cargo.lock` are now aligned with the current `src/vllm-sr/pyproject.toml` version `0.3.0`, so a future `v0.3.0` validation tag no longer fails solely because the crate source stayed at `0.2.0`.
- TD033 remains open for binding lifecycle/reset ownership, ONNX feature parity, backend docs, and broader artifact ownership across Docker, Helm, PyPI, crate, operator, dashboard, and simulator workflows.
- Validation passed: `bash -n src/vllm-sr/scripts/release.sh`; Python version-contract parse check; `cargo metadata --no-deps --format-version 1` for `candle-binding`; `make vllm-sr-test`; `make test-binding-minimal`.
- Feature integration and local smoke remain blocked by environment because Docker daemon is unavailable.

2026-05-24 ASR009 / TD033 shared release-contract checker slice:

- Current-source recheck found that release version validation was still split
  across workflow-local grep/sed snippets and the local release helper. Helm
  chart release packaging and Docker release-note image coverage were described
  in workflows, but they were not covered by the same local checker used before
  creating a tag.
- `tools/release/check_version_contract.py` now validates the vLLM-SR Python
  version, Candle Cargo manifest version, Candle lockfile package version, Helm
  release packaging override semantics, and Docker release image coverage in the
  unified release notes from one repo-level parser.
- `make release-check`, `src/vllm-sr/scripts/release.sh`, and
  `.github/workflows/release.yml` now reuse that checker, so local release prep
  and tag validation fail for the same contract drift instead of maintaining
  separate version parsers.
- TD033 remains open for native lifecycle/reset ownership, ONNX feature parity,
  backend docs, upgrade/rollback fixture validation, and broader operator,
  dashboard, and simulator artifact ownership.
- Validation passed: `python3 tools/release/check_version_contract.py`;
  `python3 tools/release/check_version_contract.py --version 0.3.0`;
  `make release-check RELEASE_VERSION=0.3.0`;
  `bash -n src/vllm-sr/scripts/release.sh`.

2026-05-24 ASR009 / TD033 Operator release-image coverage slice:

- Current-source recheck found a narrower release artifact gap: `operator-ci.yml`
  publishes `semantic-router/operator` and `semantic-router/operator-bundle`
  on `v*` tags, but the shared release checker only parsed
  `docker-release.yml` and the unified GitHub Release body did not list the
  Operator images.
- `tools/release/check_version_contract.py` now parses tagged Operator and
  bundle image names from `operator-ci.yml` and requires them to appear in the
  unified release notes alongside the router, dashboard, ROCm, and llm-katan
  images.
- `.github/workflows/release.yml` now lists `operator` and `operator-bundle`
  pull commands and tag rows in the release body.
- TD033 remains open for native lifecycle/reset ownership, ONNX feature parity,
  backend docs, upgrade/rollback fixture validation, and simulator release
  ownership, but the Operator image publication path is now covered by the
  shared release contract.
- Validation passed: `python3 tools/release/check_version_contract.py`;
  `python3 tools/release/check_version_contract.py --version 0.3.0`;
  `python3 -m py_compile tools/release/check_version_contract.py`;
  `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml tools/release/check_version_contract.py`;
  `make release-check RELEASE_VERSION=0.3.0`.

2026-05-24 ASR009 / TD033 upgrade-runbook release-image slice:

- Current-source recheck found that the shared release checker knew the full
  versioned release image set, but `website/docs/installation/upgrade-rollback.md`
  only documented a partial Docker upgrade set.
- `tools/release/check_version_contract.py` now validates that the upgrade and
  rollback runbook mentions every versioned release image parsed from the Docker
  and Operator release workflows.
- `website/docs/installation/upgrade-rollback.md` now lists the full image set:
  `dashboard`, `extproc`, `extproc-rocm`, `llm-katan`, `operator`,
  `operator-bundle`, `vllm-sr`, and `vllm-sr-rocm`.
- TD033 remains open for native lifecycle/reset ownership, ONNX feature parity,
  backend docs, real upgrade/rollback fixture execution, and simulator package
  ownership, but the release image runbook can no longer silently drift from CI.
- Validation passed: `python3 tools/release/check_version_contract.py`;
  `python3 tools/release/check_version_contract.py --version 0.3.0`;
  `python3 -m py_compile tools/release/check_version_contract.py`;
  `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml tools/release/check_version_contract.py`;
  `make release-check RELEASE_VERSION=0.3.0`.

2026-05-25 ASR009 / TD033 simulator release-workflow ownership slice:

- Current-source recheck found the simulator release debt narrower than the
  debt text: `.github/workflows/pypi-publish-vllm-sr-sim.yml` already owns an
  independent `vllm-sr-sim-v*` tag stream, validates package version against
  tag version, and emits install snippets, but the shared release checker and
  upgrade/rollback runbook did not make that contract enforceable.
- `tools/release/check_version_contract.py` now validates simulator workflow
  tag trigger/extraction, package/tag version guard, PyPI publish command, and
  install snippet, plus unified release-note and runbook coverage for the
  independent `vllm-sr-sim` package stream.
- `website/docs/installation/upgrade-rollback.md` now documents the pinned
  `vllm-sr-sim` package upgrade command and separate
  `vllm-sr-sim-v<version>` promotion flow.
- TD033 remains open for native lifecycle/reset ownership, ONNX feature parity,
  backend docs, and real upgrade/rollback fixture execution, but simulator PyPI
  release ownership can no longer silently drift from local/tag validation.
- Validation passed: `python3 tools/release/check_version_contract.py`;
  `python3 tools/release/check_version_contract.py --version 0.3.0`;
  `python3 -m py_compile tools/release/check_version_contract.py`;
  `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml tools/release/check_version_contract.py`;
  `make release-check RELEASE_VERSION=0.3.0`.

2026-05-25 ASR009 / TD033 native backend documentation slice:

- Current-source recheck found the backend-docs portion of TD033 still current:
  `NativeBackendCapabilities` exposes Candle, ONNX, and stub capability
  boundaries, but the installation docs did not tell operators which build
  selects which backend, which features fail early, or that no backend currently
  advertises explicit reset.
- `website/docs/installation/native-backends.md` now documents backend
  selection, the Candle/ONNX/stub capability table, early unsupported-feature
  failure behavior, and lifecycle expectations for process-owned native state.
- `website/sidebars.ts` links the native backend guide from the Installation
  section so the parity guidance is discoverable with configuration and upgrade
  docs.
- TD033 remains open for native lifecycle/reset ownership, deeper ONNX feature
  parity, and real upgrade/rollback fixture execution.
- Validation passed: `make docs-lint`; `cd website && npm run build:en`;
 `make agent-ci-gate CHANGED_FILES="..."`; `make agent-scorecard`;
  `git diff --check`.

2026-05-25 ASR009 / TD033 upgrade-runbook fixture-check slice:

- Current-source recheck found the release checker still only required the
  upgrade/rollback runbook to mention image names, not copyable full image refs
  or pinned command examples. That left a release-doc drift path where
  `make release-check` could pass even if the runbook lost the exact Helm,
  Docker, Python, or rollback fixture commands operators need.
- `tools/release/check_version_contract.py` now requires every release image
  parsed from Docker and Operator workflows to appear as a full
  `ghcr.io/vllm-project/semantic-router/<image>:v<version>` reference in the
  runbook.
- The same checker now validates pinned upgrade and rollback markers for Helm
  chart lookup/upgrade, safe `--reset-then-reuse-values` semantics, Helm
  rollback, Kubernetes rollout undo, Docker digest lookup and release-image
  pull target, Helm release Make target, Helm values image tag pin, Python CLI
  upgrade pin, and the independent `vllm-sr-sim` upgrade pin.
- TD033 remains open for native lifecycle/reset ownership, deeper ONNX feature
  parity, and live environment upgrade/rollback execution, but static runbook
  fixture drift is now part of the shared local/CI release contract.
- Validation passed: `python3 tools/release/check_version_contract.py`;
  `python3 tools/release/check_version_contract.py --version 0.3.0`;
  `python3 -m py_compile tools/release/check_version_contract.py`;
  `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml
  tools/release/check_version_contract.py`; `make release-check
  RELEASE_VERSION=0.3.0`.

2026-05-25 ASR009 / TD033 Candle crate workflow ownership slice:

- Current-source recheck found `.github/workflows/publish-crate.yml` already
  owned the Candle crate tag trigger, tag-version guard, CPU-only API smoke,
  `cargo check`, release build, publish dry-run, crates.io publish, and GitHub
  Release native library artifact attachment. The remaining gap was that the
  shared release checker did not validate those native artifact markers.
- `tools/release/check_version_contract.py` now includes the Candle crate
  workflow in the local/CI release contract and validates the crate publish
  guardrails plus `.a` and `.so` release artifact attachment.
- `tools/release/release_contract_markers.py` now owns the static marker sets
  so the checker remains a 352-line orchestrator instead of becoming another
  release-script hotspot.
- The checker also validates that unified release notes keep the Rust crate
  install snippet tied to `${{ needs.validate.outputs.version }}`.
- TD033 remains open for native lifecycle/reset ownership, deeper ONNX feature
  parity, and live environment upgrade/rollback execution, but native crate
  publish and artifact ownership can no longer drift outside
  `make release-check` and the tag validation workflow.
- Validation passed: `python3 tools/release/check_version_contract.py`;
  `python3 tools/release/check_version_contract.py --version 0.3.0`;
  `python3 -m py_compile tools/release/check_version_contract.py
  tools/release/release_contract_markers.py`;
  `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml
  tools/release/check_version_contract.py
  tools/release/release_contract_markers.py`; `make release-check
  RELEASE_VERSION=0.3.0`; `make agent-ci-gate CHANGED_FILES="..."`.

2026-05-24 ASR006 / TD034 decision-scoped selector persistence config slice:

- Current-source recheck found a live config/runtime mismatch rather than only
  shared-store design debt: RLDriven and GMTRouter docs exposed persistence and
  tuning fields, and both selectors implemented local persistence seams, but
  `buildModelSelectionConfig` only forwarded decision-scoped Elo and RouterDC
  config into the selector factory.
- `src/semantic-router/pkg/config/selection_config.go`,
  `src/vllm-sr/cli/algorithms.py`, and
  `dashboard/frontend/src/lib/dslSchemas.ts` now represent the documented
  RLDriven and GMTRouter algorithm fields used by the router config contract.
- `src/semantic-router/pkg/extproc/router_selection.go` now forwards
  decision-scoped RLDriven and GMTRouter config, including `storage_path`, into
  `selection.ModelSelectionConfig` while preserving selector defaults for
  omitted fields.
- `src/semantic-router/pkg/extproc/router_selection_config_test.go` covers the
  runtime mapping and default-preservation behavior, and `config/config.yaml`
  plus the selection fragments exercise the expanded public fields.
- Validation passed: `go test ./pkg/extproc -run 'TestBuildModelSelectionConfigUsesDecisionScopedLearningState|TestBuildModelSelectionConfigPreservesLearningDefaultsWhenUnset'`;
  `go test ./pkg/config -run 'TestReferenceConfig|TestConfigFragments'`.
- Validation passed: `python3 -m pytest src/vllm-sr/tests/test_algorithm_config.py -q`;
  `.venv-agent/bin/python -m ruff check --config tools/linter/python/.ruff.toml src/vllm-sr/cli/algorithms.py src/vllm-sr/cli/validator.py src/vllm-sr/cli/commands/runtime_support.py src/vllm-sr/cli/commands/runtime.py src/vllm-sr/tests/test_algorithm_config.py`;
  `npm run type-check`; `npm run lint`.
- Changed-set validation passed after formatting and test-structure cleanup:
  `make agent-ci-gate CHANGED_FILES="..."`, including `make agent-validate`,
  `make vllm-sr-test`, `make test-semantic-router`, and
  `make dashboard-check`.
- Scorecard and whitespace validation passed: `make agent-scorecard`;
  `git diff --check`.
- Feature integration remains blocked by environment, not by this code path:
  `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`,
  and `make vllm-sr-test-integration` stops at `vllm-sr-build` with the same
  Docker daemon connection failure.

2026-05-24 repo harness shellcheck virtualenv isolation slice:

- Current-source validation found the release changed-set gate could fail for a false-positive harness reason: `make agent-ci-gate` invoked `make shellcheck`, and `make shellcheck` scanned a local, untracked `.venv-fresh/.../tqdm/completion.sh` file.
- `tools/make/linter.mk` now excludes `.venv-*` directories in addition to the named repo virtualenv paths, so local agent/Python scratch environments no longer affect shell-script linting.
- This is a repo-harness reliability fix only; it does not change product runtime behavior or public APIs.
- Validation passed: `make shellcheck`; `make agent-ci-gate CHANGED_FILES="tools/make/linter.mk,src/vllm-sr/scripts/release.sh,candle-binding/Cargo.toml,candle-binding/Cargo.lock,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md"`.

2026-05-24 ASR005 / TD039 OpenClaw listener adapter slice:

- Current-source recheck found another narrow Dashboard backend config contract leak: `dashboard/backend/handlers/openclaw.go` parsed router listener YAML inline to discover the local model gateway URL.
- `dashboard/backend/routercontract/listeners.go` now owns the dashboard-facing listener endpoint adapter, including top-level and legacy `api_server.listeners` shapes.
- `dashboard/backend/handlers/openclaw.go` now consumes `routercontract.ReadFirstListenerEndpoint` and keeps only OpenClaw-specific URL precedence and host normalization logic.
- `dashboard/backend/routercontract/listeners_test.go` covers top-level listeners, legacy `api_server.listeners`, invalid ports, and parse errors; existing OpenClaw gateway tests cover env precedence over router config discovery.
- TD039 remains open for Dashboard config/deploy/setup handlers, Operator canonical translation, CLI contract/runtime ownership, and remaining runtime global publication.
- Validation passed: `go test ./handlers ./routercontract -run 'TestResolveOpenClawModelBaseURL|TestDefaultOpenClawModelBaseURL|TestReadFirstListenerEndpoint|TestOpenClawModelGatewayContainerName'` from `dashboard/backend`.
- Changed-set validation passed: `make agent-ci-gate CHANGED_FILES="..."`, including `make agent-validate`, `make vllm-sr-test`, `make vllm-sr-sim-test`, `make test-semantic-router`, and `make dashboard-check`.
- Feature integration and local smoke remain blocked by environment because `docker info` still cannot connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-25 ASR005 / TD039 Dashboard security-policy apply validation slice:

- Current-source recheck confirmed the old `applySecurityFragment` TODO was still live: generated security fragments were deep-merged and written before the merged router config was validated.
- `dashboard/backend/handlers/security_policy_config_validation.go` now owns the apply-time validation seam for security-policy generated config, using canonical endpoint rules, the router YAML parser, and parsed endpoint address checks before any atomic write or runtime apply.
- `dashboard/backend/handlers/security_policy_apply.go` now owns security-policy config write/hot-reload application, and `dashboard/backend/handlers/security_policy.go` dropped from 409 to 354 lines so the production handler no longer trips the structure warn threshold.
- The apply path now rejects invalid merged config before writing, so the security-policy endpoint cannot persist a config that the full Dashboard config editor would reject.
- `dashboard/backend/handlers/security_policy_config_validation_test.go` covers both a valid generated-policy merge and the invalid-endpoint guard that preserves the on-disk file.
- TD039 remains open for broader Dashboard config/deploy/setup ownership, Operator canonical translation, CLI contract/runtime ownership, and remaining runtime global publication.
- Validation passed: `go test ./handlers -run 'TestValidateMergedSecurityConfigAcceptsPolicyMerge|TestApplySecurityFragmentRejectsInvalidMergedConfigBeforeWrite'` from `dashboard/backend`.
- Validation passed: `make dashboard-check`.

2026-05-24 ASR008 / TD030 route-shell registry slice:

- Current-source recheck found the older TD030 text partially stale for `App.tsx`: the provider and iframe-guard shell is already narrow. The live route-shell hotspot is now `dashboard/frontend/src/app/AppRouter.tsx`, where protected route registration and repeated `AppShellLayout` wrapping still lived inline.
- `dashboard/frontend/src/app/AuthenticatedAppRoutes.tsx` now owns authenticated route registration, repeated shell wrapping, config/knowledge-base route adapters, protected redirects, and the ML Setup access redirect.
- `dashboard/frontend/src/app/AppRouter.tsx` now keeps setup loading/error handling, public routes, auth gate nesting, and top-level route state only; the file dropped from 305 to 75 lines in this slice.
- TD030 remains open for `dslStore.ts`, large page/container hotspots, broader config/interaction surface extraction, and much wider frontend coverage beyond the route manifest.
- Validation passed: `npm run type-check`; `npm run lint`; `make dashboard-check`.
- A stricter exploratory `npm run lint -- --max-warnings=0` failed on 58 existing warnings outside this slice before later warning cleanup, so the warning backlog remained active frontend debt rather than evidence of this route-shell change failing.

2026-05-24 ASR006 / TD034 Response API Redis durability and test-isolation slice:

- Current-source recheck found the Response API portion of TD034 stale: canonical defaults, runtime config comments, CLI local defaults, and the state taxonomy now treat Redis as the default durable Response API store, with `memory` documented as local-development only.
- `e2e/testcases/response_api_restart_recovery.go` already registers `response-api-restart-recovery` to verify Redis-backed Response API history survives a semantic-router pod restart in the Redis profile.
- `src/semantic-router/pkg/responsestore/redis_store_integration_test.go` now has an `integration` build tag, so default `go test ./pkg/responsestore` no longer requires a local Redis instance. Redis-backed integration coverage remains available through `go test -tags=integration ./pkg/responsestore/...` when Redis is running.
- `website/docs/troubleshooting/common-errors.md` now documents Redis Response API connection failures and the current `global.services.response_api.store_backend: redis` contract instead of the stale "Redis store not yet implemented" guidance.
- TD034 remains open for model-selection learning-state shared-store parity, dashboard/browser-local state, derived config projections, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `go test ./pkg/responsestore`; `cd e2e && go test ./testcases`; `cd src/vllm-sr && python3 -m pytest tests/test_runtime_support.py tests/test_storage_backends.py -q`.

2026-05-24 ASR006 / TD034 RL-driven learning-state slice:

- Current-source recheck found the model-selection learning-state debt narrower than the original TD034 text: Elo already has file-backed ratings when `storage_path` is configured, and `rl_driven` had a storage seam, but it only saved global/category preferences while user personalization and session context were lost on restart.
- `src/semantic-router/pkg/selection/rl_driven.go` now encodes RL-driven user preferences and session context into the existing storage format with reserved, base64-encoded category keys, preserving compatibility with older global/category rating files.
- `src/semantic-router/pkg/selection/storage.go` now makes `FileEloStorage.Close()` safe for storage instances that never called `StartAutoSave`, preventing exported file-backed storage users from hanging on shutdown.
- `src/semantic-router/pkg/selection/rl_driven_test.go` now covers `storage_path` restart loading for user-specific winner/loser preferences and per-session model stats.
- Current-source recheck also found the model-research campaign entry stale: `dashboard/backend/modelresearch/**` and `dashboard/backend/handlers/modelresearch.go` are no longer present in the repo, so TD034 should not keep treating that as an active runtime surface.
- TD034 remains open for dashboard/browser-local state, derived config projections, GMTRouter/Elo shared-store parity beyond local files, compatibility globals, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `go test ./pkg/selection -run 'TestFileEloStorage_CloseWithoutAutoSaveDoesNotBlock|TestFileEloStorage_SaveAndLoad|TestRLDrivenSelector_StoragePersistsUserAndSessionState'`; `go test ./pkg/selection`; `make agent-ci-gate CHANGED_FILES="..."`.

2026-05-24 ASR006 / TD034 startup readiness shared-state slice:

- Current-source recheck found a narrower router runtime status gap: `/startup-status` already loaded Redis-backed startup state when configured, but `/ready` still read only the file-backed status path and could false-negative in a Redis/shared-status deployment.
- `src/semantic-router/pkg/apiserver/server.go` now makes `/ready` use the same startup-state resolver as `/startup-status`.
- `src/semantic-router/pkg/apiserver/route_startup_status.go` has a private loader seam so the shared resolver path can be covered without requiring a live Redis service in unit tests.
- `src/semantic-router/pkg/apiserver/server_ready_test.go` covers readiness through that shared resolver, and `src/semantic-router/cmd/runtime_bootstrap.go` now points operators at the current `startup_status.store_backend` key in its file-backend warning.
- TD034 remains open for browser-local dashboard state, derived config projections, shared-store parity for local-file selector state, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `go test ./pkg/apiserver -run 'TestHandleReady|TestHandleStartupStatus' -count=1`.

2026-05-24 ASR006 / TD034 playground queue localStorage bounds slice:

- Current-source recheck found `useConversationStorage` already documents browser-local chat history and caps saved conversations, but `usePlaygroundQueue` still restored and persisted queued playground tasks from localStorage without shape validation or retention bounds.
- `dashboard/frontend/src/hooks/playgroundQueueStorage.ts` now owns pure queue normalization and pruning: invalid restored entries are dropped, per-conversation tasks are capped, and the number of persisted conversations is capped by newest queued task timestamp.
- `dashboard/frontend/src/hooks/usePlaygroundQueue.ts` now uses that helper on load and before every persistence write.
- `dashboard/frontend/src/hooks/playgroundQueueStorage.test.ts` covers invalid restore filtering, per-conversation task caps, and conversation-count caps.
- TD034 remains open for server-owned dashboard chat/session decisions, auth-token storage hardening, derived config projections, shared selector stores, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `npm run test:unit`; `npm run type-check`; `npm run lint`.

2026-05-24 ASR006 / TD034 conversation localStorage restore bounds slice:

- Current-source recheck found a narrower live `useConversationStorage` gap after the queue slice: saved conversations were capped on write, but restored browser-local records were accepted without shape validation or read-time bounds.
- `dashboard/frontend/src/hooks/conversationStorage.ts` now owns pure conversation normalization and pruning: invalid restored entries are dropped, duplicate IDs are collapsed after trimming, and restored conversations are capped by newest update timestamp.
- `dashboard/frontend/src/hooks/useConversationStorage.ts` now uses the helper on load and before persistence writes, removing the previous path where overlarge or malformed localStorage data could hydrate directly into React state.
- `dashboard/frontend/src/hooks/conversationStorage.test.ts` covers malformed restore filtering, max conversation caps, duplicate ID normalization, typed payload validation, and save-time pruning.
- TD034 remains open for server-owned dashboard chat/session decisions, auth-token storage hardening, derived config projections, shared selector stores, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `npm run test:unit -- conversationStorage playgroundQueueStorage routeManifest dashboardPageOverview`; `npm run type-check`; `npm run lint`.
- Local smoke was still blocked by environment, not by this code path: `docker info` cannot connect to `/Users/bitliu/.docker/run/docker.sock`.

2026-05-24 ASR006 / TD034 auth token browser-storage hardening slice:

- Current-source recheck found `dashboard/frontend/src/utils/authFetch.ts` still accepted any `localStorage` value as an auth token and then mirrored it into protected fetch headers, JS cookies, and query-token transports for embedded resources.
- `authFetch.ts` now normalizes token values at the storage boundary, rejects empty, oversized, whitespace, semicolon, and control-character tokens, clears malformed persisted values before reuse, and reuses one normalized token per fetch wrapper call.
- `dashboard/frontend/src/contexts/AuthContext.tsx` now rejects invalid login tokens before marking the session authenticated and keeps React session state aligned with the normalized stored token.
- `dashboard/frontend/src/utils/authFetch.test.ts` covers token normalization, cookie/localStorage writes, and malformed stored-token cleanup.
- TD034 remains open for server-owned session semantics, server-side chat/session product decisions, durable config projections, shared selector stores, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `npm run test:unit -- authFetch conversationStorage playgroundQueueStorage routeManifest dashboardPageOverview`; `npm run type-check`; `npm run lint`; `make dashboard-check`.

2026-05-24 ASR006 / TD034 server auth session cookie slice:

- Current-source recheck found Dashboard login and bootstrap auth handlers only returned the JWT response body while the frontend mirrored the token into browser-owned storage and cookie state. Logout cleared frontend state only.
- `dashboard/backend/auth/http_handlers.go` now sets an HttpOnly `vsr_session` cookie on login/bootstrap, and `dashboard/backend/auth/session_cookie.go` exposes a compatible `POST /api/auth/logout` handler that clears the server cookie without requiring a valid current token.
- `dashboard/backend/router/auth_routes.go` proxies `/api/auth/logout`, and `dashboard/backend/auth/middleware.go` leaves logout unauthenticated so expired or invalid browser sessions can still be cleared.
- `dashboard/frontend/src/contexts/AuthContext.tsx` now asks the server to clear the session cookie on user logout while still clearing local token state immediately.
- `dashboard/backend/auth/handlers_test.go` covers HttpOnly/SameSite/Secure login cookie issuance and logout cookie clearing; existing frontend token normalization tests remain in `authFetch.test.ts`.
- TD034 remains open for cookie-first reload semantics, production session-store policy, server-side chat/session product decisions, durable config projections, shared selector stores, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `go test ./auth ./router` from `dashboard/backend`; `npm run test:unit -- authFetch conversationStorage playgroundQueueStorage routeManifest dashboardPageOverview`; `npm run type-check`; `npm run lint`; `make dashboard-check`.

2026-05-24 ASR006 / TD034 cookie-backed auth reload slice:

- Current-source recheck found the previous server-cookie slice still incomplete on browser reload: the backend could authenticate a `vsr_session` cookie, but `AuthProvider` only refreshed the current user when a readable local token existed.
- `dashboard/frontend/src/contexts/authSession.ts` now owns the testable session-refresh contract: `/api/auth/me` is requested with same-origin credentials, `401` marks local token state stale, and a returned user counts as an authenticated session even when no local token is readable.
- `dashboard/frontend/src/contexts/AuthContext.tsx` now probes `/api/auth/me` on provider startup, so an HttpOnly session cookie can restore the user without localStorage token material.
- `dashboard/backend/auth/handlers_test.go` now covers cookie-only authentication through `AuthenticateRequest` in addition to login cookie issuance and logout clearing.
- TD034 remains open for production session-store policy, server-side chat/session product decisions, durable config projections, shared selector stores, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `npm run test:unit -- authSession authFetch conversationStorage playgroundQueueStorage routeManifest dashboardPageOverview`; `npm run type-check`; `npm run lint`; `go test ./auth ./router` from `dashboard/backend`.

2026-05-25 ASR006 / TD034 backend auth session revocation slice:

- Current-source recheck found the frontend already rejected malformed or oversized local token material, but `dashboard/backend/auth/middleware.go` still passed any non-empty bearer, cookie, or query token into JWT parsing.
- `normalizeAccessToken` now owns the backend token boundary for all three compatible transports, trimming valid token material while rejecting empty, oversized, whitespace, semicolon, and control-character values before `Service.ParseToken`.
- The same recheck found logout still only cleared browser state and cookie state while newly issued JWTs stayed valid until expiry. New tokens now carry a persisted session id, authentication rejects revoked or missing session ids for new tokens, and logout revokes the current session before clearing the cookie. Tokens without a session id remain on the legacy compatibility path.
- `dashboard/backend/auth/session_cookie.go` now sets explicit `Expires` timestamps on both issue and clear responses, while retaining the existing MaxAge, HttpOnly, SameSite, and proxy-aware Secure behavior.
- `dashboard/backend/auth/session_store.go` owns the focused SQLite-backed session persistence seam instead of growing `store.go`; it also prunes expired or revoked sessions older than the retention window on auth-store open, with expiry/revocation indexes to keep cleanup bounded.
- `dashboard/backend/auth/middleware_test.go` covers backend token normalization plus malformed bearer/query fallback behavior; `dashboard/backend/auth/session_cookie_test.go` verifies cookie-only authentication, cookie issue/clear expiry semantics, and logout revocation without growing the broad handler test file; `dashboard/backend/auth/session_store_test.go` covers the session lifecycle plus startup pruning behavior.
- TD034 remains open for production session-store policy, server-side chat/session product decisions, durable config projections, shared selector stores, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `go test -count=1 ./auth ./router` from `dashboard/backend`; `make dashboard-check`.

2026-05-25 ASR006 / TD006-TD034 Dashboard auth store ownership split:

- Current-source recheck after the session revocation slice found `dashboard/backend/auth/store.go` still exceeded the structure-rule warning threshold at 525 lines because user CRUD, role normalization, role/permission queries, audit logging, store initialization, and session pruning still shared one file.
- `dashboard/backend/auth/permission_store.go` now owns role normalization, default role-permission sync, effective-permission lookup, and permission-count/list helpers.
- `dashboard/backend/auth/audit_store.go` now owns the `AuditLog` DTO plus audit insert/list query behavior.
- `dashboard/backend/auth/store.go` now stays focused on SQLite store construction, lifecycle, and user CRUD and drops to 245 lines; `dashboard/backend/auth/session_store.go` remains the dedicated session lifecycle/pruning seam. The changed-set structure warning for `store.go` is removed, while `dashboard/backend/auth/http_handlers.go` remains a separate handler hotspot.
- TD034 is unchanged for production session-store policy and server-owned product state, but TD006/maintainability debt is narrowed for the Dashboard auth store surface.
- Validation passed: `go test -count=1 ./auth ./router` from `dashboard/backend`; `make dashboard-check`.

2026-05-25 ASR006 / TD006-TD034 Dashboard auth handler ownership split:

- Current-source recheck after the store split still found one touched Dashboard auth hotspot: `dashboard/backend/auth/http_handlers.go` carried public bootstrap/login/me handlers, admin user CRUD, admin permission/audit/password handlers, and session-user helpers in one 556-line file.
- `dashboard/backend/auth/public_handlers.go` now owns bootstrap, login, and `/me` handlers.
- `dashboard/backend/auth/admin_user_handlers.go` now owns admin user collection/item handlers plus user-update validation.
- `dashboard/backend/auth/admin_meta_handlers.go` now owns admin permission, audit-log, and password handlers.
- `dashboard/backend/auth/session_user.go` now owns session-user cloning and permission helper behavior. The old `http_handlers.go` hotspot is removed; the largest new auth handler file is 290 lines, so the changed set no longer trips auth handler/store structure warnings.
- TD034 remains open for production session-store policy and server-owned product state, but TD006/maintainability debt is narrowed for the Dashboard auth handler surface.
- Validation passed: `go test -count=1 ./auth ./router` from `dashboard/backend`; `make dashboard-check`.

2026-05-25 ASR006 / TD034-ASR009 Dashboard Helm local-state persistence guard:

- Current-source recheck found the Dashboard backend had restart-safe SQLite auth/session/workflow stores when pointed at a persistent path, but the Helm dashboard deployment did not mount a dashboard data PVC or wire `DASHBOARD_AUTH_DB_PATH` / `DASHBOARD_WORKFLOW_DB_PATH` to one. The production values profile enabled the dashboard without making that state restart-safe.
- `deploy/helm/semantic-router/templates/dashboard-pvc.yaml` now creates an optional dashboard-local state PVC, and `dashboard.persistence.*` values control existing claims, storage class, access mode, size, mount path, and annotations.
- `deploy/helm/semantic-router/templates/dashboard-deployment.yaml` now mounts the PVC at `/app/data` when enabled and wires auth/session plus workflow SQLite paths into that mounted directory.
- `deploy/helm/semantic-router/values-prod.yaml` enables dashboard-local persistence. The deployment also has a template-time guard that fails if `dashboard.replicaCount > 1`, because the current SQLite auth/session store is restart-safe local state, not a shared HA session store.
- Dashboard README and Helm values docs now describe the single-replica local-state boundary explicitly instead of implying multi-replica dashboard auth readiness.
- TD034 remains open for a real shared production auth/session store, server-owned dashboard product state, shared selector stores, durable config projections, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `helm dependency build deploy/helm/semantic-router`; `helm template sr deploy/helm/semantic-router --set dashboard.enabled=true --set dashboard.persistence.enabled=true`; `helm template sr deploy/helm/semantic-router --set dashboard.enabled=true --set dashboard.replicaCount=2` fails with the expected shared-store guard.

2026-05-25 ASR006 / TD034-ASR009 Helm selector local-state multi-replica guard:

- Current-source recheck found RL-driven, GMTRouter, and Elo can persist or hold selector learning state in local files or process memory, while the Helm production router profile can run multiple replicas through `replicaCount` or HPA.
- `deploy/helm/semantic-router/templates/deployment.yaml` now rejects multi-replica router renders when active routing decisions use `algorithm.type: elo`, `rl_driven`, or `gmtrouter`, unless `safetyGuards.rejectMultiReplicaLocalSelectorState=false` is set after accepting or externalizing replica-local learning state.
- `deploy/helm/semantic-router/values.yaml`, Helm docs, state taxonomy, TD034, and the user-facing Elo/RL-driven/GMTRouter selection docs now describe the guard and the local-state boundary.
- Scores move for system completeness, performance/availability, memory/compute, release/upgrade, and documentation freshness because Helm now prevents a concrete production misconfiguration instead of only documenting it.
- TD034 remains open for a real shared selector-state store, shared production auth/session store, server-owned dashboard product state, durable config projections, and full Docker-backed smoke/E2E.
- Validation passed: `helm lint deploy/helm/semantic-router`; `helm template sr deploy/helm/semantic-router -f deploy/helm/semantic-router/values-prod.yaml`; negative RL-driven `replicaCount=2` and GMTRouter HPA renders fail with the expected guard; `safetyGuards.rejectMultiReplicaLocalSelectorState=false` allows the explicit unsafe render.

2026-05-25 ASR009 / Helm values schema contract slice:

- Current-source recheck found the Helm chart had template guards for critical local-state constraints, but no `values.schema.json` for public deployment controls, so invalid guard, replica, autoscaling, or dashboard persistence value types could reach template rendering.
- `deploy/helm/semantic-router/values.schema.json` now validates the key public router and dashboard deployment controls while leaving subchart-specific values extensible.
- Helm docs now list the schema as part of the chart contract and explain that schema validation owns type checks while templates own cross-field production safety errors.
- Scores move for API stability and release completeness because Helm now has a machine-checked values contract for the deployment controls involved in the local-state guard.
- Validation passed: `helm lint deploy/helm/semantic-router`; `helm template sr deploy/helm/semantic-router -f deploy/helm/semantic-router/values-prod.yaml`; invalid `safetyGuards.rejectMultiReplicaLocalSelectorState=maybe` and `replicaCount=oops` renders fail schema validation before template rendering.

2026-05-25 ASR009 / Helm chart harness routing slice:

- Current-source recheck found Helm chart and values-schema changes were still classified as documentation-only by `make agent-report`, even though they can change Kubernetes deployment behavior.
- `tools/agent/task-matrix.yaml` now has a non-doc `helm-chart` rule for `deploy/helm/**`, with `make helm-lint` as the fast gate and `make helm-ci-validate HELM_REPO_UPDATE=false` as the feature gate.
- `tools/agent/repo-manifest.yaml`, `tools/agent/skill-registry.yaml`, and `docs/agent/change-surfaces.md` now map `deploy/helm/**` into the Kubernetes deployment/profile surfaces.
- `tools/make/helm.mk` now keeps default CI behavior as a repository refresh, while local agent validation can set `HELM_REPO_UPDATE=false` to validate the locked chart dependencies without being blocked by transient chart repository timeouts. The Helm validation render output now defaults to `/tmp/semantic-router-helm/default-template.yaml` so a successful feature gate does not leave repo-local YAML that later global lint scans treat as source.
- Scores move for repo harness friendliness because the same Helm safety work now routes to Helm validation instead of relying on a human to notice manual commands.
- Validation passed: `make agent-report ENV=cpu CHANGED_FILES="..."` now reports matched rules `repo-docs`, `agent_text`, `agent_exec`, `helm-chart`, and `make-and-ci`, fast tests `make agent-validate` plus `make helm-lint`, feature test `make helm-ci-validate HELM_REPO_UPDATE=false`, required local smoke, affected local E2E profile `kubernetes`, and CI E2E mode `all`. `make helm-ci-validate HELM_REPO_UPDATE=false HELM_NAMESPACE=test-namespace` passed. The default `make helm-ci-validate HELM_NAMESPACE=test-namespace` was attempted and failed on a Prometheus chart repository timeout before dependency build. `make agent-e2e-affected CHANGED_FILES="..."` reached the Kubernetes profile and failed during Kind cluster discovery because the local Docker/Kind environment is unavailable.

2026-05-25 ASR009 / Helm safety regression gate slice:

- Current-source recheck found the Helm local-state safety work was validated by manual commands and generic Helm rendering, but the exact schema and local-state safety regressions were not encoded as a stable make target.
- `tools/make/helm.mk` now exposes `make helm-safety-validate`, which renders the production profile, verifies schema rejection for invalid selector-guard and replica-count types, verifies multi-replica RL-driven and GMTRouter local-state rejection, and verifies the explicit unsafe opt-out path still renders.
- `tools/agent/task-matrix.yaml` now runs `make helm-safety-validate HELM_REPO_UPDATE=false` as a Helm feature gate, so agents do not have to remember the individual negative/positive Helm template commands.
- Helm docs list the target next to `helm-lint` and `helm-template`.
- Scores move for release completeness and repo harness friendliness because production local-state safety is now mechanically replayable.
- Validation passed: `make helm-safety-validate HELM_REPO_UPDATE=false HELM_NAMESPACE=test-namespace`.

2026-05-25 ASR009 / Helm HPA bounds guard slice:

- Current-source recheck found Helm values schema validated autoscaling field types and scalar ranges, but cross-field `autoscaling.minReplicas > autoscaling.maxReplicas` could still render an invalid HPA manifest.
- `deploy/helm/semantic-router/templates/hpa.yaml` now fails template rendering with a targeted error when min replicas exceed max replicas.
- `make helm-safety-validate` now covers the HPA bounds negative case in the same regression target as the schema and selector-local-state guard checks.
- Helm docs describe invalid HPA replica bounds as a template-owned cross-field production safety rule.
- Scores move for system completeness and availability/stability because invalid scaling configuration is now rejected before Kubernetes resource creation.
- Validation passed: `make helm-safety-validate HELM_REPO_UPDATE=false HELM_NAMESPACE=test-namespace`.

2026-05-24 ASR006 / TD034 GMTRouter local-state hardening slice:

- Current-source recheck found GMTRouter personalization state already had a local `storage_path`, but the save path wrote directly to the target file, did not create nested storage directories, and left corrupted state handling inline with no backup contract.
- `src/semantic-router/pkg/selection/gmtrouter_storage.go` now owns clone, atomic save, load, corruption backup, and normalization helpers for GMTRouter local state, keeping the selector orchestrator focused on routing behavior.
- `src/semantic-router/pkg/selection/gmtrouter.go` now delegates save/load to that helper and snapshots user state before JSON encoding.
- `src/semantic-router/pkg/selection/gmtrouter_test.go` covers directory creation, normalized reload, corrupted-file backup, and the existing persistence flow.
- TD034 remains open for shared selector-store parity beyond local files, production session-store policy, server-side chat/session product decisions, durable config projections, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `go test ./pkg/selection -run 'TestGMTRouterSelector_Persistence|TestGMTRouterStateStorageCreatesDirectoryAndNormalizes|TestGMTRouterStateStorageBacksUpCorruptedFile|TestGMTRouterSelector_PersonalizedSelection|TestRLDrivenSelector_StoragePersistsUserAndSessionState'`; `go test ./pkg/selection`.

2026-05-25 ASR006 / TD034 CLI runtime KB bootstrap-state recovery slice:

- Current-source recheck found a narrow local-runtime state gap not called out in the older TD034 text: the CLI runtime support flow wrote `.vllm-sr/knowledge_bases/.bootstrap-state.yaml` directly and let corrupted YAML abort runtime config resolution.
- `src/vllm-sr/cli/commands/runtime_kb.py` now owns KB bootstrap marker state, treats invalid bootstrap-state YAML as an empty state, and logs a warning so a damaged local marker file no longer blocks runtime config generation.
- Runtime KB marker writes now use a same-directory temporary file and `os.replace`, so local KB bootstrap marker updates do not leave partially written state on interruption.
- `src/vllm-sr/tests/test_runtime_kb.py` covers corrupt-state fallback and atomic-write cleanup.
- The state taxonomy now records CLI runtime KB bootstrap markers as restart-safe local state under the generated runtime config surface.
- TD034 remains open for shared selector-store parity beyond local files, production session-store policy, server-side chat/session product decisions, durable config projections, and full local smoke/E2E reruns once Docker is available.
- Validation passed: `python -m pytest src/vllm-sr/tests/test_runtime_support.py`.
- Validation passed: `make vllm-sr-test`.

2026-05-25 ASR006 / TD034-TD004 CLI runtime support ownership split:

- Current-source recheck after the KB recovery slice found the implementation had improved behavior but still left `src/vllm-sr/cli/commands/runtime_support.py` and `src/vllm-sr/tests/test_runtime_support.py` above the structure warning threshold.
- `src/vllm-sr/cli/commands/runtime_paths.py` now owns `.vllm-sr` runtime path and runtime-config writes, `runtime_kb.py` owns KB source resolution and bootstrap marker state, and `runtime_config_mutation.py` owns algorithm and AMD GPU-default config mutation.
- `runtime.py` now keeps serve help text in `runtime_help.py` and dropped from 457 to 398 lines, while `runtime_support.py` dropped from 703 lines to 205 lines and now acts as the runtime orchestration seam; `test_runtime_support.py` dropped from 476 lines to 321 lines after KB coverage moved to `test_runtime_kb.py`.
- TD034 is narrowed for CLI local-state ownership; TD004 was rechecked separately and closed as stale, with residual Kind/shared-suite work kept under TD037.
- Validation passed: `python -m pytest src/vllm-sr/tests/test_runtime_support.py src/vllm-sr/tests/test_runtime_kb.py src/vllm-sr/tests/test_cli_main.py -q`.

2026-05-25 ASR008 / TD004 stale-debt closure:

- Current-source recheck found the original TD004 wording stale: the Python CLI now has `DeploymentBackend`, `DockerBackend`, `K8sBackend`, and `vllm-sr serve/status/logs/stop/dashboard --target k8s`.
- `src/vllm-sr/tests/test_deployment_backend.py` already covers target resolution, K8s backend helpers, Helm value translation, Docker backend parity, and CLI target routing.
- TD004 is closed; the remaining Kind lifecycle, kubeconfig discovery, local image loading, and shared-suite topology gap stays open under TD037.
- Validation passed: `python -m pytest src/vllm-sr/tests/test_deployment_backend.py -q`.
- Validation passed: `make agent-ci-gate CHANGED_FILES="src/vllm-sr/cli/commands/runtime.py,src/vllm-sr/cli/commands/runtime_help.py,src/vllm-sr/cli/commands/runtime_support.py,src/vllm-sr/cli/commands/runtime_kb.py,src/vllm-sr/cli/commands/runtime_paths.py,src/vllm-sr/cli/commands/runtime_config_mutation.py,src/vllm-sr/tests/test_runtime_support.py,src/vllm-sr/tests/test_runtime_kb.py,src/vllm-sr/tests/test_cli_main.py,docs/agent/architecture-scorecard.md,docs/agent/tech-debt/td-004-python-cli-kubernetes-workflow-separation.md,docs/agent/tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/state-taxonomy-and-inventory.md"`.

2026-05-24 ASR008 / TD030 route-manifest frontend unit-test slice:

- Current-source recheck confirmed the TD030 frontend test gap remained current for `dashboard/frontend/src`: the tree had Playwright coverage and a separate prompt test, but no source-level frontend unit test.
- `dashboard/frontend/src/app/routeManifest.ts` now owns the serializable authenticated shell route manifest, legacy redirect manifest, and setup-aware fallback target.
- `dashboard/frontend/src/app/AuthenticatedAppRoutes.tsx` now renders shell pages from that manifest instead of embedding route metadata directly in the React route tree.
- `dashboard/frontend/src/app/routeManifest.test.ts` adds the first Vitest unit-test coverage under `dashboard/frontend/src`, locking unique shell paths, playground compact-shell flags, legacy redirects, and setup fallback behavior.
- `dashboard/frontend/package.json` adds `test:unit`, and `tools/make/dashboard.mk` wires `dashboard-test-frontend` into `make dashboard-check`.
- TD030 remains open for large route pages, `dslStore.ts`, interaction containers, broader frontend coverage, and the existing lint-warning backlog.
- Validation passed: `npm run test:unit`; `npm run type-check`; `npm run lint`; `make dashboard-check`.

2026-05-24 ASR008 / TD030 dashboard overview row-model slice:

- Current-source recheck confirmed `DashboardPage.tsx` still carried section-local signal breakdown and decision preview row shaping inline, even after the route-shell split.
- `dashboard/frontend/src/pages/dashboardPageOverview.ts` now owns the pure row models for signal breakdown bars and decision preview rows.
- `dashboard/frontend/src/pages/DashboardPage.tsx` now delegates those row calculations to the sibling support module and drops from 471 to 468 lines for this slice.
- `dashboard/frontend/src/pages/dashboardPageOverview.test.ts` covers signal sorting/scaling/fallback colors, empty signal rows, decision category labels, model-name formatting, generated names, and preview limiting.
- TD030 remains open for `dslStore.ts`, broad route pages, large interaction containers, and the existing frontend lint-warning backlog.
- Validation passed: `npm run test:unit`; `npm run type-check`; `npm run lint`; `make dashboard-check`.

2026-05-24 ASR008 / TD030 projections dependency-stability lint slice:

- Current-source recheck found the Dashboard lint backlog still included three real hook-dependency warnings in `dashboard/frontend/src/pages/ConfigPageProjectionsSection.tsx`: fallback arrays for projections, scores, and mappings were recreated on every render and then used as `useMemo` dependencies.
- The section now uses module-level empty projection arrays for those fallbacks, preserving behavior while making the dependencies stable.
- The Dashboard frontend lint backlog dropped from 58 to 55 warnings; the remaining warnings are outside this slice and still tracked as TD030/frontend debt.
- Validation passed: `npm run lint`; `make dashboard-check`.

2026-05-24 ASR008 / TD030 frontend lint boundary cleanup slice:

- Current-source recheck found the remaining Dashboard frontend lint backlog was mostly `react-refresh/only-export-components` firing on files that repo-local rules classify as support modules rather than Fast Refresh component boundaries.
- `dashboard/frontend/eslint.config.js` now keeps the rule active for normal component files while disabling it for explicit support-module patterns and allowing `AuthContext.tsx` to export the `useAuth` hook.
- `dashboard/frontend/src/components/ColorBends.tsx` now initializes Three.js uniforms with neutral constants and lets the existing prop-sync effect update runtime values, removing the final hook-dependency warning without recreating the WebGL renderer on prop changes.
- Dashboard frontend lint dropped from 55 warnings to 0 warnings.
- TD030 remains open for broad route pages, large interaction containers, and `dslStore.ts` control-plane extraction.
- Validation passed: `npm run lint`; `npm run test:unit`; `npm run type-check`; `make dashboard-check`.

## Evidence Snapshot

Repository scale and surfaces:

- 3,635 tracked files.
- Main tracked file counts include 1,181 Go files, 332 Python files, 150 Rust files, 364 TypeScript/TSX files, 405 YAML/YML files, and 584 Markdown files.
- Source-like line inventory is dominated by `src/semantic-router`, `dashboard/frontend`, `website`, `candle-binding`, `e2e`, `src/training`, `dashboard/backend`, and `deploy/operator`.

Test and validation shape:

- Go test inventory: 405 `*_test.go` files, including 332 under `src/semantic-router`, 49 under `dashboard/backend`, 6 under `deploy/operator`, and 7 under `e2e/pkg`.
- Python test inventory: 58 `test_*.py` / `*_test.py` files.
- Dashboard frontend now has source-level Vitest unit tests at `dashboard/frontend/src/app/routeManifest.test.ts`, `dashboard/frontend/src/pages/dashboardPageOverview.test.ts`, `dashboard/frontend/src/utils/authFetch.test.ts`, and `dashboard/frontend/src/contexts/authSession.test.ts`; `make dashboard-check` runs them through `dashboard-test-frontend`, and `npm run lint` exits with 0 warnings; broader page/container unit coverage is still sparse, with separate Playwright coverage under `dashboard/frontend/e2e`.
- Dashboard auth now has backend coverage for HttpOnly session-cookie issuance, logout clearing, explicit cookie expiry, cookie-only request authentication, normalized bearer/cookie/query token intake, persisted session lifecycle, and logout-time revocation in `dashboard/backend/auth/session_cookie_test.go`, `dashboard/backend/auth/session_store_test.go`, and `dashboard/backend/auth/middleware_test.go`, while preserving the existing token response contract. Dashboard auth ownership is split across focused public/admin handler files plus `dashboard/backend/auth/store.go`, `dashboard/backend/auth/session_store.go`, `dashboard/backend/auth/permission_store.go`, and `dashboard/backend/auth/audit_store.go`.
- PR0 harness validation: `make agent-validate` passed and `make agent-scorecard` reported validation status `pass` with 0 validation errors.

Harness scorecard shape:

- 105 indexed harness docs.
- 104 governed docs.
- 30 change surfaces.
- 12 primary skills, 14 fragment skills, and 4 support skills.
- 13 local `AGENTS.md` hotspot supplements.
- 16 open technical debt entries.
- 57 open execution-plan tasks.
- `make shellcheck` now ignores local `.venv-*` directories, matching the existing repo-agent virtualenv exclusion intent and preventing third-party virtualenv scripts from failing changed-set gates.

Hotspot and debt evidence:

- `deploy/operator/api/v1alpha1/semanticrouter_types.go` is 1,820 lines and remains a CRD contract hotspot.
- `deploy/operator/controllers/canonical_config_builder.go` dropped from 314 to 58 lines after backend discovery and CRD spec-family translation moved to sibling controller files; TD028 remains open for API schema, webhook, and fixture ownership.
- `dashboard/frontend/src/app/AppRouter.tsx` dropped from 305 to 75 lines after authenticated route registration moved to `dashboard/frontend/src/app/AuthenticatedAppRoutes.tsx`; route metadata now lives in `dashboard/frontend/src/app/routeManifest.ts` with `dashboard/frontend/src/app/routeManifest.test.ts` coverage. `dashboard/frontend/src/pages/DashboardPage.tsx` is 468 lines after overview row models moved to `dashboard/frontend/src/pages/dashboardPageOverview.ts` with unit coverage; TD030 remains open for the editor store and larger page or container hotspots.
- `dashboard/backend/auth/store.go` dropped from 525 to 245 lines after auth session, role/permission, and audit persistence moved to sibling store modules. The former `dashboard/backend/auth/http_handlers.go` 556-line handler hotspot is removed; public auth handlers, admin user handlers, admin metadata handlers, and session-user helpers now live in focused files with the largest new file at 290 lines.
- `dashboard/frontend/src/components/ChatComponent.tsx` is 798 lines and remains a large interaction-container hotspot.
- `src/semantic-router/pkg/extproc/server.go` dropped from 542 to 347 lines after config watch/reload ownership moved to `src/semantic-router/pkg/extproc/server_config_watch.go`; `src/semantic-router/pkg/extproc/server_reload_test.go` dropped from 683 to 283 lines after runtime/config precedence and reload-runtime publication coverage moved to focused sibling test files.
- `src/semantic-router/pkg/selection/gmtrouter.go` now delegates local-state file IO to `src/semantic-router/pkg/selection/gmtrouter_storage.go`, which has focused tests for atomic persistence support behavior and corrupted-state backups.
- `src/fleet-sim/fleet_sim/optimizer/base.py` is 621 lines after extracting threshold Pareto helpers, and still mixes analytical sizing, DES verification, simulation reporting, and compatibility exports.
- `dashboard/frontend/src/pages/BuilderPage.tsx` is 725 lines, `src/vllm-sr/cli/core.py` is 563 lines, and `src/vllm-sr/cli/commands/runtime.py` is 447 lines.
- The debt index already tracks the main architecture gaps in TD015, TD020, TD027, TD028, TD030, TD031, TD033, TD034, TD037, and TD039, with TD036, TD038, and TD040 kept as resolved-contract regression guards.
- Response API Redis restart recovery is represented by `e2e/testcases/response_api_restart_recovery.go`; default responsestore package tests no longer depend on a live Redis instance after the integration test build-tag split.

Release evidence:

- `src/vllm-sr/pyproject.toml` is at version `0.3.0`.
- `candle-binding/Cargo.toml` and its lockfile package entry are at version `0.3.0`.
- `deploy/helm/semantic-router/Chart.yaml` is at chart version `0.2.0` and uses `appVersion: "latest"` as a documented default.
- `tools/release/check_version_contract.py` is the shared local/CI release
  checker for Python/Candle/Candle-lockfile version alignment, Helm release
  packaging semantics, Docker plus Operator release image coverage in unified
  release notes, upgrade runbook full-image and fixture-command coverage,
  Candle crate publish and native artifact workflow ownership, and the
  independent `vllm-sr-sim` PyPI release workflow.
- `tools/release/release_contract_markers.py` owns static release marker sets
  used by the shared checker.
- `.github/workflows/release.yml` calls the shared release checker against `v*`
  tags and publishes unified release notes.
- `src/vllm-sr/scripts/release.sh` updates the Python package and Candle crate
  versions together before tagging, then runs the same shared release checker.
- Docker, Helm, PyPI, crate, operator, dashboard, and simulator publishing are
  distributed across separate workflows, with simulator publishing owned by the
  `vllm-sr-sim-v*` tag stream.

## PR Chain

Use staged PRs. Do not combine these into one broad rewrite.

1. Contract/API ratchet
   - Target remaining TD015 and TD039; keep TD036, TD038, and TD040 as resolved-contract regression guards rather than open debt targets.
   - Move control planes toward contract-owned seams instead of deep router runtime imports.
   - Preserve `/api/v1/*`, `/v1/*`, canonical v0.3 YAML/DSL, CRD, CLI command, and Helm compatibility unless a migration is explicitly documented and tested.
2. Runtime/state ratchet
   - Target TD031 and TD034.
   - Reduce process-wide runtime globals, make restart-sensitive state explicit, and keep memory-backed behavior documented as ephemeral or local-dev only.
3. Router pipeline ratchet
   - Target TD020.
   - Keep TD023 and TD029 as historical request/response pipeline guards rather than active debt targets.
   - Split the remaining classification hotspot into narrower backend discovery, family adapter, and request-time orchestration owners.
4. Control-plane implementation ratchet
   - Target TD004, TD027, TD028, TD030, and TD037.
   - Reduce Dashboard, Operator, Python CLI, Fleet Sim, and dev integration hotspots without widening public contracts.
5. Release/docs/native ratchet
   - Target TD033 plus release and documentation score gaps.
   - Align versioned artifacts, native binding parity, upgrade/rollback guidance, and doc freshness with validation.

## Update Protocol

Every PR in this chain must update this document when it changes architecture
evidence or score:

- adjust only dimensions supported by the PR evidence
- link to the relevant debt or plan item
- list the validation command that passed
- keep unresolved gaps in the score deductions instead of hiding them in prose

If a PR discovers a durable architecture gap that is not already covered by a
debt item, add or update an entry under `docs/agent/tech-debt/` in the same PR.

## Validation

PR0 created the scorecard and execution plan as a docs-only harness change.

Required PR0 gates:

- `make agent-validate`
- `make agent-scorecard`
- `make agent-ci-gate CHANGED_FILES="docs/agent/architecture-scorecard.md,docs/agent/plans/pl-0032-architecture-scorecard-ratchet.md,docs/agent/plans/README.md,docs/agent/README.md,tools/agent/repo-manifest.yaml"`

Code-bearing follow-up PRs must start with:

- `make agent-report ENV=cpu CHANGED_FILES="..."`

Then run the smallest applicable gate reported by the harness, such as
`make test-semantic-router`, `make dashboard-check`, `make generate-crd`,
`make vllm-sr-test`, `make vllm-sr-sim-test`, or `make test-binding-minimal`.
