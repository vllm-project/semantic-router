# TD006: Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots

## Status

Open

## Scope

architecture ratchet versus current code

## Summary

The harness correctly ratchets the repo toward smaller modules, but several legacy hotspots still depend on explicit exceptions. The v0.3 config-contract rollout also touched additional hotspot files that still need extraction-first follow-up before they can satisfy the structural target directly. The same mismatch now also covers maintained router-core seams in classification, extproc, looper, selection, modelselection, tools, and Milvus-backed memory helpers. The current rule layer now distinguishes three kinds of temporary relief for those files: file-size ratchets, per-function ratchets, and interface-size ratchets, and a narrower set of router-core seams now also requires explicit relaxed file/function checks so branch-local repairs stop failing on inherited structural growth that already shipped on the active branch.

## Evidence

- [docs/agent/architecture-guardrails.md](../architecture-guardrails.md)
- [docs/agent/repo-map.md](../repo-map.md)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [tools/linter/go/.golangci.agent.yml](../../../tools/linter/go/.golangci.agent.yml)
- [tools/agent/scripts/agent_doc_validation.py](../../../tools/agent/scripts/agent_doc_validation.py)
- [dashboard/frontend/src/pages/OpenClawPage.tsx](../../../dashboard/frontend/src/pages/OpenClawPage.tsx)
- [dashboard/frontend/src/components/ClawRoomChat.tsx](../../../dashboard/frontend/src/components/ClawRoomChat.tsx)
- [dashboard/backend/handlers/openclaw.go](../../../dashboard/backend/handlers/openclaw.go)
- [dashboard/backend/handlers/openclaw_rooms.go](../../../dashboard/backend/handlers/openclaw_rooms.go)
- [dashboard/backend/handlers/openclaw_teams.go](../../../dashboard/backend/handlers/openclaw_teams.go)
- [dashboard/backend/handlers/openclaw_workers.go](../../../dashboard/backend/handlers/openclaw_workers.go)
- [dashboard/backend/handlers/openclaw_provision.go](../../../dashboard/backend/handlers/openclaw_provision.go)
- [dashboard/backend/handlers/openclaw_test.go](../../../dashboard/backend/handlers/openclaw_test.go)
- [dashboard/backend/handlers/config.go](../../../dashboard/backend/handlers/config.go)
- [dashboard/backend/handlers/deploy.go](../../../dashboard/backend/handlers/deploy.go)
- [dashboard/backend/handlers/deploy_test.go](../../../dashboard/backend/handlers/deploy_test.go)
- [deploy/operator/controllers/semanticrouter_controller.go](../../../deploy/operator/controllers/semanticrouter_controller.go)
- [deploy/operator/controllers/semanticrouter_controller_test.go](../../../deploy/operator/controllers/semanticrouter_controller_test.go)
- [deploy/operator/controllers/helpers_test.go](../../../deploy/operator/controllers/helpers_test.go)
- [deploy/operator/api/v1alpha1/sample_validation_test.go](../../../deploy/operator/api/v1alpha1/sample_validation_test.go)
- [deploy/operator/api/v1alpha1/semanticrouter_types_test.go](../../../deploy/operator/api/v1alpha1/semanticrouter_types_test.go)
- [deploy/operator/api/v1alpha1/semanticrouter_webhook_test.go](../../../deploy/operator/api/v1alpha1/semanticrouter_webhook_test.go)
- [src/semantic-router/pkg/config/canonical_config.go](../../../src/semantic-router/pkg/config/canonical_config.go)
- [src/semantic-router/pkg/config/canonical_loader_test.go](../../../src/semantic-router/pkg/config/canonical_loader_test.go)
- [src/semantic-router/pkg/config/image_gen_plugin_test.go](../../../src/semantic-router/pkg/config/image_gen_plugin_test.go)
- [src/semantic-router/pkg/classification/classifier.go](../../../src/semantic-router/pkg/classification/classifier.go)
- [src/semantic-router/pkg/classification/embedding_classifier.go](../../../src/semantic-router/pkg/classification/embedding_classifier.go)
- [src/semantic-router/pkg/classification/embedding_classifier_test.go](../../../src/semantic-router/pkg/classification/embedding_classifier_test.go)
- [src/semantic-router/pkg/classification/hallucination_detector.go](../../../src/semantic-router/pkg/classification/hallucination_detector.go)
- [src/semantic-router/pkg/classification/mcp_classifier.go](../../../src/semantic-router/pkg/classification/mcp_classifier.go)
- [src/semantic-router/pkg/apiserver/server_test.go](../../../src/semantic-router/pkg/apiserver/server_test.go)
- [src/semantic-router/pkg/cache/cache_interface.go](../../../src/semantic-router/pkg/cache/cache_interface.go)
- [src/semantic-router/pkg/cache/cache_test.go](../../../src/semantic-router/pkg/cache/cache_test.go)
- [src/semantic-router/pkg/cache/hybrid_cache.go](../../../src/semantic-router/pkg/cache/hybrid_cache.go)
- [src/semantic-router/pkg/cache/inmemory_cache.go](../../../src/semantic-router/pkg/cache/inmemory_cache.go)
- [src/semantic-router/pkg/dsl/parser.go](../../../src/semantic-router/pkg/dsl/parser.go)
- [src/semantic-router/pkg/dsl/decompiler.go](../../../src/semantic-router/pkg/dsl/decompiler.go)
- [src/semantic-router/pkg/dsl/dsl_test.go](../../../src/semantic-router/pkg/dsl/dsl_test.go)
- [src/semantic-router/pkg/extproc/memory_helpers_test.go](../../../src/semantic-router/pkg/extproc/memory_helpers_test.go)
- [src/semantic-router/pkg/extproc/processor_res_cache.go](../../../src/semantic-router/pkg/extproc/processor_res_cache.go)
- [src/semantic-router/pkg/extproc/req_filter_looper.go](../../../src/semantic-router/pkg/extproc/req_filter_looper.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_external.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_external.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_hybrid.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_hybrid.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_mcp.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_mcp.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_milvus.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_milvus.go)
- [src/semantic-router/pkg/extproc/res_filter_jailbreak_test.go](../../../src/semantic-router/pkg/extproc/res_filter_jailbreak_test.go)
- [src/semantic-router/pkg/extproc/server.go](../../../src/semantic-router/pkg/extproc/server.go)
- [src/semantic-router/pkg/imagegen/backend_vllm_omni.go](../../../src/semantic-router/pkg/imagegen/backend_vllm_omni.go)
- [src/semantic-router/pkg/looper/base.go](../../../src/semantic-router/pkg/looper/base.go)
- [src/semantic-router/pkg/looper/client.go](../../../src/semantic-router/pkg/looper/client.go)
- [src/semantic-router/pkg/looper/confidence.go](../../../src/semantic-router/pkg/looper/confidence.go)
- [src/semantic-router/pkg/looper/ratings.go](../../../src/semantic-router/pkg/looper/ratings.go)
- [src/semantic-router/pkg/looper/rl_driven.go](../../../src/semantic-router/pkg/looper/rl_driven.go)
- [src/semantic-router/pkg/memory/milvus_store.go](../../../src/semantic-router/pkg/memory/milvus_store.go)
- [src/semantic-router/pkg/modelselection/selector.go](../../../src/semantic-router/pkg/modelselection/selector.go)
- [src/semantic-router/pkg/responsestore/redis_store.go](../../../src/semantic-router/pkg/responsestore/redis_store.go)
- [src/semantic-router/pkg/selection/automix.go](../../../src/semantic-router/pkg/selection/automix.go)
- [src/semantic-router/pkg/selection/factory.go](../../../src/semantic-router/pkg/selection/factory.go)
- [src/semantic-router/pkg/selection/gmtrouter.go](../../../src/semantic-router/pkg/selection/gmtrouter.go)
- [src/semantic-router/pkg/selection/hybrid.go](../../../src/semantic-router/pkg/selection/hybrid.go)
- [src/semantic-router/pkg/selection/rl_driven.go](../../../src/semantic-router/pkg/selection/rl_driven.go)
- [src/semantic-router/pkg/tools/tools.go](../../../src/semantic-router/pkg/tools/tools.go)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/training/model_classifier/verify_text_classification_datasets.py](../../../src/training/model_classifier/verify_text_classification_datasets.py)
- [candle-binding/src/core/config_loader.rs](../../../candle-binding/src/core/config_loader.rs)
- [dashboard/frontend/src/pages/ConfigPageDecisionsSection.tsx](../../../dashboard/frontend/src/pages/ConfigPageDecisionsSection.tsx)
- [dashboard/frontend/src/pages/ConfigPageModelsSection.tsx](../../../dashboard/frontend/src/pages/ConfigPageModelsSection.tsx)
- [dashboard/frontend/src/pages/ConfigPageSignalsSection.tsx](../../../dashboard/frontend/src/pages/ConfigPageSignalsSection.tsx)
- [dashboard/frontend/src/pages/configPageRouterDefaultsSupport.ts](../../../dashboard/frontend/src/pages/configPageRouterDefaultsSupport.ts)
- [dashboard/frontend/src/pages/configPageSupport.ts](../../../dashboard/frontend/src/pages/configPageSupport.ts)
- [docs/agent/plans/pl-0003-v0-3-config-contract-rollout.md](../plans/pl-0003-v0-3-config-contract-rollout.md)

## Why It Matters

- The harness correctly states that large hotspot files are debt, not precedent, but several code areas still depend on hotspot-specific exceptions and ratchets.
- The harness-side validation layer still includes at least one oversized script that remains above the warning threshold even after related changes land.
- OpenClaw management and dashboard UI now sit on the same debt boundary: the feature surface is active and maintained, but the implementation still spans oversized page/component/handler files that cannot yet satisfy the global structure target directly.
- The agent-specific Go complexity gate also needs explicit legacy exclusions for the same OpenClaw handlers/tests until those modules are decomposed enough to meet the global `cyclop`, `funlen`, `gocognit`, and `nestif` thresholds.
- The config-contract rollout now depends on additional large orchestrator files in dashboard/backend, operator generation, and CLI schema compatibility code; until those are decomposed, the structure ratchet will continue to surface exceptions when config work lands.
- The agent-specific Go complexity gate also still needs file-scoped exclusions for the canonical config parser and loader regression tests; otherwise small follow-up fixes in those hotspot files fail changed-file lint before any extraction work can land.
- The operator helper test suite now also sits above the shared file/function thresholds; adding focused regression coverage there should not require branch-local structural churn on every CI repair, so it is explicitly tracked as the same extraction-first debt instead of being treated as new precedent.
- The same config-rollout branch also depends on oversized dashboard/operator regression suites (`deploy_test.go` and `semanticrouter_controller_test.go`); those files now need explicit file-size ratchets until their table-driven fixtures are extracted out of the monolithic test files.
- The same structural gap now also covers the extracted ConfigPage support modules, operator API validation suites, extproc RAG and jailbreak helpers/tests, the API server regression suite, cache surface contracts/tests, and the vLLM Omni backend helper. Those files are active, maintained code, but they still exceed the shared file/function/interface thresholds often enough that changed-file validation needs explicit hotspot coverage until extraction work lands.
- The training-stack verifier added for multilingual text-classification dataset auditing currently ships as a large script-style entrypoint. It mixes CLI definition, dataset loading, judge-response parsing, threaded verification, correction export, and report generation in one file, so the shared structure gate still needs a temporary file/function ratchet until that workflow is split into adjacent helper modules.
- The Candle binding still concentrates YAML walking, default resolution, and compatibility fallbacks inside `candle-binding/src/core/config_loader.rs`; even narrow canonical-v0.3 fixes there currently trip the shared file/function limits before any extraction work can land.
- The rule layer now carries explicit `file_checks: relaxed`, `function_checks: relaxed`, and `interface_checks: relaxed` entries for the config-rollout hotspots so CI keeps ratcheting against known debt instead of blocking every rebase or schema follow-up on unchanged legacy structure.
- The Go lint layer now mirrors that posture for `canonical_config.go` and `canonical_loader_test.go`, so changed-file checks only fail on new regressions instead of re-reporting the existing canonical-config hotspot debt on every repair branch.
- The same Go lint posture now also explicitly covers the operator API validation tests under `deploy/operator/api/v1alpha1`, so sample/CRD regression maintenance is tracked as legacy structural debt instead of repeatedly blocking unrelated harness or schema follow-up work.
- The same Go lint posture now also explicitly covers the API server regression suite, extproc RAG/server helpers, and the cache interface/benchmark test hotspot so changed-file validation stops conflating legacy complexity debt with unrelated follow-up work in those packages.
- The same posture now explicitly covers `candle-binding/src/core/config_loader.rs`; this keeps the current audit focused on canonical-path correctness while preserving the requirement to extract the file later instead of blessing its current size and nesting as precedent.
- Signal-runtime follow-up now also intersects `src/semantic-router/pkg/classification/classifier.go`, `src/semantic-router/pkg/classification/embedding_classifier.go`, and the corresponding embedding regression test. Those files still exceed shared `cyclop`, `gocognit`, and `nestif` thresholds, so narrow fixes like embedding top-k or preference-default repairs re-enter hotspot debt instead of getting a clean changed-file lint result.
- DSL follow-up now also intersects `src/semantic-router/pkg/dsl/parser.go`, `src/semantic-router/pkg/dsl/decompiler.go`, and `src/semantic-router/pkg/dsl/dsl_test.go`. The structure rules already classify those files as relaxed legacy hotspots, but the Go agent lint layer had not mirrored that posture, so narrow routing-language fixes like TEST-block runtime validation still re-triggered unrelated historical complexity and errcheck debt.
- Routing-policy and looper follow-up now also intersects oversized evaluator and selector seams such as `src/semantic-router/pkg/looper/confidence.go`, `src/semantic-router/pkg/looper/client.go`, `src/semantic-router/pkg/modelselection/selector.go`, `src/semantic-router/pkg/selection/gmtrouter.go`, and `src/semantic-router/pkg/selection/rl_driven.go`. Those files are active maintained runtime surfaces, but they still exceed the shared file/function/nesting targets enough that changed-file validation needs explicit hotspot coverage until extraction work lands.
- The same mismatch now also covers maintained classifier, extproc, tooling, and Milvus-backed runtime helpers such as `src/semantic-router/pkg/classification/hallucination_detector.go`, `src/semantic-router/pkg/classification/mcp_classifier.go`, `src/semantic-router/pkg/extproc/req_filter_looper.go`, `src/semantic-router/pkg/memory/milvus_store.go`, and `src/semantic-router/pkg/tools/tools.go`. Without explicit ratchet coverage, small routing-policy or memory-path repairs in those seams keep failing on inherited structure debt instead of on new regressions.
- The branch-diff ratchet posture is still correct for untouched legacy hotspots, but several maintained router-core files on the current branch have already grown past `origin/main` as part of earlier feature work. Those specific files now need explicit `file_checks: relaxed` and/or `function_checks: relaxed` coverage in `tools/agent/structure-rules.yaml`; otherwise `make agent-ci-lint` keeps re-failing on historical branch debt even when a follow-up change only updates harness policy.
- The Go changed-file lint layer now also needs matching file-scoped exclusions for maintained cache, response-cache, Redis response-store, and selection/runtime hotspots. Otherwise a branch that only updates harness policy still fails `agent-changed-files-lint` because the current diff already carries inherited `cyclop`, `funlen`, `gocognit`, `nestif`, and `errcheck` debt in those router-core files.
- This is the right governance posture, but it remains a real code/spec gap until the worst hotspots no longer need special handling.

## Desired End State

- The global structure rules become the common case rather than something many hotspot directories can only approach gradually.
- Config contract rollout work can land by extending narrower helper modules instead of growing dashboard handler, operator controller, or CLI hotspot files.
- Training verification workflows land through smaller task-spec, dataset-loader, judge-runtime, and reporting helpers instead of one monolithic script entrypoint.
- Signal-runtime fixes land by extending narrower classifier helpers instead of reopening the monolithic `classifier.go` and `embedding_classifier.go` hotspots for every routing tweak.
- Routing-policy, looper, and selection fixes land by extending narrower evaluator, client, and selector helpers instead of reopening monolithic policy/runtime files for every threshold or telemetry tweak.
- Milvus-backed runtime work lands through shared lifecycle helpers and thinner domain adapters instead of continuing to widen `milvus_store.go` and adjacent store-specific seams.
- The temporary ratchet extensions added for the v0.3 rollout can be removed once the dashboard/backend handlers and tests, operator controller/types/tests, config tests, DSL compiler/decompiler, response-store interfaces, CLI schema modules, and Candle binding config loader are extracted below the structural thresholds.
- The temporary relaxed checks added for branch-carried router-core hotspots can be removed once the looper, selection, extproc, modelselection, tooling, and Milvus helpers are extracted back under the shared file/function limits.
- The temporary Go lint exclusions for the DSL parser/decompiler/test hotspots can be removed once those files are decomposed enough that spec-driven DSL work no longer depends on bespoke structural exceptions.

## Exit Criteria

- The highest-risk files no longer need special ratchet treatment to stay within the intended modularity envelope.
- Config rollout follow-up extracts stable schema/export/merge helpers out of the current hotspot files, simplifies canonical-config regression tests, breaks up oversized dashboard/operator regression suites, decomposes the training dataset verifier into adjacent helper modules, and decomposes the Candle binding config loader enough for the relevant lint and structure gates to pass without bespoke exceptions.
- Signal-runtime follow-up extracts reusable signal-family evaluators, match aggregation helpers, and embedding scoring/test fixtures so classifier maintenance no longer depends on the monolithic hotspot files.
- Routing-policy and memory follow-up extracts looper evaluators/clients, selector policies, extproc looper seams, and Milvus lifecycle helpers enough that those files no longer need hotspot ratchets to keep changed-file validation focused on new regressions.
- Cache, response-store, and response-pipeline follow-up extracts HNSW/cache search helpers, response-cache reconstruction helpers, Redis TLS/connection helpers, and selector policy seams enough that `.golangci.agent.yml` no longer needs file-scoped hotspot exclusions for those maintained runtime files.
