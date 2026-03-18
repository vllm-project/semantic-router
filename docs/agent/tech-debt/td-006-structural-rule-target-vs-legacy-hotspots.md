# TD006: Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots

## Status

Open

## Scope

architecture ratchet versus current code

## Summary

The harness correctly ratchets the repo toward smaller modules, but several legacy hotspots still depend on explicit exceptions. The v0.3 config-contract rollout also touched additional hotspot files that still need extraction-first follow-up before they can satisfy the structural target directly. The current rule layer now distinguishes three kinds of temporary relief for those files: file-size ratchets, per-function ratchets, and interface-size ratchets.

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
- [src/semantic-router/pkg/apiserver/server_test.go](../../../src/semantic-router/pkg/apiserver/server_test.go)
- [src/semantic-router/pkg/cache/cache_interface.go](../../../src/semantic-router/pkg/cache/cache_interface.go)
- [src/semantic-router/pkg/cache/cache_test.go](../../../src/semantic-router/pkg/cache/cache_test.go)
- [src/semantic-router/pkg/extproc/memory_helpers_test.go](../../../src/semantic-router/pkg/extproc/memory_helpers_test.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_external.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_external.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_hybrid.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_hybrid.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_mcp.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_mcp.go)
- [src/semantic-router/pkg/extproc/req_filter_rag_milvus.go](../../../src/semantic-router/pkg/extproc/req_filter_rag_milvus.go)
- [src/semantic-router/pkg/extproc/res_filter_jailbreak_test.go](../../../src/semantic-router/pkg/extproc/res_filter_jailbreak_test.go)
- [src/semantic-router/pkg/extproc/server.go](../../../src/semantic-router/pkg/extproc/server.go)
- [src/semantic-router/pkg/imagegen/backend_vllm_omni.go](../../../src/semantic-router/pkg/imagegen/backend_vllm_omni.go)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
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
- The Candle binding still concentrates YAML walking, default resolution, and compatibility fallbacks inside `candle-binding/src/core/config_loader.rs`; even narrow canonical-v0.3 fixes there currently trip the shared file/function limits before any extraction work can land.
- The rule layer now carries explicit `file_checks: relaxed`, `function_checks: relaxed`, and `interface_checks: relaxed` entries for the config-rollout hotspots so CI keeps ratcheting against known debt instead of blocking every rebase or schema follow-up on unchanged legacy structure.
- The Go lint layer now mirrors that posture for `canonical_config.go` and `canonical_loader_test.go`, so changed-file checks only fail on new regressions instead of re-reporting the existing canonical-config hotspot debt on every repair branch.
- The same Go lint posture now also explicitly covers the operator API validation tests under `deploy/operator/api/v1alpha1`, so sample/CRD regression maintenance is tracked as legacy structural debt instead of repeatedly blocking unrelated harness or schema follow-up work.
- The same Go lint posture now also explicitly covers the API server regression suite, extproc RAG/server helpers, and the cache interface/benchmark test hotspot so changed-file validation stops conflating legacy complexity debt with unrelated follow-up work in those packages.
- The same posture now explicitly covers `candle-binding/src/core/config_loader.rs`; this keeps the current audit focused on canonical-path correctness while preserving the requirement to extract the file later instead of blessing its current size and nesting as precedent.
- Signal-runtime follow-up now also intersects `src/semantic-router/pkg/classification/classifier.go`, `src/semantic-router/pkg/classification/embedding_classifier.go`, and the corresponding embedding regression test. Those files still exceed shared `cyclop`, `gocognit`, and `nestif` thresholds, so narrow fixes like embedding top-k or preference-default repairs re-enter hotspot debt instead of getting a clean changed-file lint result.
- This is the right governance posture, but it remains a real code/spec gap until the worst hotspots no longer need special handling.

## Desired End State

- The global structure rules become the common case rather than something many hotspot directories can only approach gradually.
- Config contract rollout work can land by extending narrower helper modules instead of growing dashboard handler, operator controller, or CLI hotspot files.
- Signal-runtime fixes land by extending narrower classifier helpers instead of reopening the monolithic `classifier.go` and `embedding_classifier.go` hotspots for every routing tweak.
- The temporary ratchet extensions added for the v0.3 rollout can be removed once the dashboard/backend handlers and tests, operator controller/types/tests, config tests, DSL compiler/decompiler, response-store interfaces, CLI schema modules, and Candle binding config loader are extracted below the structural thresholds.

## Exit Criteria

- The highest-risk files no longer need special ratchet treatment to stay within the intended modularity envelope.
- Config rollout follow-up extracts stable schema/export/merge helpers out of the current hotspot files, simplifies canonical-config regression tests, breaks up oversized dashboard/operator regression suites, and decomposes the Candle binding config loader enough for the relevant lint and structure gates to pass without bespoke exceptions.
- Signal-runtime follow-up extracts reusable signal-family evaluators, match aggregation helpers, and embedding scoring/test fixtures so classifier maintenance no longer depends on the monolithic hotspot files.
