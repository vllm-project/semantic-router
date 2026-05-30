# TD020: Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators

## Status

Open

## Scope

`src/semantic-router/pkg/classification/**`, `src/semantic-router/pkg/services/classification.go`, and adjacent classifier bootstrap/discovery seams

## Summary

The classification subsystem has drifted into a shallow, high-churn design where giant orchestrator files own too many unrelated responsibilities at once. `classifier.go` mixes backend selection, model initialization, category and security mapping ownership, concurrency and metrics handling, and request-time family inference. The service layer partially duplicates discovery and bootstrap work, while the unified-classifier path and legacy path are layered unclearly. The result is a hotspot that resists change: adding a new backend or signal family tends to touch the same large constructor and shared state table instead of a narrow seam.

Current-source recheck on 2026-05-24 confirmed the debt remains open, but one
service-layer ownership slice has been narrowed. Legacy category, PII, and
jailbreak mapping bootstrap moved out of `pkg/services` and into
`classification.NewLegacyClassifierFromConfig`, with the mapping gate tests now
owned by `pkg/classification`. A 2026-05-25 recheck then found the next live
slice in `model_discovery.go`: directory normalization, model scanning,
LoRA/legacy classification, preferred architecture selection, and legacy
unified-classifier label loading were still bundled into discovery and
initialization. Scanning and preferred selection now live in
`classification.model_discovery_scan`, while legacy label loading lives in
`classification.unified_legacy_labels` with focused coverage for ordered
mappings and sparse mapping rejection. Services-side PII, security,
recommendation, and unified-batch API slices also moved out of the main service
orchestrator into focused `services.classification_*` helpers. The request-time
signal orchestration slice was narrowed next: jailbreak, PII, embedding, and
preference evaluators now live behind family/support files, and
`classifier_signal_context.go` dropped below the 400-line structure warning
threshold. Signal model-family lifecycle methods for fact-check,
hallucination, feedback, preference, and language then moved out of the
candidate model-selection file, so `classifier_model_select.go` now matches its
name and dropped from 423 to 146 lines. That extracted model-family lifecycle
file was later split by family as
`classification.classifier_fact_hallucination_lifecycle`,
`classification.classifier_feedback_lifecycle`,
`classification.classifier_preference_lifecycle`, and
`classification.classifier_language_lifecycle`, deleting the cross-family
holding file. The signal evaluation bridge was then
narrowed so public signal result contracts, used-signal dependency expansion,
authz header evaluation, and decision-engine bridging no longer share
`classifier_signal_eval.go`; that file now keeps only public all-signal
convenience entrypoints and dropped from 431 to 13 lines. The remaining debt is still
substantial: category, jailbreak, and PII native initializer/inference adapters
were then split out of the deleted 403-line `classifier_init.go` into
family-owned files. The service signal contract slice was also narrowed:
DTOs, response shaping, matched-signal extraction, and unmatched-signal
collection moved out of `services.classification_signal_contract`, which now
keeps only eval execution and decision-trace evaluation. Remaining
`embedding_classifier.go` was also narrowed: backend FFI indirection and
keyword-embedding model initialization now live in
`classification.embedding_classifier_backend`, candidate collection, worker
fanout, and preload result collection live in
`classification.embedding_classifier_preload_support`, while matched-rule
scoring, top-k shaping, cosine similarity, preload stats, and prototype-bank
rebuilds live in `classification.embedding_classifier_scoring`. The public
embedding request paths were narrowed again so
`classification.embedding_classifier_text` owns text classification and
`classification.embedding_classifier_multimodal` owns multimodal request
validation/cache/scoring, leaving `embedding_classifier.go` with construction,
state, model-type resolution, modality indexing, and DTOs only. Remaining
keyword rule preprocessing, ordered dispatch/cache state, and regex/fuzzy
matching were then split out of `keyword_classifier.go` into
`classification.keyword_classifier_regex`,
`classification.keyword_classifier_dispatch`, and
`classification.keyword_classifier_match`. The MCP category path was then
split so protocol/client lifecycle, response parsing, and `list_categories`
mapping conversion live in `classification.mcp_classifier_client`, while
Classifier-level bootstrap, category-name translation, entropy reasoning, and
probability-quality metrics live in `classification.mcp_classifier_runtime`.
Basic hallucination detection and enhanced NLI explanation were then separated:
`classification.hallucination_detector_nli` now owns NLI labels/results,
enhanced result contracts, NLI initialization/classification, enhanced span
filtering, threshold defaults, and explanation/severity adjustment. Classifier
construction was then narrowed: `classification.classifier_option_rules` owns
rule-family option builders, and `classification.classifier_option_backends`
owns category/MCP option wiring plus native jailbreak/PII dependency selection.
The KB classifier was then split so exemplar embedding preload and prototype
rebuilds live in `classification.category_kb_embeddings`, while label/group
matching, metric calculation, and result shaping live in
`classification.category_kb_scoring`. Modality detection was then separated
from category entropy routing: `classification.classifier_modality` owns
classifier, keyword, and hybrid modality detection while
`classifier_category_entropy.go` keeps category-with-entropy orchestration,
fallback handling, and entropy metrics. The central request-time signal file was
then narrowed again: keyword, domain, fact-check, user-feedback, reask, and
context evaluator implementations now live in
`classification.classifier_signal_rule_evaluators`, complexity signal result
mutation and metric emission live in `classification.classifier_signal_complexity`,
and request-time modality signal mutation lives in
`classification.classifier_signal_modality_eval`. `classifier_signal_context.go`
now keeps readiness, text selection, image-cache setup, dispatcher execution,
and post-processing only. Remaining signal grouping, projection,
discovery/bootstrap contracts, and long-term protection against the central
signal dispatcher growing back into a broad hotspot still need further
extraction. Signal group ownership was then narrowed: group winner/default
resolution plus softmax scoring moved to
`classification.classifier_signal_group_resolution`, partition trace construction
moved to `classification.classifier_signal_group_trace`, and matched-signal
output limiting moved to `classification.classifier_signal_output_policy`, so
`classifier_signal_groups.go` keeps only group entrypoints and member resolution.
Projection ownership was then narrowed: dependency ordering moved to
`classification.classifier_projection_order`, input accessors and matching moved
to `classification.classifier_projection_inputs`, output matching/confidence and
boundary-distance math moved to `classification.classifier_projection_outputs`,
and trace merge behavior moved to `classification.classifier_projection_trace`.
`classifier_projections.go` now keeps only the projection execution entrypoint.
Unified classifier ownership was narrowed again after the native result split:
LoRA binding initialization, native capability checks, and lazy-init
concurrency guards moved to `classification.unified_classifier_lora`, leaving
`unified_classifier.go` as legacy native initialization plus `ClassifyBatch`
dispatch. Model discovery ownership was then narrowed: path validation moved to
`classification.model_discovery_validation`, discovery-info response shaping
moved to `classification.model_discovery_info`, and unified classifier
auto-initialization moved to `classification.unified_auto_init`.
`model_discovery.go` now keeps model path contracts, architecture models,
public discovery entrypoints, and architecture detection only.
The vLLM jailbreak adapter was then split so
`classification.vllm_jailbreak_parser` owns parser selection, auto fallback,
Qwen3Guard safety/severity/category parsing, JSON parsing, simple parsing, and
category extraction, while `vllm_classifier.go` keeps remote vLLM request
orchestration and `ClassResult` mapping. Contrastive preference ownership was
then narrowed: concurrent rule-example embedding preload now lives in
`classification.contrastive_preference_embeddings`, while detailed query
scoring, deterministic score ordering, example collection, and prototype-bank
rebuilds live in `classification.contrastive_preference_scoring`.
`contrastive_preference_classifier.go` keeps public types, construction, and
threshold/margin decisioning only. The public preference wrapper was then split
so `classification.preference_classifier_external` owns the external LLM client,
prompt, route JSON, call orchestration, and output parser, while
`classification.preference_classifier_contrastive` owns contrastive construction
and conversation text extraction; `preference_classifier.go` now keeps only the
public wrapper and dispatch. Complexity classifier ownership was then
narrowed: candidate task fanout, embedding collection, and prototype-bank
rebuilds moved to `classification.complexity_candidate_embeddings`; text,
multimodal-text, and request-image query embedding loading moved to
`classification.complexity_query_embeddings`; and per-rule scoring, text/image
fusion, difficulty labeling, and result logging moved to
`classification.complexity_rule_scoring`. The split also fixed public
image-based complexity classification so it no longer requires an injected
request image cache. The shared prototype-bank helper was then narrowed:
`classification.prototype_bank` owns prototype contracts and bank construction,
`classification.prototype_clustering` owns dedupe, similarity-matrix,
clustering, tie-breaking, and medoid selection, while
`prototype_scoring.go` now keeps only score options and runtime query-to-bank
aggregation.

## Evidence

- [src/semantic-router/pkg/classification/classifier.go](../../../src/semantic-router/pkg/classification/classifier.go)
- [src/semantic-router/pkg/classification/classifier_builtin_models.go](../../../src/semantic-router/pkg/classification/classifier_builtin_models.go)
- [src/semantic-router/pkg/classification/classifier_construction.go](../../../src/semantic-router/pkg/classification/classifier_construction.go)
- [src/semantic-router/pkg/classification/classifier_option_backends.go](../../../src/semantic-router/pkg/classification/classifier_option_backends.go)
- [src/semantic-router/pkg/classification/classifier_option_rules.go](../../../src/semantic-router/pkg/classification/classifier_option_rules.go)
- [src/semantic-router/pkg/classification/classifier_category_init.go](../../../src/semantic-router/pkg/classification/classifier_category_init.go)
- [src/semantic-router/pkg/classification/classifier_jailbreak_init.go](../../../src/semantic-router/pkg/classification/classifier_jailbreak_init.go)
- [src/semantic-router/pkg/classification/classifier_model_select.go](../../../src/semantic-router/pkg/classification/classifier_model_select.go)
- [src/semantic-router/pkg/classification/classifier_modality.go](../../../src/semantic-router/pkg/classification/classifier_modality.go)
- [src/semantic-router/pkg/classification/classifier_pii_init.go](../../../src/semantic-router/pkg/classification/classifier_pii_init.go)
- [src/semantic-router/pkg/classification/classifier_projections.go](../../../src/semantic-router/pkg/classification/classifier_projections.go)
- [src/semantic-router/pkg/classification/classifier_projection_inputs.go](../../../src/semantic-router/pkg/classification/classifier_projection_inputs.go)
- [src/semantic-router/pkg/classification/classifier_projection_order.go](../../../src/semantic-router/pkg/classification/classifier_projection_order.go)
- [src/semantic-router/pkg/classification/classifier_projection_outputs.go](../../../src/semantic-router/pkg/classification/classifier_projection_outputs.go)
- [src/semantic-router/pkg/classification/classifier_projection_trace.go](../../../src/semantic-router/pkg/classification/classifier_projection_trace.go)
- [src/semantic-router/pkg/classification/classifier_signal_authz.go](../../../src/semantic-router/pkg/classification/classifier_signal_authz.go)
- [src/semantic-router/pkg/classification/classifier_signal_context.go](../../../src/semantic-router/pkg/classification/classifier_signal_context.go)
- [src/semantic-router/pkg/classification/classifier_signal_complexity.go](../../../src/semantic-router/pkg/classification/classifier_signal_complexity.go)
- [src/semantic-router/pkg/classification/classifier_signal_decision.go](../../../src/semantic-router/pkg/classification/classifier_signal_decision.go)
- [src/semantic-router/pkg/classification/classifier_signal_embedding_helpers.go](../../../src/semantic-router/pkg/classification/classifier_signal_embedding_helpers.go)
- [src/semantic-router/pkg/classification/classifier_signal_eval.go](../../../src/semantic-router/pkg/classification/classifier_signal_eval.go)
- [src/semantic-router/pkg/classification/classifier_signal_jailbreak.go](../../../src/semantic-router/pkg/classification/classifier_signal_jailbreak.go)
- [src/semantic-router/pkg/classification/classifier_signal_group_resolution.go](../../../src/semantic-router/pkg/classification/classifier_signal_group_resolution.go)
- [src/semantic-router/pkg/classification/classifier_signal_group_trace.go](../../../src/semantic-router/pkg/classification/classifier_signal_group_trace.go)
- [src/semantic-router/pkg/classification/classifier_fact_hallucination_lifecycle.go](../../../src/semantic-router/pkg/classification/classifier_fact_hallucination_lifecycle.go)
- [src/semantic-router/pkg/classification/classifier_feedback_lifecycle.go](../../../src/semantic-router/pkg/classification/classifier_feedback_lifecycle.go)
- [src/semantic-router/pkg/classification/classifier_language_lifecycle.go](../../../src/semantic-router/pkg/classification/classifier_language_lifecycle.go)
- [src/semantic-router/pkg/classification/classifier_preference_lifecycle.go](../../../src/semantic-router/pkg/classification/classifier_preference_lifecycle.go)
- [src/semantic-router/pkg/classification/classifier_signal_output_policy.go](../../../src/semantic-router/pkg/classification/classifier_signal_output_policy.go)
- [src/semantic-router/pkg/classification/classifier_signal_pii.go](../../../src/semantic-router/pkg/classification/classifier_signal_pii.go)
- [src/semantic-router/pkg/classification/classifier_signal_preference_support.go](../../../src/semantic-router/pkg/classification/classifier_signal_preference_support.go)
- [src/semantic-router/pkg/classification/classifier_signal_results.go](../../../src/semantic-router/pkg/classification/classifier_signal_results.go)
- [src/semantic-router/pkg/classification/classifier_signal_rule_evaluators.go](../../../src/semantic-router/pkg/classification/classifier_signal_rule_evaluators.go)
- [src/semantic-router/pkg/classification/classifier_signal_usage.go](../../../src/semantic-router/pkg/classification/classifier_signal_usage.go)
- [src/semantic-router/pkg/classification/classifier_signal_modality_eval.go](../../../src/semantic-router/pkg/classification/classifier_signal_modality_eval.go)
- [src/semantic-router/pkg/classification/embedding_classifier.go](../../../src/semantic-router/pkg/classification/embedding_classifier.go)
- [src/semantic-router/pkg/classification/embedding_classifier_backend.go](../../../src/semantic-router/pkg/classification/embedding_classifier_backend.go)
- [src/semantic-router/pkg/classification/embedding_classifier_multimodal.go](../../../src/semantic-router/pkg/classification/embedding_classifier_multimodal.go)
- [src/semantic-router/pkg/classification/embedding_classifier_preload_support.go](../../../src/semantic-router/pkg/classification/embedding_classifier_preload_support.go)
- [src/semantic-router/pkg/classification/embedding_classifier_scoring.go](../../../src/semantic-router/pkg/classification/embedding_classifier_scoring.go)
- [src/semantic-router/pkg/classification/embedding_classifier_text.go](../../../src/semantic-router/pkg/classification/embedding_classifier_text.go)
- [src/semantic-router/pkg/classification/prototype_bank.go](../../../src/semantic-router/pkg/classification/prototype_bank.go)
- [src/semantic-router/pkg/classification/prototype_clustering.go](../../../src/semantic-router/pkg/classification/prototype_clustering.go)
- [src/semantic-router/pkg/classification/prototype_scoring.go](../../../src/semantic-router/pkg/classification/prototype_scoring.go)
- [src/semantic-router/pkg/classification/keyword_classifier.go](../../../src/semantic-router/pkg/classification/keyword_classifier.go)
- [src/semantic-router/pkg/classification/keyword_classifier_dispatch.go](../../../src/semantic-router/pkg/classification/keyword_classifier_dispatch.go)
- [src/semantic-router/pkg/classification/keyword_classifier_match.go](../../../src/semantic-router/pkg/classification/keyword_classifier_match.go)
- [src/semantic-router/pkg/classification/keyword_classifier_regex.go](../../../src/semantic-router/pkg/classification/keyword_classifier_regex.go)
- [src/semantic-router/pkg/classification/category_kb_classifier.go](../../../src/semantic-router/pkg/classification/category_kb_classifier.go)
- [src/semantic-router/pkg/classification/category_kb_embeddings.go](../../../src/semantic-router/pkg/classification/category_kb_embeddings.go)
- [src/semantic-router/pkg/classification/category_kb_scoring.go](../../../src/semantic-router/pkg/classification/category_kb_scoring.go)
- [src/semantic-router/pkg/classification/classifier_category_entropy.go](../../../src/semantic-router/pkg/classification/classifier_category_entropy.go)
- [src/semantic-router/pkg/classification/hallucination_detector.go](../../../src/semantic-router/pkg/classification/hallucination_detector.go)
- [src/semantic-router/pkg/classification/hallucination_detector_nli.go](../../../src/semantic-router/pkg/classification/hallucination_detector_nli.go)
- [src/semantic-router/pkg/classification/mcp_classifier.go](../../../src/semantic-router/pkg/classification/mcp_classifier.go)
- [src/semantic-router/pkg/classification/mcp_classifier_client.go](../../../src/semantic-router/pkg/classification/mcp_classifier_client.go)
- [src/semantic-router/pkg/classification/mcp_classifier_runtime.go](../../../src/semantic-router/pkg/classification/mcp_classifier_runtime.go)
- [src/semantic-router/pkg/classification/unified_classifier.go](../../../src/semantic-router/pkg/classification/unified_classifier.go)
- [src/semantic-router/pkg/classification/unified_classifier_lora.go](../../../src/semantic-router/pkg/classification/unified_classifier_lora.go)
- [src/semantic-router/pkg/classification/unified_classifier_types.go](../../../src/semantic-router/pkg/classification/unified_classifier_types.go)
- [src/semantic-router/pkg/classification/unified_classifier_cgo_results.go](../../../src/semantic-router/pkg/classification/unified_classifier_cgo_results.go)
- [src/semantic-router/pkg/classification/unified_classifier_stub.go](../../../src/semantic-router/pkg/classification/unified_classifier_stub.go)
- [src/semantic-router/pkg/classification/unified_classifier_cgo_test.go](../../../src/semantic-router/pkg/classification/unified_classifier_cgo_test.go)
- [src/semantic-router/pkg/classification/vllm_classifier.go](../../../src/semantic-router/pkg/classification/vllm_classifier.go)
- [src/semantic-router/pkg/classification/vllm_jailbreak_parser.go](../../../src/semantic-router/pkg/classification/vllm_jailbreak_parser.go)
- [src/semantic-router/pkg/classification/vllm_jailbreak_parser_test.go](../../../src/semantic-router/pkg/classification/vllm_jailbreak_parser_test.go)
- [src/semantic-router/pkg/classification/contrastive_preference_classifier.go](../../../src/semantic-router/pkg/classification/contrastive_preference_classifier.go)
- [src/semantic-router/pkg/classification/contrastive_preference_embeddings.go](../../../src/semantic-router/pkg/classification/contrastive_preference_embeddings.go)
- [src/semantic-router/pkg/classification/contrastive_preference_scoring.go](../../../src/semantic-router/pkg/classification/contrastive_preference_scoring.go)
- [src/semantic-router/pkg/classification/preference_classifier.go](../../../src/semantic-router/pkg/classification/preference_classifier.go)
- [src/semantic-router/pkg/classification/preference_classifier_contrastive.go](../../../src/semantic-router/pkg/classification/preference_classifier_contrastive.go)
- [src/semantic-router/pkg/classification/preference_classifier_external.go](../../../src/semantic-router/pkg/classification/preference_classifier_external.go)
- [src/semantic-router/pkg/classification/preference_classifier_test.go](../../../src/semantic-router/pkg/classification/preference_classifier_test.go)
- [src/semantic-router/pkg/classification/complexity_classifier.go](../../../src/semantic-router/pkg/classification/complexity_classifier.go)
- [src/semantic-router/pkg/classification/complexity_candidate_embeddings.go](../../../src/semantic-router/pkg/classification/complexity_candidate_embeddings.go)
- [src/semantic-router/pkg/classification/complexity_query_embeddings.go](../../../src/semantic-router/pkg/classification/complexity_query_embeddings.go)
- [src/semantic-router/pkg/classification/complexity_rule_scoring.go](../../../src/semantic-router/pkg/classification/complexity_rule_scoring.go)
- [src/semantic-router/pkg/classification/complexity_classifier_test.go](../../../src/semantic-router/pkg/classification/complexity_classifier_test.go)
- [src/semantic-router/pkg/classification/model_discovery.go](../../../src/semantic-router/pkg/classification/model_discovery.go)
- [src/semantic-router/pkg/classification/model_discovery_info.go](../../../src/semantic-router/pkg/classification/model_discovery_info.go)
- [src/semantic-router/pkg/classification/model_discovery_scan.go](../../../src/semantic-router/pkg/classification/model_discovery_scan.go)
- [src/semantic-router/pkg/classification/model_discovery_validation.go](../../../src/semantic-router/pkg/classification/model_discovery_validation.go)
- [src/semantic-router/pkg/classification/unified_auto_init.go](../../../src/semantic-router/pkg/classification/unified_auto_init.go)
- [src/semantic-router/pkg/classification/unified_legacy_labels.go](../../../src/semantic-router/pkg/classification/unified_legacy_labels.go)
- [src/semantic-router/pkg/classification/unified_legacy_labels_test.go](../../../src/semantic-router/pkg/classification/unified_legacy_labels_test.go)
- [src/semantic-router/pkg/classification/legacy_factory.go](../../../src/semantic-router/pkg/classification/legacy_factory.go)
- [src/semantic-router/pkg/classification/legacy_factory_test.go](../../../src/semantic-router/pkg/classification/legacy_factory_test.go)
- [src/semantic-router/pkg/services/classification.go](../../../src/semantic-router/pkg/services/classification.go)
- [src/semantic-router/pkg/services/classification_pii.go](../../../src/semantic-router/pkg/services/classification_pii.go)
- [src/semantic-router/pkg/services/classification_pii_response.go](../../../src/semantic-router/pkg/services/classification_pii_response.go)
- [src/semantic-router/pkg/services/classification_security.go](../../../src/semantic-router/pkg/services/classification_security.go)
- [src/semantic-router/pkg/services/classification_signal_contract.go](../../../src/semantic-router/pkg/services/classification_signal_contract.go)
- [src/semantic-router/pkg/services/classification_signal_matched.go](../../../src/semantic-router/pkg/services/classification_signal_matched.go)
- [src/semantic-router/pkg/services/classification_signal_response.go](../../../src/semantic-router/pkg/services/classification_signal_response.go)
- [src/semantic-router/pkg/services/classification_signal_types.go](../../../src/semantic-router/pkg/services/classification_signal_types.go)
- [src/semantic-router/pkg/services/classification_recommendation.go](../../../src/semantic-router/pkg/services/classification_recommendation.go)
- [src/semantic-router/pkg/services/classification_unified_batch.go](../../../src/semantic-router/pkg/services/classification_unified_batch.go)
- [src/semantic-router/pkg/services/classification_test.go](../../../src/semantic-router/pkg/services/classification_test.go)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)

## Why It Matters

- A single hotspot becomes the edit point for backend discovery, mapping policy, inference flow, and metrics, which increases change risk and review cost.
- The current design makes it hard to test one classification seam in isolation because bootstrap, family logic, and runtime orchestration are entangled.
- The service layer no longer owns legacy mapping loading; model scanning, model path validation, discovery-info reporting, unified auto-initialization, legacy unified label loading, built-in category/jailbreak/PII family behavior, native category/jailbreak/PII init and inference adapters, vLLM jailbreak parser policy, contrastive preference embedding preload/scoring, preference external/contrastive wrapper ownership, complexity candidate/query/scoring ownership, request-time jailbreak/PII/embedding/preference signal evaluators, generic keyword/domain/fact-check/feedback/reask/context/complexity/modality evaluator ownership, signal group application/resolution/trace/output-policy ownership, projection ordering/input/output/trace ownership, signal result/usage/authz/decision bridge helpers, embedding backend initialization, embedding candidate-preload fanout, embedding matched-rule scoring/prototype maintenance, shared prototype-bank construction/clustering/scoring, keyword rule preprocessing, keyword dispatch/cache state, keyword regex/fuzzy matching, MCP client/protocol handling, MCP category mapping bootstrap, MCP entropy/metrics runtime ownership, hallucination detector basic/NLI runtime ownership, classifier option construction/backend selection/rule option building, KB embedding preload/prototype rebuild and label/group/metric scoring, modality detection, category entropy reasoning/metrics, family-owned fact-check/hallucination/feedback/preference/language lifecycle helpers, PII/security API handling, signal DTO/response/matched-signal helpers, recommendation helpers, unified-batch wrappers, unified-classifier shared type/stat helpers, LoRA lifecycle guards, and native result decoding now have narrow helper seams, but additional request-time orchestration seams remain split across service assembly and `classifier.go`.
- The structure-rule ratchet already treats these files as legacy hotspots, which confirms the code shape has exceeded the intended architecture.

## Desired End State

- Model discovery, path validation, discovery reporting, and auto-initialization live behind dedicated helpers or adapters instead of the main request-time classifier orchestration.
- Per-family classification concerns such as category, jailbreak, and PII inference can evolve behind narrow seams rather than shared giant structs.
- Native family initializer and inference adapters stay family-owned instead of sharing one category/jailbreak/PII file.
- vLLM remote inference transport stays separate from safety-output parser policy and parsing tests.
- Contrastive preference construction, rule embedding preload, detailed scoring, and prototype-bank rebuilds stay in separate files.
- External LLM preference prompt/call/parser handling and contrastive preference conversation extraction stay outside the public preference classifier wrapper.
- Complexity construction, candidate embedding preload, query embedding loading, per-rule scoring, and prototype-bank rebuilds stay in separate files.
- Request-time signal evaluators stay in family-owned files instead of growing the central signal dispatcher.
- Generic request-time evaluator implementations stay out of `classifier_signal_context.go`, which remains an orchestration/post-processing seam.
- Signal group application, winner/default resolution, partition trace construction, and output limiting stay in separate files.
- Projection execution, dependency ordering, input matching, output confidence/boundary math, and trace merging stay in separate files.
- Embedding backend initialization, FFI indirection, candidate preload fanout, matched-rule scoring, and prototype-bank maintenance stay outside the text/image request-flow orchestrator.
- Shared prototype-bank construction, clustering/compression, and runtime scoring stay in separate files so embedding, contrastive preference, complexity, and KB classifiers do not edit one mixed helper.
- Keyword classifier construction, ordered dispatch/cache state, regex preprocessing, and regex/fuzzy matching stay in separate files.
- KB manifest/classify orchestration, exemplar embedding/prototype rebuilds, and label/group/metric scoring stay in separate files.
- MCP client/protocol handling and MCP category runtime/entropy handling stay in separate files from the public MCP contracts.
- Basic hallucination detection and NLI-backed explanation/filtering stay in separate files.
- Fact-check/hallucination, feedback, preference, and language classifier lifecycle wrappers stay in family-owned files instead of a shared model-family holding file.
- Classifier option orchestration, rule-family option builders, and backend dependency selection stay in separate files.
- Modality detection stays separate from category entropy reasoning and metrics.
- Service API DTOs, response shaping, matched-signal extraction, and eval execution stay in separate files.
- Public signal result contracts, used-signal dependency expansion, authz header evaluation, and decision-engine bridging stay separate from family evaluators and convenience entrypoints.
- The unified batch path and legacy path expose explicit interfaces and wiring points instead of being mixed into one monolithic runtime surface; native invocation/result decoding and LoRA lifecycle guards should remain separate from public type/stat surfaces.
- Service-level composition owns assembly only; it does not reimplement mapping or discovery concerns that belong inside the classification package.

## Exit Criteria

- `classifier.go`, `classifier_construction.go`, `category_kb_classifier.go`, `classifier_category_entropy.go`, `complexity_classifier.go`, `embedding_classifier.go`, `prototype_scoring.go`, `unified_classifier.go`, `classifier_signal_context.go`, `classifier_signal_eval.go`, `classifier_signal_groups.go`, `classifier_projections.go`, `model_discovery.go`, `vllm_classifier.go`, `contrastive_preference_classifier.go`, `preference_classifier.go`, `mcp_classifier.go`, and `services/classification.go` have materially reduced responsibility counts, with family adapters, discovery/bootstrap logic, path validation, discovery reporting, native result decoding, native family init/inference adapters, unified LoRA lifecycle ownership, vLLM parser policy, contrastive preference embedding/scoring ownership, preference external/contrastive wrapper ownership, complexity candidate/query/scoring ownership, embedding backend/preload/scoring/request-path ownership, shared prototype-bank construction/clustering/scoring ownership, family-owned model lifecycle ownership, request-time signal evaluators, generic signal evaluator implementations, signal group resolution/trace/output-policy ownership, projection ordering/input/output/trace ownership, result/usage/authz/decision bridges, classifier option builder ownership, KB embedding/scoring ownership, modality detection ownership, MCP client/runtime ownership, and mapping ownership extracted into dedicated modules.
- New classifier backends or signal families can be added through narrow seams without editing a giant shared constructor or state table.
- Classification tests cover the extracted seams independently enough that bootstrap, mapping, and request-time inference can fail separately.
- The classification hotspot no longer requires relaxed structure treatment for the same responsibilities that are currently bundled together.
