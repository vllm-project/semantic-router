# TD023: Extproc Request Pipeline Phases Have Collapsed Across Request Filters

## Status

Closed

## Scope

`src/semantic-router/pkg/extproc/req_filter_*.go`, request-routing orchestration, and adjacent request-context mutation helpers

## Summary

The extproc request path has accumulated several distinct concerns into overlapping request filters. `req_filter_classification.go` now owns prompt compression, signal extraction, matched-signal propagation, decision evaluation, plugin bootstrap, and auto-model selection context. `req_filter_memory.go` combines search gating heuristics, personal/greeting heuristics, LLM-backed query rewriting, and memory request shaping. `req_filter_response_api.go` translates transport formats, chains conversation history, and stores response lifecycle state. These are active product behaviors, but the current seams make the request pipeline hard to reason about because semantic evaluation, enrichment, and transport translation are not clearly separated.

## Evidence

- [src/semantic-router/pkg/extproc/req_filter_classification.go](../../../src/semantic-router/pkg/extproc/req_filter_classification.go)
- [src/semantic-router/pkg/extproc/req_filter_classification_signal.go](../../../src/semantic-router/pkg/extproc/req_filter_classification_signal.go)
- [src/semantic-router/pkg/extproc/req_filter_classification_runtime.go](../../../src/semantic-router/pkg/extproc/req_filter_classification_runtime.go)
- [src/semantic-router/pkg/extproc/req_filter_memory_search.go](../../../src/semantic-router/pkg/extproc/req_filter_memory_search.go)
- [src/semantic-router/pkg/extproc/req_filter_memory_rewrite.go](../../../src/semantic-router/pkg/extproc/req_filter_memory_rewrite.go)
- [src/semantic-router/pkg/extproc/req_filter_memory_context.go](../../../src/semantic-router/pkg/extproc/req_filter_memory_context.go)
- [src/semantic-router/pkg/extproc/req_filter_response_api.go](../../../src/semantic-router/pkg/extproc/req_filter_response_api.go)
- [src/semantic-router/pkg/extproc/req_filter_modality.go](../../../src/semantic-router/pkg/extproc/req_filter_modality.go)
- [src/semantic-router/pkg/extproc/processor_req_body_routing.go](../../../src/semantic-router/pkg/extproc/processor_req_body_routing.go)
- [src/semantic-router/pkg/extproc/processor_req_body_memory.go](../../../src/semantic-router/pkg/extproc/processor_req_body_memory.go)
- [src/semantic-router/pkg/extproc/req_filter_looper_internal.go](../../../src/semantic-router/pkg/extproc/req_filter_looper_internal.go)
- [src/semantic-router/pkg/extproc/req_filter_memory_context_test.go](../../../src/semantic-router/pkg/extproc/req_filter_memory_context_test.go)
- [src/semantic-router/pkg/extproc/req_filter_classification_signal_test.go](../../../src/semantic-router/pkg/extproc/req_filter_classification_signal_test.go)
- [src/semantic-router/pkg/extproc/AGENTS.md](../../../src/semantic-router/pkg/extproc/AGENTS.md)

## Why It Matters

- New request-time behavior can require edits in multiple filters that each mutate the same request context, increasing the chance of phase-order regressions.
- Transport-specific features such as Response API translation are harder to evolve independently because they live near semantic routing and memory-enrichment logic.
- Memory and query-rewrite behavior are harder to test in isolation when they are coupled to request heuristics and router config resolution in one file.
- The package already has local rules for processor/router hotspots, but the phase-collapsed request filters still lack an equally explicit boundary story.

## Desired End State

- The extproc request path is expressed as explicit phases such as transport translation, prompt preprocessing, signal evaluation, enrichment, and post-decision plugin preparation.
- Each phase owns one dominant responsibility and exposes a narrow handoff contract instead of mutating broad request context opportunistically.
- Memory gating/query rewriting is separated from memory transport or request-body injection concerns.
- Response API translation and conversation-history handling stay at the transport edge instead of sharing ownership with decision evaluation.

## Exit Criteria

- Prompt compression and signal evaluation are factored enough that adding a new signal or preprocessing behavior does not require reopening one giant request-evaluation flow.
- Memory search gating, query rewriting, and request enrichment are split into narrower helpers or phase-specific modules.
- Response API translation and persistence state are isolated behind a dedicated transport seam.
- Extproc-local `AGENTS.md` explicitly calls out the request-filter hotspots and the phase-boundary rules contributors must follow.

## Resolution

- Memory request-time responsibilities are now split into dedicated search-gating, query-rewrite, and memory-context helpers, while request-body enrichment stays in `processor_req_body_memory.go`.
- Signal-phase responsibilities are now split between signal-input preparation/compression, signal-result projection, and decision/runtime orchestration helpers instead of continuing to grow inline inside `req_filter_classification.go`.
- Response API translation already remained behind `req_filter_response_api.go`, so after the memory and signal extractions the remaining request-time seams align with the intended transport vs evaluation vs enrichment split.
- Extproc-local `AGENTS.md` explicitly names the request-filter hotspots and the narrower phase rules contributors should follow.

## Validation

- `go test ./pkg/extproc -run 'Test(PrepareSignalEvaluationInput|ApplySignalResultsToContext|CollectMatchedSignalRules|ShouldSearchMemory|ContainsPersonalPronoun|IsGreeting|BuildSearchQuery|ExtractConversationHistory|FormatHistoryForPrompt|TruncateForLog|FormatMemoriesAsContext|InjectMemoryMessages)'`
- `make test-semantic-router`
- `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/req_filter_memory_search.go,src/semantic-router/pkg/extproc/req_filter_memory_rewrite.go,src/semantic-router/pkg/extproc/req_filter_memory_context.go,src/semantic-router/pkg/extproc/req_filter_memory_context_test.go,src/semantic-router/pkg/extproc/req_filter_memory.go,src/semantic-router/pkg/extproc/req_filter_classification.go,src/semantic-router/pkg/extproc/req_filter_classification_signal.go,src/semantic-router/pkg/extproc/req_filter_classification_signal_test.go,src/semantic-router/pkg/extproc/req_filter_classification_runtime.go"`
