# TD029: Extproc Response Pipeline Phases Still Collapse Normalization, Streaming, Replay, and Response-Side Warnings

## Status

Closed

## Scope

`src/semantic-router/pkg/extproc/processor_res_body*.go`, `src/semantic-router/pkg/extproc/res_filter_*.go`, and adjacent response-pipeline tests

## Summary

The extproc request path now has its own subsystem-specific debt record, but the response path still concentrates several distinct phases into one response pipeline seam. `processor_res_body.go` splits the top-level response flow into looper replay capture, provider normalization, streaming dispatch, and non-streaming handling. `processor_res_body_pipeline.go` then combines usage accounting, cache updates, Response API translation, response-side jailbreak and hallucination warnings, memory store scheduling, factual-warning shaping, and router-replay mutation. Streaming behavior owns TTFT accounting, chunk parsing, metadata projection, response reconstruction, and cache finalization in a parallel seam. These are active behaviors, but response changes still tend to cross normalization, streaming, replay, and warning/mutation responsibilities at once.

## Evidence

- [src/semantic-router/pkg/extproc/processor_res_body.go](../../../src/semantic-router/pkg/extproc/processor_res_body.go)
- [src/semantic-router/pkg/extproc/processor_res_body_pipeline.go](../../../src/semantic-router/pkg/extproc/processor_res_body_pipeline.go)
- [src/semantic-router/pkg/extproc/processor_res_body_streaming.go](../../../src/semantic-router/pkg/extproc/processor_res_body_streaming.go)
- [src/semantic-router/pkg/extproc/res_filter_hallucination.go](../../../src/semantic-router/pkg/extproc/res_filter_hallucination.go)
- [src/semantic-router/pkg/extproc/res_filter_jailbreak.go](../../../src/semantic-router/pkg/extproc/res_filter_jailbreak.go)
- [src/semantic-router/pkg/extproc/req_filter_response_api.go](../../../src/semantic-router/pkg/extproc/req_filter_response_api.go)
- [src/semantic-router/pkg/extproc/req_filter_looper_response.go](../../../src/semantic-router/pkg/extproc/req_filter_looper_response.go)
- [src/semantic-router/pkg/extproc/processor_res_usage.go](../../../src/semantic-router/pkg/extproc/processor_res_usage.go)
- [src/semantic-router/pkg/extproc/processor_res_cache.go](../../../src/semantic-router/pkg/extproc/processor_res_cache.go)
- [src/semantic-router/pkg/extproc/processor_res_memory.go](../../../src/semantic-router/pkg/extproc/processor_res_memory.go)
- [src/semantic-router/pkg/extproc/processor_res_usage_test.go](../../../src/semantic-router/pkg/extproc/processor_res_usage_test.go)
- [src/semantic-router/pkg/extproc/processor_res_cache_test.go](../../../src/semantic-router/pkg/extproc/processor_res_cache_test.go)
- [src/semantic-router/pkg/extproc/processor_res_memory_test.go](../../../src/semantic-router/pkg/extproc/processor_res_memory_test.go)
- [src/semantic-router/pkg/extproc/processor_res_body_streaming_test.go](../../../src/semantic-router/pkg/extproc/processor_res_body_streaming_test.go)
- [src/semantic-router/pkg/extproc/AGENTS.md](../../../src/semantic-router/pkg/extproc/AGENTS.md)
- [docs/agent/tech-debt/td-023-extproc-request-pipeline-phase-collapse.md](td-023-extproc-request-pipeline-phase-collapse.md)
- [docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md](td-006-structural-rule-target-vs-legacy-hotspots.md)

## Why It Matters

- New response-time behavior can still require edits across provider normalization, streaming reconstruction, response-side safety warnings, cache/replay, and transport mutation in the same slice.
- Response warnings and replay/caching are harder to test in isolation when they remain tightly coupled to non-streaming response shaping and streaming finalization.
- The extproc local rules mention processor hotspots, but the response path still lacks a subsystem-specific architecture target equivalent to the request path cleanup that closed TD023.

## Desired End State

- The extproc response path is expressed as explicit phases such as provider normalization, streaming accumulation/finalization, non-streaming usage accounting, replay/cache persistence, and response-side warning or mutation shaping.
- Each phase owns one dominant responsibility and exposes a narrow handoff contract instead of accumulating more work inside one response pipeline seam.
- Response-side jailbreak and hallucination warnings stay behind focused helpers instead of forcing unrelated response mutation logic back through the main pipeline.

## Exit Criteria

- New response-side features no longer require reopening both the streaming and non-streaming response seams unless the feature is intentionally cross-cutting.
- Provider normalization, replay/cache persistence, and response warnings have clearer primary owners than the current broad response pipeline.
- Extproc-local `AGENTS.md` explicitly names the response-phase hotspots and the extraction-first rules contributors should follow.

## Retirement Notes

- Usage accounting, token metrics, and cost recording were extracted from `processor_res_body_pipeline.go` and `processor_res_body_streaming.go` into a dedicated `processor_res_usage.go` with 7 new targeted unit tests in `processor_res_usage_test.go`.
- Semantic cache writes, streaming response reconstruction, and cache entry management were extracted into `processor_res_cache.go`; 15 existing cache tests were consolidated from the former `processor_res_body_pipeline_test.go` and `processor_res_body_streaming_test.go` into `processor_res_cache_test.go`.
- Memory store scheduling was extracted into `processor_res_memory.go` with 5 dedicated tests in `processor_res_memory_test.go`, including 3 new edge-case tests (jailbreak skip, global auto-store fallback, both auto-stores disabled).
- `processor_res_body_pipeline.go` shrank from 322 to 132 lines and now only orchestrates the non-streaming response path and Response API translation.
- `processor_res_body_streaming.go` shrank from 376 to 90 lines and now only handles streaming chunk parsing, TTFT, metadata extraction, and finalization.
- Extproc-local `AGENTS.md` now explicitly names the primary response-phase owners so future contributors discover the narrower seams before widening them.
- All existing tests pass without regression across the full `pkg/extproc` package.
