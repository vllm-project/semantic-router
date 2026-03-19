# `pkg/extproc` Local Rules

## Scope

- `src/semantic-router/pkg/extproc/**`
- local rules for extproc processors and router hotspots

## Responsibilities

- Treat `processor_req_body.go`, `processor_res_body.go`, `processor_req_header.go`, `processor_res_header.go`, and `router.go` as orchestration files, not dumping grounds for new helpers.
- Treat `req_filter_classification.go`, `req_filter_memory.go`, `req_filter_response_api.go`, and `req_filter_modality.go` as request-phase hotspots, not as general-purpose homes for every new pre-routing behavior.
- Keep the main processor files aligned with runtime phase seams.
- Keep response normalization, streaming accumulation/finalization, replay/cache persistence, and response-side warning shaping on separate seams instead of letting `processor_res_body*.go` or `res_filter_*.go` absorb everything.
- Keep runtime ownership aligned with the project layers:
  - `signal` extracts facts from request or response content
  - `decision` combines signals with boolean control logic
  - `algorithm` chooses among models after a decision matches
  - `plugin` applies post-decision processing
  - `global` is reserved for intentionally cross-cutting behavior

## Change Rules

- Legacy hotspot size is debt, not precedent.
- New body mutation helpers, routing response builders, memory helpers, or streaming/cache helpers belong in adjacent `processor_*_*.go` files.
- Prefer seams that match runtime phases: request extraction, decision evaluation, routing response construction, streaming handling, replay/caching, response shaping.
- Do not put signal extraction into decision helpers, boolean decision logic into signal extractors, or model-selection heuristics into plugin handlers.
- If a feature needs multiple candidate models after a decision matches, add or extend algorithm-oriented helpers instead of burying the choice inside a signal or plugin branch.
- Do not add new provider-specific or plugin-specific branches directly into the main processor functions when a helper or strategy file can hold that behavior.
- Keep prompt preprocessing, signal evaluation, and matched-signal projection separate from transport translation or memory enrichment.
- Keep memory search gating, query rewriting, and request-body enrichment on narrower seams instead of adding more mixed concerns to `req_filter_memory.go`.
- Keep Response API translation and persistence state at the transport edge; do not mix it with routing-decision or classification logic.
- Keep provider response normalization, streaming TTFT/finalization, router replay/cache writes, and response-side safety warnings on narrower seams instead of rejoining them in one response helper.
- When touching request or response processors, run targeted tests for the affected flow before considering broader package tests. Full `pkg/extproc` runs depend on optional local model artifacts and may fail for environment reasons unrelated to the refactor.
