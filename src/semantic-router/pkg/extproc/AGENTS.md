# `pkg/extproc` Local Rules

## Scope

- `src/semantic-router/pkg/extproc/**`
- local rules for extproc processors and router hotspots

## Responsibilities

- Treat `processor_req_body.go`, `processor_res_body.go`, `processor_req_header.go`, `processor_res_header.go`, and `router.go` as orchestration files, not dumping grounds for new helpers.
- Keep the main processor files aligned with runtime phase seams.
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
- When touching request or response processors, run targeted tests for the affected flow before considering broader package tests. Full `pkg/extproc` runs depend on optional local model artifacts and may fail for environment reasons unrelated to the refactor.
