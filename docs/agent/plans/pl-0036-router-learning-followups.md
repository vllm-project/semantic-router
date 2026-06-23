# Router Learning Adaptation and Protection

## Goal

Implement the current
`website/docs/proposals/router-learning-memory-and-adaptations.md` proposal:
clean-break Router Learning API, adaptation, protection, routing sampling,
outcome ingestion, replay diagnostics, offline recipe-learning hooks, local
validation, AMD validation, and PR update.

## Scope

- Replace old public learning config names instead of compatibility aliasing
  them.
- Expose public runtime concepts as `adaptation` and `protection`.
- Keep request-time routing fail-open and free of synchronous external storage
  reads.
- Use Router Replay as the durable event log.
- Keep online protection state and model experience in process for the first
  implementation.
- Implement `routing_sampling` as the day-0 online adaptation strategy.
- Let decisions bypass or observe learning without hard-coded privacy,
  security, or local-only concepts.
- Validate the implementation locally before AMD deployment validation.

## Exit Criteria

- Router config accepts `global.router.learning.adaptation` and
  `global.router.learning.protection`.
- Router config rejects old public learning config names without rewrite or
  compatibility fill.
- Decision config supports `adaptations.mode`, `adaptations.adaptation.mode`,
  `adaptations.protection.mode`, optional
  `adaptations.protection.stability_weight`, and optional
  `adaptations.protection.switch_margin`.
- Runtime ordering is:
  `base selector -> protection preflight -> adaptation -> protection switch guard -> final model`.
- Adaptation supports `candidate_set: decision`, `tier`, and `global`.
- Adaptation implements `strategy: routing_sampling` with posterior mean,
  guarded sampling, reliability/cost/latency/cache adjustments, and replay
  diagnostics.
- Protection supports `conversation` and `session` scopes, identity headers,
  sampling suppression, switch guard, cache/tool-loop/session protection, and
  deterministic rescue.
- Outcomes can be ingested through `POST /v1/router/outcomes`, linked by
  `replay_id`, and used to update model-targeted online experience.
- Response headers use compact multi-header learning diagnostics.
- Router Replay records typed `learning.protection_preflight`,
  `learning.adaptation`, and `learning.protection` diagnostics.
- Offline recipe-learning hooks can export replay/outcome evidence and report
  useful metrics or findings.
- Local gates from `make agent-report` pass or have concrete environment
  evidence.
- AMD validation passes against the agentic recipe and captures routing,
  protection, adaptation, replay, dashboard, API, and cache evidence.
- The PR title, description, and content match the final implementation.

## Task List

- [x] RLA-001 Update proposal to current clean-break architecture.
- [ ] RLA-002 Replace Go and Python config schema with adaptation/protection.
- [ ] RLA-003 Replace runtime learning contracts with typed adaptation and
  protection pipeline.
- [ ] RLA-004 Implement protection state, preflight, switch guard, and rescue.
- [ ] RLA-005 Implement model experience and `routing_sampling`.
- [ ] RLA-006 Add outcome ingestion API and online experience updates.
- [ ] RLA-007 Update response headers and Router Replay diagnostics.
- [ ] RLA-008 Update recipe, docs, examples, and tests.
- [ ] RLA-009 Run local validation gates and fix failures.
- [ ] RLA-010 Run AMD deployment validation and capture evidence.
- [ ] RLA-011 Update PR title, description, and final content.

## Next Action

Finish the runtime cutover from old method-keyed learning adapters to the
adaptation/protection pipeline. The current worktree has partial config and CLI
schema changes; next edits should make `src/semantic-router/pkg/extproc` compile
against the clean-break API and then add outcome/replay/header coverage.

## Operating Rules

- Do not reintroduce config rewrite or compatibility migration for old learning
  config names.
- Do not mention external reference projects in repo docs or code.
- Do not make request routing depend on synchronous external storage.
- Do not expose broad public `memory.enabled`, `states.enabled`, or
  `experience.enabled` toggles.
- Keep policy/result structures typed; use `map[string]interface{}` only at
  serialization boundaries.
- Keep implementation slices independently testable.
- Record intentional architecture gaps in this plan or indexed tech debt rather
  than leaving them only in chat.

## Related Docs

- `website/docs/proposals/router-learning-memory-and-adaptations.md`
- `docs/agent/change-surfaces.md`
- `docs/agent/module-boundaries.md`
