# Router Learning Session-Aware Migration

## Goal

Implement the first Router Learning adaptation described in
`website/docs/proposals/router-learning-memory-and-adaptations.md`: move
session-aware routing from decision-local `algorithm.type: session_aware` into
global `router.learning.adaptations.session_aware`, expose generic diagnostics,
update the agentic AMD recipe/eval surface, and verify the behavior locally and
on the AMD deployment.

## Scope

- Add the clean Router Learning config surface for the first implementation.
- Reject legacy session-aware algorithm configuration with actionable errors.
- Keep first-version online state single-replica and in-process.
- Keep request routing free of required external storage reads.
- Reuse Router Replay as the event log and avoid depending on replay payloads.
- Update the agentic routing recipe and tests to use the new API.
- Validate semantic routing, stability, bypass, cache evidence, latency, and
  replay explainability.

## Exit Criteria

- Config loading accepts `global.router.learning.adaptations.session_aware`.
- Config loading rejects `algorithm.type: session_aware` and legacy
  `global.router.model_selection.session_aware` / `model_switch_gate` shapes.
- Session-aware behavior applies after base decision selection across decisions.
- Decisions can opt out with `adaptations.session_aware.mode: bypass`.
- Missing session/conversation identity does not fail requests.
- Responses expose generic learning diagnostics and replay records include
  `learning.adaptations.session_aware`.
- The AMD agentic recipe no longer uses the synthetic session-aware decision.
- Required local gates and AMD regression pass or have documented environment
  blockers with concrete evidence.

## Task List

- [x] RL-001 Derive requirements from proposal and current runtime code.
- [x] RL-002 Implement Router Learning config and validation.
- [x] RL-003 Implement session-aware adaptation runtime.
- [x] RL-004 Add generic learning diagnostics, response header, and replay data.
- [x] RL-005 Update agentic AMD recipe and eval coverage.
- [x] RL-006 Run local gates.
- [ ] RL-007 Run AMD deployment regression.

## Next Action

Run the AMD deployment regression with the agentic SAARS recipe and capture
routing, learning, replay, and dashboard evidence. Local Docker-based
integration/build gates are deferred to the AMD host because the local Docker
daemon is unavailable.

## Operating Rules

- Do not add a config rewrite command.
- Do not add multi-replica online-state support in the first implementation.
- Do not make request routing depend on synchronous external storage reads.
- Do not compatibility-fill older session-aware-specific replay fields for the
  new diagnostics object.

## Related Docs

- `website/docs/proposals/router-learning-memory-and-adaptations.md`
- `docs/agent/module-boundaries.md`
- `docs/agent/change-surfaces.md`
