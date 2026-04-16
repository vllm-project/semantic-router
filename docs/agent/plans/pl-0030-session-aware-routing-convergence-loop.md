# Session-Aware Routing Convergence Loop

## Goal

- Retire the legacy `routing.session_states` / `SESSION_STATE` public surface instead of extending it further.
- Converge the repository on one production-oriented session-aware routing contract built around runtime-derived session facts, session-aware signals, and explicit selection policy wiring.
- Close the workstream only after config, runtime, replay/session identity, docs, and targeted validation all agree on the same steady-state behavior.

## Scope

- `docs/agent/plans/**`
- `src/semantic-router/pkg/config/**`
- `src/semantic-router/pkg/dsl/**`
- `src/semantic-router/pkg/extproc/**`
- `src/semantic-router/pkg/selection/**`
- `src/semantic-router/pkg/selection/lookuptable/**`
- `src/semantic-router/pkg/routerreplay/**`
- maintained config / recipe assets that exercise routing contracts
- targeted docs and validation paths for the touched surfaces
- nearest local rules for `pkg/config` and `pkg/extproc`

## Exit Criteria

- The repository no longer exposes `routing.session_states` or `SESSION_STATE` as supported steady-state config or DSL surface.
- Runtime-derived session facts are the only durable source of session-aware inputs consumed by routing logic.
- The config contract exposes one explicit session-aware control surface for decision-time routing, instead of spreading behavior across legacy schema leftovers and hidden selector heuristics.
- Session-aware routing can evaluate stay-versus-switch behavior using replay-backed lookup-table signals plus real session identity, not pseudo-session heuristics alone.
- Targeted docs, maintained assets, and changed-set validation reflect the steady-state contract and no longer teach the removed legacy surface.

## Task List

- [x] `SAR001` Create and index the durable execution plan for the session-aware routing workstream.
- [x] `SAR002` Remove the legacy `routing.session_states` / `SESSION_STATE` public surface from config, canonical export/import, DSL, maintained assets, and tests.
- [ ] `SAR003` Define the steady-state session-aware config contract around runtime-derived session facts, decision-time control knobs, and supported algorithm surfaces.
- [ ] `SAR004` Add the missing session-aware signal family and validator/catalog wiring so decisions can reference session / lookup signals explicitly.
- [ ] `SAR005` Implement the session-aware selector path that evaluates stay-versus-switch behavior using turn index, previous model, lookup-table priors, and fallback defaults.
- [ ] `SAR006` Replace pseudo-session replay grouping with a real session identity contract across replay ingestion, lookup-table derivation, and response-side telemetry.
- [ ] `SAR007` Add the production gates still missing for retention, shadow / audit output, and operator-visible debugging of session-aware decisions.
- [ ] `SAR008` Update maintained docs, examples, and targeted validation so the new session-aware contract is the only documented public path.
- [ ] `SAR009` Run the validation ladder for the changed surfaces, record results here, and add indexed debt only for gaps that remain after the loop closes.

## Current Loop

- Loop status: opened on 2026-04-15.
- Completed in this loop:
  - confirmed the repository-native harness and plan requirements before editing
  - removed `routing.session_states` from the config schema, canonical routing surface, DSL grammar/compiler/decompiler/validator chain, maintained recipe assets, and affected tests
  - created this execution plan so the remaining multi-loop session-aware routing work is resumable from the repository alone
- Next loop focus:
  - execute `SAR003` by defining the steady-state session-aware public contract without reintroducing legacy schema compatibility into runtime parsing

## Decision Log

- 2026-04-15: remove `routing.session_states` directly instead of keeping a compatibility bridge in the steady-state runtime contract.
- 2026-04-15: treat runtime-derived session facts plus explicit session-aware routing controls as the only forward path for multi-turn routing.
- 2026-04-15: keep lookup tables as reusable substrate, but do not treat the current pseudo-session replay heuristic as production-complete session identity.
- 2026-04-15: use one execution plan because the remaining work spans config, DSL, extproc, selection, replay, maintained assets, and validation loops.

## Follow-up Debt / ADR Links

- `issue #1753` stable multi-turn session-aware routing goal
- `docs/agent/lookup-tables.md`
- Add an ADR or debt entry only if the steady-state session-aware contract still diverges after this loop series completes.
