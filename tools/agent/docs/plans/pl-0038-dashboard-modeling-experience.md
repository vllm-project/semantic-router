# PL-0038: Dashboard Modeling Experience

## Goal

Deliver a coherent dashboard experience for connecting models, authoring
Mixture-of-Models, operating the Playground, and understanding router status.

## Scope

- Simplify model connection and preserve advanced provider metadata.
- Make Mixture-of-Models authoring decision-first and repair recipe views.
- Unify navigation, dialogs, banners, loading states, and responsive behavior.
- Keep Playground transport OpenAI-compatible, streaming, and tool-capable.
- Retain dashboard identity and invitation management without inference-access
  policy, API-key, team, or budget management.
- Tune and rename built-in recipes included in this dashboard-focused change.

## Non-Goals

- API-key authentication, access-policy, rate-limit, budget, or inference-team
  management.
- Router protocol codec changes or provider-pricing schema changes.
- A new agent service or agent-specific Router protocol.

## Exit Criteria

- Model connection and Mixture-of-Models creation work end to end.
- Playground streams responses and completes a tool-use loop.
- Dashboard status, invitation, navigation, and loading flows are coherent.
- Repository gates selected by the task matrix pass.
- The deployed dashboard reflects the reviewed source without compatibility
  glue or intermediate designs.

## Task List

- [x] `TASK-01` Define the navigation and shared product surfaces.
- [x] `TASK-02` Rework model connection and Mixture-of-Models authoring.
- [x] `TASK-03` Replace legacy Playground progress UI and preserve metadata.
- [x] `TASK-04` Complete real-environment interaction and regression checks.
- [x] `TASK-05` Audit scope, quality, and repository gates.
- [x] `TASK-06` Prepare the signed reviewable change and CI handoff.

## Next Action

Keep the plan current while review feedback and CI findings are resolved.

## Operating Rules

- Keep the local source tree canonical; use remote systems only for build,
  deployment, and runtime validation.
- Prefer shared components over page-specific visual exceptions.
- Remove superseded implementations rather than maintaining parallel paths.

## Related Docs

- [Mixture-of-Models overview](../../../website/docs/overview/mom-model-family.md)
- [Architecture guardrails](../architecture-guardrails.md)
- [Feature-complete checklist](../feature-complete-checklist.md)
