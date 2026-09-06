# PL-0040: MoM Routing Hardening

## Goal

Make every built-in Mixture-of-Models recipe express request intent,
capability constraints, routing policy, and session behavior through explicit
signals and projections, with deterministic selection and evidence-backed
defaults.

## Scope

- Review the balance, speed, cost, accuracy, and vault recipes as one coherent
  built-in contract.
- Separate request facts and semantic observations from policy-bearing
  projections and final decisions.
- Strengthen tool-use intent, capability eligibility, quality and latency
  objectives, session continuity, switch gating, and bounded learning where
  the existing runtime can enforce them safely.
- Keep canonical recipes, packaged CLI assets, dashboard materialization,
  documentation, and conformance probes synchronized.
- Validate deterministic routing locally and through the supported AMD
  workflow, with private receipts kept outside tracked source.

## Non-Goals

- Adding a second routing runtime or an opaque model-selection service.
- Treating declared tools as proof that the current turn must execute one.
- Making infrastructure failures or transient latency alone trigger semantic
  model escalation.
- Enabling cross-session learning by default or allowing learning to bypass
  capability and policy constraints.
- Adding tunable dimensions that do not change a tested routing outcome.

## Exit Criteria

- Every added or changed signal has a documented semantic contract and is
  consumed by a projection or decision with a concrete purpose.
- Hard request constraints are enforced before ranking, while soft objectives
  remain decision-local algorithm inputs.
- Session-aware switching preserves continuity without pinning stale or
  ineligible models.
- Built-in static and live conformance probes cover positive, negative,
  collision, multi-turn, and boundary cases for the changed behavior.
- Repository-selected validation, CLI integration, dashboard checks, and
  required CI domains pass.
- AMD validation produces reproducible ignored evidence without placing
  private infrastructure information in tracked artifacts.

## Task List

- [x] `TASK-01` Complete the signal, projection, decision, algorithm, session,
  and learning gap matrix for all built-in recipes.
- [x] `TASK-02` Implement only the gaps with a measurable routing or safety
  benefit and update their semantic contracts.
- [x] `TASK-03` Synchronize canonical recipes, packaged assets, dashboard
  materialization, documentation, and conformance probes.
- [x] `TASK-04` Run the selected local and live gates, repair regressions, and
  record private AMD validation evidence.
- [x] `TASK-05` Reconcile remaining architecture gaps into durable debt or
  explicitly reject them as unnecessary complexity.
- [x] `TASK-06` Re-audit built-in candidate-pool cardinality, Looper quorum,
  generated-stage context growth, and Blend terminal-context ownership.
- [x] `TASK-07` Enforce minimum candidate pools, fail impossible quorums,
  re-check Looper stage context, and add the Blend terminal-context lane.
- [x] `TASK-08` Synchronize packaged assets and pass all repository-selected
  validation, dashboard, CLI, Router, and conformance gates.
- [ ] `TASK-09` Revalidate the final commit on both requested AMD nodes, push
  the PR update, and wait for every required GitHub check.

## Next Action

Create the signed commit from the validated snapshot, synchronize its final
source receipt on both requested AMD nodes, push the PR update, and wait for
every required GitHub check before completing `TASK-09`.

## Operating Rules

- Express raw request or conversation observations as signals and combine
  them into policy-facing projections; do not hide policy in extractors.
- Apply hard capability and safety constraints before any cost, latency,
  quality, or load ranking.
- Keep learning bounded by the selected decision and its eligible model set.
- Require negative and collision probes for heuristic semantic signals.
- Run the smallest applicable gate first and repair failures before expanding.
- Keep private prompts, endpoints, host identities, logs, and validation
  receipts in ignored task-specific directories.

## Related Docs

- [Architecture guardrails](../architecture-guardrails.md)
- [Change surfaces](../change-surfaces.md)
- [Feature-complete checklist](../feature-complete-checklist.md)
- [Testing strategy](../testing-strategy.md)
- [Typed request capability eligibility](../tech-debt/td-054-typed-request-capability-eligibility-gap.md)
- [Evidence-calibrated session switch gate](../tech-debt/td-055-evidence-calibrated-session-switch-gate-gap.md)
