# TD055: Evidence-Calibrated Session Switch Gate Gap

## Status

Open.

## Owner Plan

[PL-0040: MoM Routing Hardening](../plans/pl-0040-mom-routing-hardening.md)

## Release Relevance

Session-aware selection protects active tool loops and non-portable context and
can reset stale state after decision drift. It does not yet have a calibrated
recent-window gate that distinguishes model-attributable lack of progress from
tool, provider, or infrastructure failures before switching models.

## Scope

- `src/semantic-router/pkg/selection/session_aware.go`
- `src/semantic-router/pkg/selection/session_aware_scoring.go`
- `src/semantic-router/pkg/sessiontelemetry/`
- `src/semantic-router/pkg/routerreplay/`
- `src/semantic-router/pkg/extproc/router_learning_*`

## Summary

The next session-aware increment needs typed trajectory evidence rather than a
larger collection of keyword weights. A bounded recent window should capture
correction, repeated no-progress output, verified recovery, active mutations,
and failure provenance. A switch gate can then require corroborated,
model-attributable evidence while preserving hard continuity locks.

## Evidence

- Tail-aware `active_tool_loop` prevents historical tool activity from pinning
  unrelated later turns.
- Decision drift resets stale session state, and protection cannot reintroduce
  a candidate outside the current decision.
- Router Replay and Looper expose useful trajectory fields, but the session
  scorer does not consume a normalized recent-window progress contract.
- Provider errors, tool errors, and semantic answer failures remain confounded
  if treated as one negative reward.
- The [agent routing protection baseline](../../../../website/docs/benchmarking/agent-routing-protection.md)
  adds deterministic production-protection contracts with maintained session
  fixtures. Its explicit missing-coverage list still includes rescue/failure
  attribution, evidence-calibrated switching and measured task benefit; this
  baseline does not close TD055 or issue #2338.

## Why It Matters

Switching too eagerly increases cost, cache loss, and latency; switching too
late repeats an unproductive model. Misattributing tool or infrastructure
failures to model quality also teaches the learner the wrong preference.

## Desired End State

Session-aware selection makes a switch only when hard eligibility permits it
and a bounded, privacy-safe trajectory gate has enough model-attributable
evidence. Successful progress can de-escalate after continuity constraints are
cleared. Learning consumes the same failure provenance and never bypasses the
gate or the current decision boundary.

## Exit Criteria

- Define typed recent-window progress and failure-provenance facts without
  retaining private request content.
- Preserve active tool-loop, workflow-state, and non-portable-context locks.
- Require calibrated corroboration or confirmation before escalation and
  de-escalation.
- Keep provider, capacity, tool, policy, and semantic failures distinct in
  replay and learning attribution.
- Demonstrate higher recovery rate without a regression in unnecessary-switch
  rate, latency, or cost on a held-out multi-turn evaluation set.
