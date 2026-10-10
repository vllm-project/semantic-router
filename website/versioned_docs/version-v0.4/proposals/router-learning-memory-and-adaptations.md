---
title: "Router Learning: Self-Improving Model Routing"
description: Records the implemented online adaptation, route protection, and offline recipe-learning contract.
created: 2026-06-20
status: Implemented
---

> **Status:** Implemented · **Created:** 2026-06-20

## Problem

A semantic decision evaluates the current request. It does not, by itself, remember
whether a model was a poor fit in similar runs or whether switching models would break
conversation, tool-loop, or prefix-cache continuity.

Putting that state inside decision rules would make policy opaque and replica-dependent.
Router Learning therefore runs after the matched decision and base selector. The
recipe remains the policy boundary.

## Implemented design

Router Learning has three responsibilities:

| Component | Purpose | Time scale |
| --- | --- | --- |
| Adaptation | Propose a model from bounded runtime experience. | Request path. |
| Protection | Decide whether exploration or a model switch is safe. | Request path. |
| Recipe learning | Analyze replay and outcomes and propose reviewable recipe changes. | Offline. |

```text
matched decision and base selector
  -> protection preflight
  -> adaptation proposal
  -> protection switch guard
  -> final model
  -> replay and outcome updates
```

Adaptation may propose a different model. Protection has the final say on whether that
proposal becomes the selected model.

## Public configuration boundary

| Surface | Meaning |
| --- | --- |
| `global.router.learning.enabled` | Enables the Router Learning pipeline. |
| `global.router.learning.adaptation` | Selects online model-choice behavior. |
| `global.router.learning.protection` | Configures conversation or session stability. |
| `global.router.learning.state_store` | Optionally shares protection state across replicas. |
| `routing.decisions[].adaptations` | Applies, observes, or bypasses learning for one decision. |

The implemented adaptation strategy is `routing_sampling`. Historical algorithm names
are not aliases for this surface. Decisions remain semantic and keep their existing
selection algorithm.

## Candidate sets

Adaptation searches only the configured candidate set:

| Value | Candidate models |
| --- | --- |
| `decision` | Models in the matched decision's `modelRefs`. |
| `tier` | Models from decisions in the matched decision tier. |
| `global` | Models in the deployed recipe inventory. |

`decision` is the narrow default. Broader scopes still obey provider availability,
cost and reliability guards, and decision-level bypass.

## Protection

Protection keeps a model stable within either a conversation or session identity. A
preflight guard suppresses unsafe stochastic exploration during protocol-sensitive
steps. A switch guard weighs the proposed gain against cache, handoff, tool-loop, and
switch-history costs.

If required identity headers are absent, protection fails open and records
diagnostics. A sensitive decision can set `adaptations.mode: bypass`, which prevents
both adaptation and protection from changing the base selection. `observe` computes
diagnostics without changing the final model.

## Experience and outcomes

Experience is evidence, not policy. It may include explicit outcome labels, failure
signals, latency, effective cost, cache reuse, and reliability observations. The
strategy uses that bounded evidence to score or sample candidates.

Outcomes must attach to a stable replay identifier and record the base, proposed, and
final model. Delayed or duplicate outcomes need idempotent handling. Raw request
content is not required for the online experience key.

## State and failure behavior

Protection state can use a bounded local store and an optional shared Redis store.
Request-path reads use a strict timeout. A remote-store failure must not make the
inference request depend on an unbounded network call.

Detailed candidate scores, identity hashes, switch costs, and evidence belong in
Router Replay. Response headers stay compact and describe only the methods, actions,
scopes, and reason codes needed for request-level inspection.

## Offline recipe learning

The offline loop consumes replay, outcomes, and optional evaluation cases. It produces
findings, metrics, candidate variants, patch suggestions, and optional seed artifacts.
It does not edit or deploy the active recipe automatically.

This separation makes recipe changes reviewable and lets operators reproduce an
experiment before promotion.

## Scope and non-goals

Router Learning does not:

- rematch the semantic decision;
- override a decision-level bypass;
- expand beyond the configured candidate set;
- synchronously rewrite deployed recipes;
- make the online request path depend on an LLM agent; or
- treat session affinity as a replacement for authorization.

## Evaluation

Evaluate adaptation and protection independently before evaluating the combined
pipeline. Report route quality, switch rate, regret or fit outcomes, latency, cache
effects, failure recovery, and stability within declared identities. Compare
`apply` against `observe` on the same replay sample before enabling model changes.

## Open questions

- When is a broader `tier` or `global` candidate set worth the added risk?
- Which outcome sources are reliable enough to update online experience?
- How should stale experience decay across model or prompt-template versions?
- When should offline seed artifacts be imported into a live deployment?

## References

- [Router Learning overview](../tutorials/learning/overview)
- [Adaptation](../tutorials/learning/adaptations)
- [Protection](../tutorials/learning/protection)
- [Decision-level controls](../tutorials/learning/decision-adaptations)
- [Memory and replay](../tutorials/learning/memory-and-replay)
