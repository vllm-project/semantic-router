---
title: Model Execution Fallback
description: Defines the ownership and safety boundary for future cross-model fallback after an upstream execution failure.
created: 2026-08-10
status: Proposal
---

> **Status:** Proposal · **Created:** 2026-08-10

## Problem

Endpoint retries and passive outlier ejection can choose another replica of the same
logical model. They cannot safely switch to another logical model because provider
translation, credentials, context limits, cost policy, output format, and session
continuity may change.

## Proposal

Introduce an execution-orchestration boundary after model selection. It owns bounded
cross-model attempts without rematching the semantic decision.

Each attempt records:

- logical model and physical endpoint;
- retry-safe failure class;
- whether response bytes have been committed;
- provider request identifier;
- observed token or cost usage;
- session and conversation identity; and
- remaining attempts and previously visited models.

Fallback candidates must be declared compatible by the matched route. The orchestrator
cannot widen the decision's model pool.

## Ownership

| Layer | Responsibility |
| --- | --- |
| Signals and decisions | Select the intended route and candidate policy. |
| Selection and Router Learning | Choose the initial logical model. |
| Data-plane reliability | Retry or eject replicas within that model's backend. |
| Execution orchestrator | Decide whether a failed logical-model attempt may move to a compatible fallback. |

Cross-model fallback is not another Decision Engine pass.

## Safety rules

- Do not switch models after response bytes are committed.
- Authentication and malformed-request failures do not retry by default.
- Context-window or policy failures may fall back only to an explicitly compatible
  model.
- Streaming requires a pre-commit boundary and a declared partial-output policy.
- Each attempt gets its own timeout while the whole chain has one deadline and cost
  budget.
- Replay and billing must identify every attempted model.
- Session and tool-loop continuity guards apply before a switch.

## Scope and non-goals

This proposal covers logical-model changes after upstream execution failure. It does
not replace same-model transport retry, circuit breaking, outlier ejection, or
ordinary semantic selection.

| Dimension | Gateway Transport Retry | Router Model Fallback |
| :--- | :--- | :--- |
| **Layer** | Envoy / network reverse-proxy | Semantic Router Data Plane |
| **Scope** | Same logical model, alternative replica endpoints | Different logical model from hard-eligible candidates |
| **Request Mutation** | None (identical wire bytes retransmitted) | Re-encoded from neutral request through candidate backend codec |
| **Credentials & Auth** | Shared across replicas | Model- and provider-scoped credentials re-resolved |
| **Context & Capabilities** | Unchanged | Target model context window and wire capabilities re-validated |
| **Decision Engine** | Completely bypassed | Uses initial decision candidate ordering; never re-classifies |

## Resolved decisions

- **Where are compatibility sets declared?** In the decision's `modelRefs`. The Decision Engine validates context windows and required capabilities to construct `VSREligibleModelRefs`.
- **Which provider failures are safe to replay?** Connection drops, connect/read timeouts, and allowlisted 5xx (`502`, `503`, `504`) before response commit. 4xx errors and context cancellations are non-retryable.
- **How is usage reconciled when a failed attempt consumed tokens?** An `ExecutionRecord` tracks all attempts. Final client usage reflects only the successful model; discarded tokens are recorded separately for auditing without duplicate billing.
- **Which streaming protocols expose a reliable commit point?** SSE and chunked HTTP commit at the first byte chunk or header dispatch (`ResponseWriter.Flush()`). Fallback is only allowed prior to this point.
- **How does protection state record a fallback-driven switch?** The fallback orchestrator records attempt outcomes and emits audit diagnostics capturing the failed model, reason, and selected fallback candidate.

## References

- [Routing-scope decision](./batch-and-capacity-aware-routing)
- [Router Learning](./router-learning-memory-and-adaptations)
- [Canonical configuration guide](../installation/configuration)
