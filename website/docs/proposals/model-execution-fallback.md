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

## Open questions

- Where are compatibility sets declared?
- Which provider failures are safe to replay?
- How is usage reconciled when a failed attempt consumed tokens?
- Which streaming protocols expose a reliable commit point?
- How does protection state record a fallback-driven switch?

## References

- [Routing-scope decision](./batch-and-capacity-aware-routing)
- [Router Learning](./router-learning-memory-and-adaptations)
- [Canonical configuration guide](../installation/configuration)
