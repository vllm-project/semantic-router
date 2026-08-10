# Model Execution Fallback

## Status

Design boundary for a future execution orchestration layer.

## Problem

Endpoint retries and outlier ejection can recover another replica of the same
logical model. They cannot safely switch to another logical model because the
request body, provider translation, credentials, cost policy, and session
continuity may all change.

## Ownership

- Signal and decision layers select the intended logical route.
- Selection and Router Learning select the initial model.
- Envoy retries transport failures inside that model's backend cluster.
- A future execution orchestrator owns cross-model fallback after an upstream
  attempt fails.

Cross-model fallback must not be implemented as another DecisionEngine pass.

## Proposed Contract

Each attempt records:

- selected logical model and physical endpoint
- retry-safe failure class
- request streaming state
- provider request identifier
- cost reservation and observed usage
- session/conversation identity
- maximum attempts and visited models

Fallback is allowed only before response bytes are committed. Context-window
and policy failures may select an explicitly declared compatible fallback.
Authentication errors, invalid requests, and mid-stream failures do not retry
by default.

## Required Safety Gates

- idempotency key propagated across attempts
- bounded attempt count and total timeout
- no repeated model in one chain
- compatibility check for tools, response format, modality, and context window
- Router Replay record for every attempt and final outcome
- outcome evidence returned to Router Learning

## Delivery Order

1. Ship endpoint-level retry, circuit breaker, and passive outlier ejection.
2. Add typed failure classification and replay-only diagnostics.
3. Implement non-streaming fallback for connect/reset failures.
4. Add explicit context-window and policy fallback mappings.
5. Evaluate streaming fallback only after an API contract is defined.
