# Learning Adaptations

## Overview

Learning adaptations run after a semantic decision and its base model selection.
They can use Router Learning states and experience to keep, switch, or observe a
model choice across requests.

The public adaptation namespace is:

```yaml
global:
  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
          enabled: true
        bandit:
          enabled: true
        elo:
          enabled: false
        personalization:
          enabled: false
```

Most decisions do not need local adaptation config. Use
`routing.decisions[].adaptations.<name>` only when a decision needs `apply`,
`observe`, `bypass`, or sparse tuning overrides.

## Key Advantages

- Keeps cross-request learning out of `decision.algorithm`.
- Lets hard policy decisions bypass learning explicitly.
- Gives every adaptation the same method-keyed diagnostics shape.
- Keeps request routing fail-open when states or experience are missing.

## What Problem Does It Solve?

Agents and long-running sessions need stable routing behavior, but base
selection algorithms only score the current request. Learning adaptations add
bounded cross-request context after the base route: active tool loops, warm
prefix cache, explicit feedback, cost tradeoffs, and future personalization.

This keeps the base algorithm simple while making the final route aware of
session continuity and learned evidence.

## When to Use

- A route should stay on the current model during an active conversation or
  session.
- A decision is a privacy or policy boundary and must bypass learning.
- You want to observe what learning would do before allowing it to change
  routes.
- You want conservative bandit behavior for explicit feedback or cost-aware
  exploration.

## Adaptation Types

| Adaptation | Status | Scope | States | Best For |
| --- | --- | --- | --- | --- |
| `session_aware` | Supported | `conversation` or `session` | Current model, tool-loop state, cache evidence, switch history | Agent continuity and prefix-cache protection |
| `bandit` | Day-0 | `decision` by default | Impressions and explicit feedback rewards | Cost/quality exploration with bounded online learning |
| `elo` | Day-0 | `decision` by default | Pairwise feedback ratings | Feedback-driven quality ranking |
| `personalization` | Day-0 | `decision` by default | User preference states | Personalized routing from explicit feedback |

`session_aware` is the first production adaptation. `bandit` has a conservative
day-0 runtime: without reward states, an exploration budget, or explicit
cost/latency goals, it records diagnostics but keeps the base model. `elo` and
`personalization` also use conservative day-0 states: without explicit feedback,
they record diagnostics and keep the base model. None of these should be
configured as `decision.algorithm.type`.

## Bandit Example

```yaml
global:
  router:
    learning:
      enabled: true
      adaptations:
        bandit:
          enabled: true
          algorithm: linucb
          scope: decision
          goals:
            quality: 0.7
            cost: 0.2
            latency: 0.1
          tuning:
            exploration_budget: 0.05
```

Use `goals` as a weighted map. The router normalizes weights internally.
`quality` uses base selector scores or explicit feedback reward states, `cost`
uses configured model pricing, and `latency` is reserved for runtime latency
states.

Explicit feedback submitted to `/api/v1/feedback` updates enabled Router
Learning states. Decision-scoped adaptations use `decision_name`; session-scoped
adaptations require `session_id`; conversation-scoped adaptations require both
`session_id` and `conversation_id`. `personalization` additionally requires
`user_id`.

## Configuration

```yaml
routing:
  decisions:
    - name: privacy_boundary
      adaptations:
        session_aware:
          mode: bypass
        bandit:
          mode: bypass
        elo:
          mode: observe
        personalization:
          mode: bypass
```

Use `bypass` when a decision is a hard policy boundary. Use `observe` when an
adaptation should emit headers and replay diagnostics without changing the final
model.

## Diagnostics

Response headers stay compact and method-keyed:

```http
x-vsr-learning-methods: bandit,session_aware
x-vsr-learning-actions: bandit=switch,session_aware=stay
x-vsr-learning-scopes: bandit=decision,session_aware=conversation
x-vsr-learning-reasons: bandit=score_win,session_aware=cache_hot
x-vsr-learning-modes: bandit=apply,session_aware=apply
```

Full score details, state-key hashes, experience status, and identity
diagnostics belong in Router Replay under `learning.adaptations`.
