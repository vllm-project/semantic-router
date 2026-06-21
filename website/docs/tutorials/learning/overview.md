# Router Learning

## Overview

Router Learning is the router layer for cross-request routing intelligence. It
adjusts the model proposed by semantic decisions without making learning state
part of `decision.algorithm`.

The first production adaptation is session-aware learning. The same namespace
also contains day-0 and roadmap learning adaptations:

- `global.router.learning.adaptations.session_aware` enables the adaptation.
- `global.router.learning.adaptations.bandit` enables conservative
  feedback/cost-aware online scoring.
- `global.router.learning.adaptations.elo` enables day-0 pairwise rating
  states.
- `global.router.learning.adaptations.personalization` enables day-0 user
  preference states.
- `routing.decisions[].adaptations.session_aware.mode` controls hard decision
  boundaries.
- `x-session-id` identifies the long-lived client session.
- `x-conversation-id` identifies one agent run or conversation.
- `x-vsr-learning-*` summarizes bounded adaptation results in response headers.
- Router Replay stores the full trace under `learning.adaptations`.

Use Router Learning when a decision should remain semantic, but repeated
requests should consider router memory such as current model, tool-loop state,
prefix-cache evidence, handoff cost, or switch history.

See [Learning Adaptations](./adaptations) for the adaptation map and
[Session-Aware Learning](./session-aware) for the production continuity
configuration.

## Key Advantages

- Keeps semantic decisions readable and request-local.
- Gives cross-request adaptations one shared global namespace.
- Records bounded diagnostics in response headers and Router Replay.
- Lets hard policy decisions bypass adaptation without changing route rules.

## What Problem Does It Solve?

Routing can need memory that is not part of the current prompt: current model,
conversation identity, cache warmth, provider continuation state, or previous
switches. Router Learning owns that cross-request evidence and applies it after
the normal decision engine proposes a model.

## When to Use

- Agent sessions should avoid unnecessary model switches.
- Prefix-cache or provider-state continuity affects routing quality or cost.
- You want to evaluate an adaptation in `observe` mode before enforcing it.
- A recipe needs a shared place for future online routing intelligence.

## Configuration

```yaml
global:
  router:
    learning:
      enabled: true
      adaptations:
        session_aware:
          enabled: true
          scope: conversation
        bandit:
          enabled: false
          scope: decision
        elo:
          enabled: false
        personalization:
          enabled: false
```

Decisions only need local config when they opt out, observe, or need a sparse
scope/tuning override:

```yaml
adaptations:
  session_aware:
    mode: bypass
  bandit:
    mode: observe
  elo:
    mode: observe
  personalization:
    mode: bypass
```

## Protection Scopes

`session_aware.scope` chooses how strongly the router protects an established
model:

| Scope | Identity key | Behavior |
| --- | --- | --- |
| `conversation` | `x-conversation-id` | Protects one agent run. A new conversation in the same session can route again. |
| `session` | `x-session-id` | Protects the session model across conversations until the session idles out or a decision bypasses adaptation. |

Use `conversation` when each agent run should be independently routed. Use
`session` when the application wants one model choice to remain stable across
multiple runs inside the same client session.

## Header And Replay

The `x-vsr-learning-*` header family is intentionally compact:

```http
x-vsr-learning-methods: session_aware
x-vsr-learning-actions: session_aware=hard_lock
x-vsr-learning-scopes: session_aware=conversation
x-vsr-learning-reasons: session_aware=hard_lock=tool_loop
x-vsr-learning-modes: session_aware=apply
```

It tells a client which learning methods ran, what action each method took, why,
and which scope was active. Detailed fields such as base selected model, final
selected model, cache warmth, memory token counts, and candidate score traces
belong in Router Replay, keyed by `x-vsr-replay-id`.
