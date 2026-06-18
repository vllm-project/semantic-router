# Session-Aware Learning

## Overview

Session-aware learning keeps agent conversations stable without making
session continuity a semantic route.

Each request still routes through normal decisions first. A simple task can
propose a simple model, a complex task can propose a stronger model, a privacy
task can propose a local model, and a domain task can propose a domain model.
After that base proposal, session-aware learning decides whether to keep the
current protected model for continuity, cache preservation, or a tool loop.

## Key Advantages

- Preserves model continuity during tool loops and provider-owned continuations.
- Prices prefix-cache loss and model handoff cost before switching.
- Supports `conversation` scope for agent runs and stricter `session` scope.
- Lets privacy or safety decisions bypass adaptation cleanly.

## What Problem Does It Solve?

Agent traffic often spans multiple requests. A later turn can look simple in
isolation while still depending on a previous complex model choice, warm prefix
cache, or tool-call state. Session-aware learning evaluates that stay-vs-switch
trade-off after the base decision proposes a model.

## When to Use

- One agent run should stay stable across tool calls and follow-ups.
- A session can start a new conversation that should be routed again.
- Expensive frontier-model handoffs need a higher bar than cheap local switches.
- Policy decisions need explicit `bypass` boundaries.

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
          identity:
            headers:
              session: x-session-id
              conversation: x-conversation-id
          tuning:
            idle_timeout_seconds: 300
            min_turns_before_switch: 1
            switch_margin: 0.05
            cache_weight: 0.20
            handoff_penalty: 0.05
            handoff_penalty_weight: 1.0
            switch_history_weight: 0.04
            max_cache_cost_multiplier: 2.5
```

`scope: conversation` protects one agent run. A new `x-conversation-id` in the
same `x-session-id` can route again, while cache and handoff cost still inform
the decision. This is the right default for coding agents that create a new run
for each user prompt but need model stability inside the run's tool loop.

`scope: session` is stricter. The session's established model is protected
across conversations until the session idles out or a decision bypasses the
adaptation. Use it when the application wants one session-level model choice to
remain stable across multiple user-initiated runs.

| Scope | What is protected | What can re-route |
| --- | --- | --- |
| `conversation` | Turns sharing the same `x-conversation-id` | A new `x-conversation-id` in the same `x-session-id` |
| `session` | Turns sharing the same `x-session-id` | Idle timeout or a decision with `adaptations.session_aware.mode: bypass` |

If the required identity headers are missing, the adaptation no-ops and the
base decision result remains final.

## Decision Boundaries

Most decisions do not need local configuration. The default mode is `apply`.

Use `bypass` for hard policy boundaries:

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      modelRefs:
        - model: local-private-model
      adaptations:
        session_aware:
          mode: bypass
```

Use `observe` to evaluate the adaptation without changing the final model:

```yaml
adaptations:
  session_aware:
    mode: observe
```

A decision can also override `scope` or a small tuning subset when it needs a
stricter boundary than the global default:

```yaml
adaptations:
  session_aware:
    mode: apply
    scope: session
    tuning:
      switch_margin: 0.10
```

Unset local fields inherit from the global session-aware learning config.
Unknown adaptation names and unsupported `session_aware` fields fail
validation, so typos do not silently change routing behavior.

## Diagnostics

Successful routed responses include a compact summary when learning ran:

```http
x-vsr-learning-methods: session_aware
x-vsr-learning-actions: session_aware=stay
x-vsr-learning-scopes: session_aware=conversation
x-vsr-learning-reasons: session_aware=stay_has_best_adjusted_score
x-vsr-learning-modes: session_aware=apply
```

The headers are designed for lightweight clients and CLI displays. They contain:

| Header | Meaning |
| --- | --- |
| `x-vsr-learning-methods` | Learning methods summarized by this response. |
| `x-vsr-learning-actions` | Method-keyed runtime results, such as `session_aware=switch`. |
| `x-vsr-learning-scopes` | Method-keyed protection scopes, such as `session_aware=conversation`. |
| `x-vsr-learning-reasons` | Method-keyed short reasons for actions. |
| `x-vsr-learning-modes` | Method-keyed decision-level controls, such as `session_aware=apply` or `session_aware=observe`. |

Client UIs should translate the compact action into user-facing routing
language instead of showing the raw enum directly:

| Header value | When it appears | Suggested display |
| --- | --- | --- |
| `x-vsr-learning-actions: session_aware=select` with `x-vsr-learning-reasons: session_aware=missing_previous_model` | First routed request for the active `scope`; there is no previous protected model. | Hide in the main route footer/overview. Show `new run` or `first route for this run` only in debug signals. |
| `x-vsr-learning-actions: session_aware=stay` | Learning kept the current model because continuity, cache, handoff cost, or session policy outweighed the base route change. | `kept run model` or `kept session model`. |
| `x-vsr-learning-actions: session_aware=switch` | Learning allowed a model change after stay-vs-switch scoring. | `switch/run/switch allowed` in compact route columns, `model switched` in detail text. |
| `x-vsr-learning-actions: session_aware=hard_lock` with `x-vsr-learning-reasons: session_aware=hard_lock=tool_loop` | A tool loop is active, so the model is pinned for tool-call stability. | `hard lock/run/tool loop` in compact route columns, `tool-loop pinned` in detail text. |
| `x-vsr-learning-actions: session_aware=hard_lock` with `x-vsr-learning-reasons: session_aware=hard_lock=context_portability...` | Provider-specific state would not be portable across models. | `context pinned`. |
| `x-vsr-learning-actions: session_aware=hard_lock` with `x-vsr-learning-reasons: session_aware=hard_lock=min_turns` | The conversation has not reached the minimum turn count before switching. | `warmup pinned`. |
| `x-vsr-learning-actions: session_aware=bypass` | The decision opted out of learning, for example a privacy boundary. | `learning bypassed`. |
| `x-vsr-learning-actions: session_aware=noop` | Learning was enabled but could not run, usually because identity headers were missing. | Hide in the main UI. Show `missing ids` only in debug signals when the reason is `identity_missing`. |

When `x-vsr-learning-modes` contains `session_aware=observe`, the action is an
audit result and must be displayed as a non-applied prediction, such as `would
switch model` or `would keep run model`. Status labels that are
learning-specific should use neutral text styling; the selected model and
decision can keep the router accent color.

Router Replay stores the full structured trace:

```json
{
  "learning": {
    "adaptations": {
      "session_aware": {
        "mode": "apply",
        "scope": "conversation",
        "action": "stay",
        "reason": "stay_has_best_adjusted_score",
        "base_selected_model": "small-model",
        "selected_model": "frontier-model",
        "cache_warmth": 0.91,
        "memory_prompt_tokens": 3840
      }
    }
  }
}
```

Use the replay record when you need full debugging detail: base proposal,
final model, protected current model, candidate scores, cache warmth, memory
token counters, phase, and hard-lock state. The response header only carries
the bounded summary.

`decision.algorithm.type=session_aware` is removed from the public config
contract. If a route needs an explicit base selector, configure a normal
decision algorithm such as `hybrid`; otherwise omit `algorithm` and let the
router use its default base selection.
