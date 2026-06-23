# Protection

## Overview

Protection keeps agent conversations stable without making continuity a
semantic route. Each request still routes through normal decisions first. After
adaptation proposes a model, protection decides whether to hold the current
model, allow the switch, or perform a bounded rescue switch.

## Key Advantages

- Keeps model choices stable inside agent conversations or whole sessions.
- Protects prefix cache, tool-loop continuity, and handoff cost.
- Suppresses random exploration during protocol-sensitive steps.
- Still permits deterministic switches and bounded rescue when evidence is
  strong enough.
- Lets sensitive decisions bypass protection through decision-local controls.

## What Problem Does It Solve?

Agent requests are not independent. Tool calls, provider state, prefix cache,
and user-visible continuity can make unnecessary model switches expensive or
confusing. Protection gives the router a scoped stability guard without turning
session continuity into a semantic decision rule.

## When to Use

- A conversation should keep using the same model unless a switch is worth the
  stability cost.
- A full session should remain stable across multiple user-initiated runs.
- Tool-loop or protocol state makes random exploration unsafe.
- A weaker protected model should still be escapable through bounded rescue.

## Configuration

```yaml
global:
  router:
    learning:
      enabled: true
      protection:
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
          stability_weight: 1.0
```

## Scopes

| Scope | What is protected | What can re-route |
| --- | --- | --- |
| `conversation` | Turns sharing one `x-conversation-id`. | A new `x-conversation-id` in the same `x-session-id`. |
| `session` | Turns sharing one `x-session-id`. | Idle timeout or a decision with `adaptations.mode: bypass`. |

Use `conversation` when each agent run should be independently routed. Use
`session` when one session-level model choice should remain stable across
multiple user-initiated runs.

If the configured identity headers are missing, protection fails open and
records diagnostics instead of failing the request.

## Guards

Protection has two guard points:

- **preflight** suppresses stochastic sampling during tool/protocol/routine
  continuation steps.
- **switch guard** accepts or rejects adaptation's proposed model using cache,
  handoff, tool-loop, session, and switch-history cost.

The switch rule is:

```text
switch if proposal_gain >= switch_margin + stability_weight * switch_cost
```

Protection can also allow a deterministic `rescue_switch` when the current
model appears underpowered because of repeated failures, retries, failed
verification, or explicit outcome evidence.

## Decision Boundaries

Most decisions do not need local configuration. Use `bypass` for hard policy
boundaries:

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      modelRefs:
        - model: local-private-model
      adaptations:
        mode: bypass
```

Use `observe` to collect diagnostics without changing the final model:

```yaml
adaptations:
  protection:
    mode: observe
```

## Diagnostics

```http
x-vsr-learning-methods: protection
x-vsr-learning-actions: protection=hold_current
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: protection=cache_cost_high
```

Client UIs should translate raw actions into user-facing text. For example,
`hold_current` can display as "kept run model", `allow_switch` as "switch
allowed", `rescue_switch` as "rescue switch", and `bypass` as "learning
bypassed".

Router Replay stores the full protection trace: identity source and hash,
protected model, base model, proposal model, final model, switch cost, cache
evidence, tool-loop state, mode, scope, action, and reason.
