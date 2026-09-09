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
multiple user-initiated runs. Both scopes reset continuity when the matched
decision changes, and neither scope can retain a previous model outside the
current adaptation candidate set.

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

## Progress Gate

The switch rule above is a one-shot cost comparison. The progress gate adds a
second, evidence-based approval stage: a proposed switch is only committed when
the session's recent trajectory justifies it. This prevents thrashing from
single noisy turns (one empty reply, one provider hiccup).

### Configuration

```yaml
global:
  router:
    learning:
      protection:
        tuning:
          progress_gate:
            enabled: true        # default false — zero behavior change
            mode: observe        # observe | enforce
            window_size: 8               # recent turns kept as evidence
            window_ttl_seconds: 900      # evidence older than this expires
            min_window_outcomes: 3       # attributable turns required
            min_consecutive_regressions: 2  # escalation threshold
            min_consecutive_recoveries: 2   # downgrade threshold
            cooldown_seconds: 120        # min quiet period between switches
            max_switches_per_window: 2   # oscillation guard
```

Omitting the whole `progress_gate` section keeps the gate disabled. Any tuning
field can be set alone; the rest inherit their packaged defaults.

### How it decides

Every turn's outcome is classified into a typed, content-minimal fact:
`progress`, `no_progress`, `regression` (model-attributable), or
`provider_error` / `tool_error` / `missing` (environment noise, never counted
against the model). The gate keeps a bounded window of these facts per session
and derives regression/recovery streaks, a progress trend, and evidence
coverage.

A switch proposal is suppressed, in priority order, when:

| Reason | Condition |
| --- | --- |
| `hard_constraint_conflict` | Tool loop or non-portable context — hard locks win over evidence |
| `cold_start` | No observable outcomes yet |
| `insufficient_evidence` | Fewer attributable outcomes than `min_window_outcomes`, or the streak/trend thresholds are not met |
| `cooldown` | Last switch was less than `cooldown_seconds` ago |
| `oscillation_guard` | Already switched `max_switches_per_window` times in the window |

Escalation (proposing a stronger model) requires the regression streak plus a
non-positive trend; downgrading uses consecutive recoveries instead. Hard
constraints stay authoritative in both modes: the gate can only suppress a
switch, never force one.

### Modes

`observe` evaluates the gate and records the full verdict in Router Replay but
never changes the outcome — use it to measure the would-suppress rate before
trusting `enforce`. `enforce` holds the current model when the verdict says
suppress, but only after re-checking that the current model is still a valid
candidate for the request; a suppression never invents a routing target.

Both directions are gated: after switching to a stronger model, dropping back
to a cheaper one also needs consecutive recovery evidence.

### Replay

Every gated switch or suppression records a `switch_gate` section in Router
Replay: evidence version, mode, decision, suppression reason, switch origin
(escalation/downgrade), regression/recovery streaks, trend, window size,
attributable and missing counts, cooldown timers, and the oscillation counter.
Replay therefore explains every model change without access to the router's
memory.

## Decision Boundaries

Most decisions do not need local configuration. Use `bypass` for hard policy
boundaries:

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      description: Keep privacy-sensitive traffic on the local model.
      priority: 200
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
