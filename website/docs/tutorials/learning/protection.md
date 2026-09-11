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

The progress gate evaluates recent outcomes before accepting a model switch
proposed by protection or rescue. It does not select a model, call a model or
tool, or replace the request's authorization and routing policies.

### Configuration

```yaml
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        scope: conversation
        identity:
          headers: {session: x-session-id, conversation: x-conversation-id}
        tuning:
          progress_gate:
            enabled: true
            mode: observe
            calibration_id: ""
            window_size: 8
            window_ttl_seconds: 900
            min_window_outcomes: 3
            min_consecutive_regressions: 2
            min_consecutive_recoveries: 2
            cooldown_seconds: 120
            max_switches_per_window: 2
```

The gate is disabled when the section or `enabled` is omitted. `enabled` and
`mode` are independent: `observe` records soft evidence decisions without
applying them; `enforce` applies them. Neither mode relaxes hard constraints.
The outer protection `apply`/`observe`/`bypass` modes remain separate.

`calibration_id` identifies the external profile that supplied the thresholds,
using `name@version`. It is required when enabling `enforce`; an empty value in
`observe` explicitly means uncalibrated defaults. For example,
`team/session-policy@2026-09` is an identity, not a calibration result. The
router neither trains a calibrator nor verifies the quality of that profile.
Do not use the synthetic E2E fixture identity as a production calibration.

`window_size` accepts 1–256 samples; `window_ttl_seconds` accepts 1–86400 seconds.
Threshold relationships are validated after omitted fields receive defaults.
For example, reducing the window to two samples also requires reducing the
default `min_window_outcomes: 3`. Evidence TTL is distinct from session idle
expiry and the shared store's retention TTL. Retained history cannot be
recreated by subsequently increasing the window.

### Evidence and decisions

Response capture records positive output as `progress`, explicitly reported
zero output as `no_progress`, and unavailable usage as `missing`. Positive
output is a transport-level proxy, not a claim that the answer was correct.
Authenticated, owned outcome ingest can refine a turn to `progress` or
`regression`. Known provider failures remain non-attributable even if a later
verdict says `failed`. `tool_error` is supported as a typed fact; the response
path does not guess tool success from a tool message's presence.

Facts are bounded by count and event time. Duplicate request/model samples
merge; different models and unknown request identities do not. Ingest owns the
quality verdict, capture owns measured usage, and known infrastructure failure
provenance wins over both. Missing confidence, cost, or latency remains unknown,
not a fabricated zero. Cost trends use configured pricing; latency trends use
observed time to first token. These optional trends are exposed for replay,
not used as an additional undocumented selection score.

| Reason | Condition |
| --- | --- |
| `hard_constraint_conflict` | Active tool loop or non-portable context |
| `cold_start` | No observable evidence, including a fully expired window |
| `insufficient_evidence` | Too few attributable samples, or unmet streak/trend thresholds |
| `cooldown` | Time since the last recorded model change is below the configured interval |
| `oscillation_guard` | The age-bounded window already contains the configured maximum switches |

Escalation requires consecutive same-category negative outcomes and a
non-positive progress trend; de-escalation requires consecutive recoveries.
Known de-escalation is determined from the same quality index and exact
candidate reasoning effort. Missing or incomparable quality evidence uses the
conservative escalation path. Missing and non-attributable outcomes neither
extend nor break a streak, but do not count toward attributable coverage.

### Hard constraints and Replay

Before applying a proposal or restoring the current model, the integration
rechecks the existing candidate boundary, configured backend, context limit,
protocol capabilities, portability locks, and the selected algorithm's hard
SLO/quality filters. The gate cannot introduce a candidate excluded by policy.
If the current model is no longer eligible, a soft suppression cannot restore
it. When no eligible outcome remains, the request fails closed through the
existing selection error path.

Replay's `session_policy.switch_gate` contains the evidence version,
`calibration_id`, evidence metrics and their availability, decision/reason,
switch direction, candidate names, switch history, and actual application.
`enforced` describes the gate mode; `applied`, `application_reason`, and
`final_model` describe what happened. A hard lock can apply in either mode.
A rejected rescue is retained in `rescue_switch_gate` alongside the eventual
main-path decision. A non-switch does not invent a gate verdict.

The maintained `progress-gate` E2E profile exercises request capture, authenticated
idempotent feedback, suppression, allowed switches, cooldown, oscillation, and
Replay over HTTP. Its dedicated deployment leaves `router-replay` tests
unchanged.

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
