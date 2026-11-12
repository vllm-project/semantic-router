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
      adaptation:
        enabled: false
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

This enables protection independently of online model-choice adaptation. Both
components otherwise default to enabled when the master switch is enabled.

## Scopes

| Scope | What is protected | What can re-route |
| --- | --- | --- |
| `conversation` | Turns sharing one `x-conversation-id`. | A new `x-conversation-id` in the same `x-session-id`, or an explicit release. |
| `session` | Turns sharing one `x-session-id`. | Idle timeout, a decision with `adaptations.mode: bypass`, or an explicit release. |

- **Use `conversation`** when each agent run should be routed on its own. A
  changed decision resets the minimum-turn and continuity-cost preferences.
- **Use `session`** when one model choice should hold across several
  user-initiated runs. A changed decision or conversation alone does not
  release an eligible current model; idle timeout, bypass, candidate exclusion,
  qualified rescue or an explicit release can.
- **Neither scope** can keep a previous model outside the admitted candidate
  set.

If the configured identity headers are missing, protection fails open and
records diagnostics instead of failing the request.

### Agents that delegate to subagents

A harness such as Claude Code sends one session ID for a run and its subagents.
Under `scope: session` that holds every subagent on the model its parent was
given, even when a decision routes the subagent's request elsewhere. Name the
header that identifies each agent to hold them separately:

```yaml
protection:
  enabled: true
  scope: session
  identity:
    headers:
      session: x-claude-code-session-id
      agent: x-claude-code-agent-id
```

Claude Code sends `x-claude-code-agent-id` only on subagent requests, so the
main agent keeps its session key and each subagent gets its own. A subagent is
held on the model chosen for its first request, under the same rules as a
session. Without `agent`, protection behaves as before.

Codex CLI sends `session-id` and `thread-id` on every request. A root thread
sends its own ID in both. A review child that Codex starts on that thread keeps
the root's `session-id` and sends its own `thread-id`:

```yaml
protection:
  enabled: true
  scope: session
  identity:
    headers:
      session: session-id
      agent: thread-id
```

- **The root's key** holds both its session ID and its thread ID, because the
  root also sends `thread-id`.
- **Both IDs stay the same** across tool continuations and across turns after
  `codex exec resume`, so protection holds the root and each review child on
  their own models for the whole session.
- **`codex exec review`** starts a new session, so its review child gets its
  own key either way.

These headers were checked with Codex CLI 0.167.0, configured as the
[protocol compatibility guide](../../installation/protocol-compatibility)
requires.

### Explicit release

A harness that knows its task has changed phase can ask for a fresh decision
without starting a new session. It sends the release header on one request;
protection releases the hold on that request's key, runs the recipe again and
holds whatever the recipe selects.

- **One run.** The session ID does not change, so Replay and run-level
  reporting keep one run.
- **A new decision, not a new model.** The recipe can select the same model
  again.

```yaml
protection:
  release:
    enabled: true                        # default
    header: x-vsr-protection-release     # default
    min_user_turns_between: 2            # default; 0 removes the limit
```

Any non-empty value requests a release. It applies only to the request's
protection key: with an `agent` header configured, releasing one subagent
leaves the main agent and the other subagents held.

The Router removes the header before the request reaches a backend, whether or
not release is enabled. Config load therefore accepts only an `x-` header name,
and rejects:

- an identity header;
- a credential header, such as `x-api-key` or `x-goog-api-key`;
- a gateway header: `x-envoy-*`, `x-authz-*`, `x-forwarded-*`, `x-request-id`;
- another `x-vsr-` control header.

A release is honored only when all of these hold:

| Condition | Otherwise |
| --- | --- |
| The key already holds a model. A new session or agent gets a normal first decision instead. | — |
| The request is at a portable turn boundary, with no active tool loop and no nonportable context. | `release_not_at_turn_boundary` |
| At least `min_user_turns_between` user turns lie between the last honored release on the key and this one. | `release_interval` |
| `release.enabled` is not `false`. | `release_disabled` |
| Protection runs in `apply` mode for the request. | `protection_bypassed` (a decision with `adaptations.mode: bypass`), `protection_disabled`, `missing_identity`, or `release_observe_only` (`protection.mode: observe`) |

With the default `min_user_turns_between: 2`, a release on turn 3 allows the
next one on turn 6, and `0` removes the limit. A user turn is a request outside
a tool loop; tool hops do not count.

What a release does:

- **Refused:** the current hold stays, the reason appears in `x-vsr-warnings`,
  and the interval is not used up.
- **Honored:** it skips the held-model shortcut and rescue. Hard candidate
  eligibility still applies, and Replay records `release_switch` with the
  previous and selected model.
- **Honored, same model:** when the recipe selects the held model again, the
  interval is still used up, and Replay records `hold_current` with the reason
  `release_kept_current`.
- **With the progress gate:** its evidence, cooldown and oscillation rules
  don't suppress a release; the interval replaces them, and the gate trace
  records `bypassed_by_release`.
- **Under `protection.mode: observe`:** routing is unchanged, and the release
  is recorded as not honored, without using up the interval.

:::warning
Do not put the release header in Claude Code's `ANTHROPIC_CUSTOM_HEADERS`.
That setting is fixed at startup and sent on every request, so every request
would ask for a release. Protection would then re-decide every
`min_user_turns_between + 1` user turns, which effectively turns the hold off.
:::

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

Protection can also allow a deterministic `rescue_switch` when repeated
backend failures or replay-linked outcomes show the current model is
underpowered:

- **Failure evidence.** With protection identity present, backend `429` and
  `5xx` responses supply failure observations, even when adaptation is
  disabled. This doesn't enable adaptive model choice or quality updates.
- **Not an outcome.** A correction or retry prompt alone is not an owned model
  outcome, though its signals can change the matched decision.
- **Not failover.** Rescue needs another eligible proposal and enough
  evidence; it is not immediate backend failover.

Rescue boundaries:

- Rescue is limited to eligible models at a portable turn boundary. It cannot
  override an active tool-loop lock or a nonportable-context lock.
- When those hard locks are absent, minimum-turn and session-continuity
  preferences can yield to rescue.
- Adaptation and protection keep any candidate restrictions from the
  decision's selector, lexicographic bands included.

Ownership locks, with protection in `apply` mode and the configured identity
present:

- **Outside the pool:** a hard lock whose previous model is outside the
  admitted pool returns `503`, instead of transferring ownership or forcing
  that model.
- **Responses history** becomes portable once the Router has expanded it into
  the complete stateless request; a retained `previous_response_id` alone
  creates no lock. Unknown response IDs fail during history lookup, before
  selection.
- **An active tool loop** locks while the conversation stays on the decision
  that admitted the loop. When classification moves to another decision, the
  lock releases and the new decision routes from its own candidates, traced as
  `decision_drift`.
- **Opaque provider state** stays bound to the model that created it.

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

- **Off by default.** The gate is disabled when the section or `enabled` is
  omitted.
- **Modes.** `enabled` and `mode` are independent: `observe` records soft
  evidence decisions without applying them, and `enforce` applies them.
  Neither relaxes hard constraints, and the outer protection
  `apply`/`observe`/`bypass` modes stay separate.

`calibration_id` names the external profile that supplied the thresholds, as
`name@version`:

- It is required for `enforce`; an empty value in `observe` means
  uncalibrated defaults.
- `team/session-policy@2026-09` is an identity, not a calibration result: the
  Router neither trains a calibrator nor checks the profile's quality.
- Don't use the synthetic E2E fixture identity as a production calibration.

Windows and validation:

- `window_size` accepts 1–256 samples, and `window_ttl_seconds` 1–86400
  seconds.
- Threshold relationships are validated after omitted fields get their
  defaults: reducing the window to two samples also means reducing the default
  `min_window_outcomes: 3`.
- Evidence TTL is separate from session idle expiry and the shared store's
  retention TTL. Raising the window later cannot recreate history that has
  expired.

### Evidence and decisions

What each outcome means:

| Outcome | Source |
| --- | --- |
| `progress` | Positive output in response capture: a transport-level proxy, not a claim that the answer was correct. Authenticated, owned outcome ingest can also set it. |
| `no_progress` | Explicitly reported zero output. |
| `missing` | Usage unavailable. |
| `regression` | Authenticated, owned outcome ingest. |
| `tool_error` | A typed fact; the response path doesn't guess tool success from a tool message's presence. |

Known provider failures stay non-attributable, even if a later verdict says
`failed`.

How facts combine:

- **Bounded** by count and event time. Duplicate request/model samples merge;
  different models and unknown request identities do not.
- **Ownership.** Ingest owns the quality verdict, capture owns measured usage,
  and known infrastructure-failure provenance wins over both.
- **Unknown stays unknown.** Missing confidence, cost or latency is never a
  fabricated zero.
- **Trends.** Cost trends use configured pricing, and latency trends use
  observed time to first token. They are exposed for replay, not used as a
  hidden selection score.

| Reason | Condition |
| --- | --- |
| `hard_constraint_conflict` | Active tool loop or non-portable context |
| `cold_start` | No observable evidence, including a fully expired window |
| `insufficient_evidence` | Too few attributable samples, or unmet streak/trend thresholds |
| `cooldown` | Time since the last recorded model change is below the configured interval |
| `oscillation_guard` | The age-bounded window already contains the configured maximum switches |

Streaks:

- **Escalation** needs consecutive same-category negative outcomes and a
  non-positive progress trend; **de-escalation** needs consecutive recoveries.
- **Direction** comes from the same quality index and the exact candidate
  reasoning effort. Missing or incomparable quality evidence takes the
  conservative escalation path.
- **Gaps.** Missing and non-attributable outcomes neither extend nor break a
  streak, and don't count toward attributable coverage.

### Hard constraints and Replay

Before applying a proposal or restoring the current model, the integration
rechecks:

- the existing candidate boundary and configured backend;
- the context limit and protocol capabilities;
- portability locks;
- the selected algorithm's hard SLO and quality filters.

The gate cannot introduce a candidate that policy excludes. If the current
model is no longer eligible, a soft suppression cannot restore it; when no
eligible outcome remains, the request fails closed through the existing
selection error path.

Replay's `session_policy.switch_gate` records:

- the evidence version, `calibration_id`, the evidence metrics and their
  availability;
- the decision and reason, switch direction, candidate names and switch
  history;
- what happened: `enforced` describes the gate mode, while `applied`,
  `application_reason` and `final_model` describe the outcome. A hard lock can
  apply in either mode.

A rejected rescue is kept in `rescue_switch_gate`, beside the eventual
main-path decision. A non-switch does not invent a gate verdict.

The maintained `progress-gate` E2E profile exercises request capture, authenticated
idempotent feedback, suppression, allowed switches, cooldown, oscillation, and
Replay over HTTP. Its dedicated deployment leaves `router-replay` tests
unchanged.

## Decision Boundaries

Most decisions do not need local configuration. Use `bypass` for hard policy
boundaries:

```yaml
recipes:
  - name: default
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
x-vsr-trace: session_phase=user_turn, learning=(protection;action=hold_current;scope=conversation;reason=cache_cost_high)
```

Client UIs should translate raw actions into user-facing text:

| Action | Display as |
| --- | --- |
| `hold_current` | "kept run model" |
| `allow_switch` | "switch allowed" |
| `rescue_switch` | "rescue switch" |
| `bypass` | "learning bypassed" |

Router Replay stores the full protection trace: identity source and hash,
protected model, base model, proposal model, final model, switch cost, cache
evidence, tool-loop state, mode, scope, action, and reason.
