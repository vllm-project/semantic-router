# History Reset

## Overview

`history_reset` is a route-local context plugin that removes complete eligible
prior turns from the provider-bound request after an accepted topic change. It
never detects the topic change itself: it consumes a typed topic-continuity
signal and turns an accepted result into a bounded, observable removal.

The plugin is disabled by default. It mutates only the context view sent
upstream; stored conversations, Router Memory, session accounting, provider
state, and tool authorization are untouched.

## Key Advantages

- Reclaims context window after a conversation changes subject.
- Removes whole turns and whole tool exchanges, never fragments of either.
- Keeps instructions, the live turn, and protected content in every path.
- Fails open or closed according to an explicit policy.
- Emits content-free receipts describing what was examined and removed.

## What Problem Does It Solve?

Long-lived assistant sessions accumulate history that stops being relevant once
the user moves to an unrelated subject. Sending that history costs input tokens
and can pull the model back toward the previous topic, while deleting the
request wholesale would drop protected instructions, the live question, or half
of a tool exchange.

## When to Use

Use it on decisions serving long multi-topic conversations where stale turns
are pure cost. Do not use it when the application depends on the router
forwarding its complete history, or when removed turns must remain available
without configuring recovery.

## Current Availability

Live removal is not yet available. An enabled policy is rejected during
configuration validation with `history_reset_trigger_unavailable` because no
`topic_continuity` signal family is registered yet; that dependency is tracked
by issue #3342. Disabled policies validate and round-trip today, so the
contract can be reviewed, stored, and rendered before the trigger lands.

Registration alone is not enough to enable the feature. A producer implements
the router's topic-continuity source, and the router refuses to activate a
configuration whose enabled policy has no such producer wired, rather than
starting a route that could only preserve or reject every request. The producer
is asked for its result after the permitted history is resolved, so it
classifies the same conversation the action transforms.

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: history_reset
    configuration:
      enabled: false
      trigger:
        signal: topic_boundary
        min_confidence: 0.9
        accepted_versions:
          - v1
      scope: eligible_history
      failure_mode: fail_open
      limits:
        max_history_turns: 128
        max_history_bytes: 1048576
        timeout_ms: 50
      recovery:
        enabled: false
        store: response_cache
        ttl_seconds: 900
        max_bytes_per_request: 10485760
        max_total_bytes: 268435456
        max_retrievals: 8
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Omission or `false` performs no reset work at all. |
| `trigger.signal` | — | Recipe-local topic-continuity signal that may authorize removal. Required when enabled. |
| `trigger.min_confidence` | — | Acceptance threshold in `(0, 1]`. Required when enabled; there is no implicit default. A confidence that is not a finite value in `[0, 1]` is rejected outright. |
| `trigger.accepted_versions` | — | The producing signal contracts this policy trusts. Required when enabled: a result whose version is not listed cannot authorize removal, so there is no implicit "trust any version". |
| `scope` | `eligible_history` | The only supported scope. Ranking and selective retention belong to other actions. |
| `failure_mode` | `fail_open` | `fail_open` preserves history on failure; `fail_closed` rejects before provider dispatch. |
| `limits.max_history_turns` | `128` | Upper bound on turns examined per request. |
| `limits.max_history_bytes` | `1048576` | Upper bound on the history *text* the policy inspects; tool arguments and media are not part of that view. Recoverable removal is bounded separately by `recovery.max_bytes_per_request`. |
| `limits.timeout_ms` | `50` | Budget for the action's own work: selecting removable turns, closing over tool dependencies, and persisting recovery content. It is checked between steps, not inside one: the topic signal and the shared transformation view are prepared before the action runs, and encoding a single message for recovery cannot be interrupted once started. |
| `recovery.enabled` | `false` | When true, removed turns must be stored recoverably before removal commits. |
| `recovery.max_bytes_per_request` | `1048576` | Per-request payload bound. Each removed message's encoded size is computed before it is encoded, so a message that would not fit is refused without allocating it, and the finished envelope is then checked exactly. When `context_compression` also enables recovery, the effective bound is the stricter of the two: neither action can widen the other's budget. |

Configuration cannot widen what may be removed. Eligibility and protection are
owned by the shared context-transformation layer, and a policy that names a
protected message simply has its proposal rejected.

## What Is Always Preserved

- System and developer instructions.
- The live user turn, including its tool continuation.
- Authorization and safety context, when a trusted router component has marked
  the message. The action consumes that provenance; it never infers
  authorization or safety significance from message text, so an unmarked
  historical message is treated as ordinary history.
- Multimodal and other opaque protected content.
- Retrieved RAG and Memory content, which is outside the eligible scope.
- Complete tool call and result pairs required by anything retained.

A turn is removable only when every one of its messages is eligible and nothing
retained depends on it. Mixed turns are kept whole.

## Evidence and Failure Behavior

Only an accepted topic change authorizes removal. The result must identify the
contract that produced it — an unversioned result is treated as unsupported
rather than trusted — and it must be bound to the request it describes: the router computes an identity for the resolved
original history and the live turn, and evidence carrying a different binding
is treated as stale. Uncertainty never authorizes removal.

| Condition | `fail_open` | `fail_closed` |
| --- | --- | --- |
| Accepted change with removable history | Remove the eligible turns | Same |
| Continuation, or no removable history | Preserve history; normal no-op | Same |
| Missing, unknown, conflicting, low-confidence, invalid-confidence, stale, fallback, wrong-signal, or unsupported-version evidence | Preserve history and record the reason | Reject before provider dispatch |
| Planning limit exceeded | Preserve history | Reject before provider dispatch |
| Required recovery unavailable or its write fails | Preserve history | Reject before provider dispatch |

Every outcome records a bounded terminal reason, for example
`evidence_continuation`, `evidence_stale`, `history_limit_exceeded`, or
`recovery_write_failed`. A preserved request keeps its enrichment, tools,
metadata, and generation exactly as they were. Under `fail_closed` these
conditions answer `503`, and the rejection is captured in Router Replay like any
other router-side refusal. A continuation, an inherited internal follow-up, and
an accepted change with nothing eligible left to remove are ordinary no-ops:
neither failure mode rejects them.

## Recovery

With `recovery.enabled: true`, the removed turns are written to the shared
context-recovery store *before* the removal commits, and the model is offered a
reserved retrieval tool whose accepted keys are exactly the ones this request
issued. A failed or oversized write preserves the history instead of removing
content that could not be stored.

The stored payload is a versioned envelope that keeps the removed messages in
order with their roles, content, and tool links, so a retrieval returns the
original exchange rather than a summary. Retrieval never re-executes historical
tool calls and never changes durable conversation state.

Recovery is one request-level facility shared with `context_compression`: one
store, one budget, one reserved tool, one key set. When both plugins enable it,
their `recovery.store` values must match — configuration validation rejects a
decision that asks for two different stores — and each remaining bound resolves
to the stricter of the two, which is the bound the action actually enforces
before it persists anything. The store is built once per process, so every
decision that enables recovery must agree on one backend and one
`max_total_bytes`; the router refuses to activate a configuration where two
routes disagree.

Required recovery is not supported on streaming requests, because the retrieval
follow-up has nowhere to run. Such a request preserves its history under
`fail_open` or is rejected under `fail_closed`, with the
`streaming_recovery_unsupported` reason. Reset never silently downgrades a
recoverable removal to an irreversible one.

## Supported Paths

Reset runs in the shared context stage, after retrieval and memory enrichment
and before compression, so every supported ingress format is handled once in
the neutral request rather than per protocol. OpenAI Chat Completions, the
Responses API, and Anthropic Messages all keep their own semantics on the way
out: instructions, tool call and result links, multimodal blocks, and request
metadata survive the encode step unchanged.

Responses requests need one extra step. Part of their conversation can live
behind `previous_response_id`, and that stored history is normally materialized
just before the provider call — after the context stage. When a reset policy is
enabled, the router resolves the permitted stored history first, so the reset
sees the conversation the provider would actually receive, and dispatch does
not prepend it a second time. Stored input, lineage, conversation membership,
and public response IDs stay owned by the Responses API and are never altered
by a reset. If that history cannot be resolved, the action is blocked with
`history_unresolved` and the configured failure mode decides.

One ordering constraint applies to the topic-continuity producer. Signals are
evaluated before the stored history is resolved, so a producer that classifies
at that point sees the pre-materialization request; its result will not carry
this request's binding and the action rejects it as stale instead of acting on a
partial conversation. Consuming the resolved snapshot is part of the trigger
integration. Materialization is not moved ahead of signal extraction for all
Responses traffic, because that would change the inputs every existing
classifier sees.

Internal router hops — the algorithm loop and recovery follow-ups — continue a
public turn that was already evaluated. They inherit that completion and never
start a new topic-change event, so an appended tool result or a model change
cannot trigger a second removal. The hop still registers its (non-removing)
history step, which keeps the shared live-history compression protection
consistent between the ingress request and its internal continuations.

Requests dispatched through an external gateway without a recipe-local decision
have no reset policy to apply and behave exactly as before.

## Interaction With Context Compression

Reset runs before compression in the shared context pipeline. Enabling any
history action — including this one — also turns on shared protection against
compressing history blocks belonging to the live turn, multimodal messages, and
messages whose turn ownership is unknown. That protection applies even when
reset removes nothing, so enabling reset can reduce the savings a separately
configured `context_compression` policy reports. Leaving reset disabled keeps
existing compression eligibility unchanged.

## Observability

Every configured evaluation records a content-minimized receipt: the trigger
identity and evidence status, the eligible scope, examined, retained,
protected, and removed counts, the outcome, the terminal reason, and recovery
status. Receipts never contain message text, tool arguments, or recovery keys.
Counts always describe what the router committed, never what a policy proposed.

Router Replay stores the same fields under `history_reset` when replay capture
is enabled. Independently of replay, the router emits
`llm_history_reset_evaluations_total` (labelled by decision, outcome, and
reason), `llm_history_reset_removed_turns`, `llm_history_reset_removed_messages`,
and `llm_history_reset_recovery_total`. Metric labels stay low cardinality and
carry no conversation content or recovery keys.
