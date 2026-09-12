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
| `trigger.min_confidence` | — | Acceptance threshold in `(0, 1]`. Required when enabled; there is no implicit default. |
| `scope` | `eligible_history` | The only supported scope. Ranking and selective retention belong to other actions. |
| `failure_mode` | `fail_open` | `fail_open` preserves history on failure; `fail_closed` rejects before provider dispatch. |
| `limits.max_history_turns` | `128` | Upper bound on turns examined per request. |
| `limits.max_history_bytes` | `1048576` | Upper bound on history bytes examined per request. |
| `limits.timeout_ms` | `50` | Planning budget per request. |
| `recovery.enabled` | `false` | When true, removed turns must be stored recoverably before removal commits. |

Configuration cannot widen what may be removed. Eligibility and protection are
owned by the shared context-transformation layer, and a policy that names a
protected message simply has its proposal rejected.

## What Is Always Preserved

- System and developer instructions.
- The live user turn, including its tool continuation.
- Authorization and safety context supplied as trusted router metadata.
- Multimodal and other opaque protected content.
- Retrieved RAG and Memory content, which is outside the eligible scope.
- Complete tool call and result pairs required by anything retained.

A turn is removable only when every one of its messages is eligible and nothing
retained depends on it. Mixed turns are kept whole.

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
