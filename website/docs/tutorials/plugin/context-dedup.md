# Context Dedup

## Overview

`context_dedup` is a route-local context plugin that removes the later copy of prior turns a request carries twice in a row, so repeated history stops costing input tokens without changing what the conversation means. It never guesses that two different texts are equivalent: only a complete text-only turn, or a block of consecutive turns, that immediately repeats an identical block is removed, and the earlier copy is always the one kept.

The plugin is disabled by default. It mutates only the context view sent upstream; stored conversations, Router Memory, session accounting, provider state, and tool authorization are untouched.

## Key Advantages

- Reclaims context window when a client re-sends history it already sent.
- Removes whole turns only, never fragments of a turn or a tool exchange.
- Keeps confirmations, retries, corrections, and repeats separated by other content.
- Keeps instructions, the live turn, and protected content in every path.
- Needs no recovery store: every removed message has a byte-identical retained twin.
- Emits content-free receipts describing what was examined, retained, and removed.

## What Problem Does It Solve?

Clients that rebuild a request from a transcript, retry a send after a timeout, or prepend server-side history to input they already hold can deliver the same turns twice in one request. The model then reads the same exchange more than once, which costs tokens and can bias it toward repeated material. Removing text that merely looks similar is unsafe: a repeated "yes" is a confirmation, a repeated question is a retry, and a second identical tool execution may have happened at a different time. This plugin removes only the narrow, provable case.

## When to Use

Use it on decisions that serve clients known to re-send history, or as a cheap safeguard ahead of `context_compression` on long-session routes. Do not use it when the application depends on the router forwarding its request byte for byte, or when repeated turns carry meaning for the application even though they are identical.

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: context_dedup
    configuration:
      enabled: false
      normalization: exact
      failure_mode: fail_open
      limits:
        max_history_turns: 128
        max_history_bytes: 1048576
        max_segment_turns: 64
        timeout_ms: 50
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Omission or `false` performs no deduplication work at all. |
| `normalization` | `exact` | `exact` compares text byte for byte. `whitespace` collapses runs of white space and trims the ends first. Neither mode changes case, punctuation, or Unicode form. |
| `failure_mode` | `fail_open` | `fail_open` preserves the request when the policy cannot decide safely; `fail_closed` rejects it before provider dispatch. |
| `limits.max_history_turns` | `128` | Upper bound on history turns examined per request. Beyond it the whole step is skipped; a prefix is never deduplicated. |
| `limits.max_history_bytes` | `1048576` | Upper bound on the history text the policy inspects. Tool arguments and media are not part of that view. |
| `limits.max_segment_turns` | `64` | Largest block of consecutive turns one repeat may span. A whole history re-sent as one block needs a bound at least as large as that history. Must not exceed `max_history_turns`. |
| `limits.timeout_ms` | `50` | Budget for the policy's own scan. The shared transformation view is prepared before the policy runs and is not inside this budget. |

Configuration cannot widen what may be removed. Eligibility and protection are owned by the shared context-transformation layer, and a policy that names a protected message simply has its proposal rejected.

## What Is Removed

A turn is a user message together with the assistant and tool messages that follow it. A turn is a candidate only when every one of these holds:

| Rule | Requirement | Retained reason when it fails |
| --- | --- | --- |
| Eligible | Every message is unprotected history. Instructions, the live turn, multimodal content, authorization and safety context, RAG and Memory content, and messages with unknown turn membership are never candidates. | `ineligible` |
| No tool exchange | No message belongs to a tool call or result. Repeated tool executions are kept whether their call IDs repeat or differ. | `tool_exchange` |
| Complete | The turn opens with a user message and contains an assistant reply. A repeated user message without a reply is a retry or a confirmation. | `incomplete_turn` |
| Text only | Every message has text the policy can see, and nothing else. | `opaque_content` |

Candidate turns are then compared in two stages. The first compares roles and normalized text through the shared view. The second proves every matched message pair against the neutral request: same role, equal item IDs or an empty later ID, the same blocks, only text or reasoning blocks, and equal citations, cache directives, signatures, and reasoning scope. A refusal keeps the turn (`refusal_retained`); any other difference keeps it (`identity_mismatch`).

Only adjacent repetition is removed. A block of consecutive candidate turns that is immediately followed by an identical block loses its later copy, and the same position is checked again so a triple send collapses to one copy. Identical turns separated by other content are temporal repetitions and are kept (`non_adjacent`). Because the earlier copy is always the one kept, the order and provenance of everything retained never change.

| Conversation | Result |
| --- | --- |
| `U1 A1 U1 A1 U2` | The second `U1 A1` is removed. |
| `U1 A1 U2 A2 U1 A1 U2 A2 U3` | The second `U1 A1 U2 A2` block is removed. |
| `U1 A1 U1 A2 U2` | Kept: the answers differ, so this is a correction or a regenerated reply. |
| `U1 U1 A1` | Kept: the lone repeated `U1` is a retry without a reply. |
| `U1 A1 "yes" "yes"` | Kept: a repeated one-message turn is never complete. |
| `U1 A1 U2 A2 U1 A1 U3` | Kept: the repeat is not adjacent to its original. |
| A repeated tool call and result with the same or different IDs | Kept: tool exchanges are outside the policy. |

## What Is Always Preserved

- System and developer instructions.
- The live user turn, including its tool continuation.
- Authorization and safety context marked by a trusted router component.
- Multimodal and other opaque content.
- Retrieved RAG and Memory content, which is outside the eligible scope.
- Every tool call and result pair.
- Every turn the policy cannot prove equal to the turn before it.

## Failure Behavior

| Condition | `fail_open` | `fail_closed` |
| --- | --- | --- |
| Adjacent repeated turns found | Remove the later copies | Same |
| No duplicates, or no eligible history | Preserve the request; normal no-op | Same |
| Planning limit exceeded | Preserve the request and record the reason | Reject before provider dispatch |
| Timeout or cancellation during the scan | Preserve the request | Reject before provider dispatch |
| Request representation cannot be proven (no neutral request) | Preserve the request | Reject before provider dispatch |

Every outcome records a bounded terminal reason, for example `applied`, `no_duplicates`, `history_limit_exceeded`, `cancelled`, or `equivalence_unverifiable`. A preserved request keeps its enrichment, tools, metadata, and generation exactly as they were.

## Supported Paths

Deduplication runs in the shared context stage, after retrieval and memory enrichment and before compression, so OpenAI Chat Completions, the Responses API, and Anthropic Messages are handled once in the neutral request. Instructions, tool links, multimodal blocks, and request metadata survive the encode step unchanged.

Responses items carry IDs. A later copy without IDs is treated as a repeat of the earlier copy that has them, which is the shape produced when a client re-sends history it received from a provider. A later copy carrying a different ID is a different item and is kept.

Internal router hops, such as the algorithm loop, do not run this step until the shared context stage is extended to internal requests; the ingress request is where repeated history arrives.

## Interaction With Context Compression

Deduplication runs before compression in the shared context pipeline. Enabling any history action, including this one, also turns on shared protection against compressing history blocks that belong to the live turn, multimodal messages, and messages whose turn ownership is unknown. That protection applies even when nothing is removed, so enabling deduplication can reduce the savings a separately configured `context_compression` policy reports. Leaving it disabled keeps existing compression eligibility unchanged.

Response caching is unaffected: the cache key is computed from the request as received and from the decision's plugin configuration, and deduplication is a deterministic function of both that preserves behavior.

## Observability

Every configured evaluation records a content-minimized receipt: the outcome, the terminal reason, the normalization mode, examined message and turn counts, candidate and protected counts, retained and removed counts, removed text bytes, the number of duplicate segments, per-reason retention counts, and a bounded list of removed segments identified by pre-transform message positions. Receipts never contain message text. Recovery is reported as `not_required`, because every removed message has an identical retained twin.

Router Replay stores the same fields under `context_dedup` when replay capture is enabled. Independently of replay, the router emits `llm_context_dedup_evaluations_total` (labelled by decision, outcome, and reason), `llm_context_dedup_removed_turns`, `llm_context_dedup_removed_messages`, and `llm_context_dedup_removed_text_bytes`. Metric labels stay low cardinality and carry no conversation content.
