# Topic Continuity Signal

## Overview

`topic_continuity` reports whether the live user turn still depends on the
retained conversation history. It produces bounded, typed **evidence** —
`continuation`, `change`, or `unknown` — that context policies (for example a
history-reset action) can read before they decide anything.

The signal never changes the request. It does not reset, compress, or store
history, and it is **not decision-referenceable**: routing decisions cannot
name it in their rules. Every rule declared by the selected recipe is
evaluated once per provider-bound request, immediately before the context
transformation stage, and the typed results are stored on the request
context.

Evaluation is purely lexical and deterministic. No classifier, embedding
model, or other inference runs, and identical input always produces an
identical result.

## Key Advantages

- separates "the user changed topic" from "the evidence is missing or weak",
  so context policy can act only on positive evidence
- reads the original history captured before RAG and Memory enrichment, so
  injected content never changes the evidence
- bounded by counted limits (turns, bytes, messages, content blocks, entities)
  rather than wall-clock time
- content-minimized observability: Router Replay and metrics record classes,
  reasons, coverage, and counts, never conversation text

## What Problem Does It Solve?

Long sessions accumulate history that stops being relevant after the user
moves on. A context policy that removes history on a topic change must not
confuse a real change with missing evidence: deleting context the live turn
still needs is far more expensive than keeping context it no longer needs.
`topic_continuity` makes that distinction explicit and typed, and it is
deliberately asymmetric — any missing or ambiguous evidence yields `unknown`,
never `change`.

## When to Use

Declare a `topic_continuity` rule when a context policy in the same recipe
consumes it. Declaring a rule that nothing consumes still evaluates it (the
result appears in Router Replay), so declare rules only where they are used.

## Configuration

```yaml
routing:
  signals:
    topic_continuity:
      - name: topic_boundary
        description: Whether the live turn still depends on retained history.
        # every field below is optional; shown with defaults
        include_assistant: true
        thresholds:
          continuation: 0.35
          change: 0.08
        limits:
          max_prior_turns: 8
          max_turn_bytes: 16384
          max_input_bytes: 147456   # default (max_prior_turns + 1) * max_turn_bytes
```

| Field | Default | Valid range |
| --- | --- | --- |
| `include_assistant` | `true` | `true` or `false` |
| `thresholds.continuation` | `0.35` | `change < continuation < 1` |
| `thresholds.change` | `0.08` | `0 <= change < continuation` |
| `limits.max_prior_turns` | `8` | 1–32 |
| `limits.max_turn_bytes` | `16384` | 256–65536 |
| `limits.max_input_bytes` | `(max_prior_turns + 1) * max_turn_bytes` | 1024–1048576, at least `max_turn_bytes` |

A recipe may declare at most 8 rules, and names must be unique and trimmed.
If the derived `max_input_bytes` would exceed 1 MiB, validation fails and asks
for an explicit value; nothing is clamped silently.

A decision condition or projection input of type `topic_continuity` is
rejected during validation.

## Result

Each rule produces one result:

| Field | Meaning |
| --- | --- |
| `class` | `continuation`, `change`, or `unknown` |
| `reason` | A bounded reason code whose prefix names its class, for example `continuation_reference`, `change_explicit_marker`, or `unknown_history_beyond_window` |
| `confidence` | A fixed-formula heuristic strength in [0, 1], **not** a calibrated probability. `0` for `unknown`. |
| `coverage` | `full`, `window`, or `partial` — see below |
| `scope` | Whether assistant text was included, whether excluded content (tool arguments or results, reasoning, media) was present, and whether a feature cap was reached |
| `schema_version` | `v1`. Consumers gate on this. |
| `evaluator_version` | The lexicon and formula revision, for example `lexical.1` |

### Two confidence tiers for `change`

- **Explicit change** (`change_explicit_marker`, confidence `0.9`): the live
  turn starts with a change phrase such as "Unrelated question:" or "New
  topic:".
- **Disjoint change** (`change_disjoint`, confidence `0.3`–`0.6`): the live
  turn shares no meaningful words or entities with any evaluated turn and is a
  self-contained prose request. This is absence of observed overlap, which is
  weaker evidence; treat it as lexical separation strength.

A consumer that wants only explicit evidence can require a confidence above
`0.6` or filter on the reason.

### Coverage

`coverage` is relative to the configured evidence policy:

- `full` — every policy-selected turn was processed, nothing was truncated,
  and no turn exists beyond `max_prior_turns`.
- `window` — the window was complete, but older turns exist beyond it.
- `partial` — something was truncated or dropped, or a cap was reached.

**`change` always implies `coverage: full`.** A long session whose eligible
history does not fit in the window yields `unknown_history_beyond_window`
instead of `change`, so a destructive consumer never removes turns that were
not evaluated. Raise `max_prior_turns` (up to 32) and the byte limits if you
need `change` in longer sessions.

## Evaluation Rules

- **Live turn**: the final message must be a user message, or a tool result
  inside a tool exchange started by a user turn. A final assistant message
  (including an empty prefill) yields `unknown_no_live_user_turn`.
- **Evidence**: user text and, when `include_assistant` is set, assistant text
  of each prior turn, plus tool *names*. Instructions, tool arguments, tool
  results, reasoning, and media are never read.
- **Truncation** keeps the head and tail of an over-long block as separate
  segments, so no word or phrase can form across the cut.
- **Continuation evidence** comes from reference phrases ("as you said",
  "the function above"), acknowledgements ("thanks"), and word and entity
  overlap (identifiers, file paths, quoted spans, numbers) with recent turns.
- **Change phrases** are matched only at the start of the live turn, never
  inside code, backticks, or double quotes, never after a negation ("not
  unrelated"), and never inside paired single quotes. Any other occurrence is
  ambiguous and blocks `change`.

The phrase lexicons are English-first. Text in other languages still uses word
and entity overlap (CJK text is tokenized into characters and bigrams), but
cannot use the phrase rules.

## Cost

Evaluation work is bounded by the limits above and by fixed caps (512
messages and 4,096 content blocks scanned, 16 tool names per turn, 1,024
entities per text segment). With the defaults, the worst case — every turn
filled to the 16 KiB limit — costs a few milliseconds; ordinary conversations
cost far less. In addition, the original history is decoded once per request
when the selected recipe declares at least one rule.

## Observability

Results appear in Router Replay records (class, reason, confidence, coverage,
scope, versions, and content-free scores), in the
`llm_topic_continuity_evaluations_total{class,reason,coverage,fallback}`
metric, and in a `topic_continuity_evaluated` log event. No conversation text
or content digest is recorded.

## Related Signals

- [`conversation`](./conversation) — structural request facts such as
  message counts and tool flows.
- [`context`](./context) — token-count bands for the request.
