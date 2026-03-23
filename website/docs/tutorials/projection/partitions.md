---
sidebar_position: 2
---

# Partitions

## Overview

`routing.projections.partitions` coordinates competing `domain` or `embedding` signals and keeps one winner.

Use partitions when:

- several related domain or embedding signals can all match the same request
- downstream routing should work from one resolved winner instead of multiple overlapping matches
- you want fallback behavior when nothing in the partition fires

## What Problem Does It Solve?

Without partitions, a request can match several nearby domain or embedding lanes at once. That is often undesirable for routing:

- a request should usually have one main domain winner, not four partially matched domains
- an intent lane should usually collapse to one best-fit embedding category before decisions evaluate it
- repeated route rules become harder to reason about when every decision has to defend against overlapping matches

Partitions solve that by coordinating the detector results after signal extraction but before decision evaluation.

## How Partitions Behave at Runtime

In the current implementation:

- partitions only accept `domain` or `embedding` members
- all members in one partition must share the same type
- `default` is required and must also appear in `members`
- if several members matched, the runtime keeps one winner
- if no member matched, the runtime synthesizes the `default` member into the matched set

Supported semantics:

- `exclusive`: keep the highest-confidence winner as-is
- `softmax_exclusive`: keep the same winner, but renormalize contender confidences with softmax using `temperature`

Two practical consequences:

- decisions still reference the winning member by its native type such as `type: domain` or `type: embedding`
- decisions do not reference the partition name itself

So partitions are not "named projection outputs" in the same sense as mappings. They are coordination over existing signal names.

## Canonical YAML

```yaml
routing:
  projections:
    partitions:
      - name: balance_domain_partition
        semantics: softmax_exclusive
        temperature: 0.10
        members: [law, business, health, history, other]
        default: other

      - name: balance_intent_partition
        semantics: softmax_exclusive
        temperature: 0.18
        members: [code_general, architecture_design, research_synthesis, general_chat_fallback]
        default: general_chat_fallback
```

## DSL

```dsl
PROJECTION partition balance_intent_partition {
  semantics: "softmax_exclusive"
  temperature: 0.18
  members: ["code_general", "architecture_design", "research_synthesis", "general_chat_fallback"]
  default: "general_chat_fallback"
}
```

## Config Fields

| Field | Meaning |
|-------|---------|
| `name` | partition identifier for config and DSL |
| `semantics` | winner-selection mode: `exclusive` or `softmax_exclusive` |
| `temperature` | only meaningful for `softmax_exclusive`; lower values make the winner more decisive |
| `members` | existing `domain` or `embedding` signal names to coordinate |
| `default` | fallback member synthesized when none of the members matched |

## When to Use

Use partitions when:

- one request should have one dominant domain before routing
- several embedding lanes represent alternative intents and should collapse to one winner
- you want downstream decisions to stay simple and read the winning raw signal directly

## When Not to Use

Do not use partitions when:

- multiple members should remain independently visible to decisions
- the group mixes unrelated concepts that should not compete with each other
- you need a reusable named tier like `balance_reasoning`; that belongs in a mapping, not a partition

## Design Notes

- Keep raw detector definitions under `routing.signals`; partitions only coordinate them.
- Group members that belong to the same routing question, such as one domain family or one embedding family.
- Add `default` when downstream routing should keep a stable fallback even if no member clearly wins.
- If you use `softmax_exclusive` on embedding partitions, native DSL validation can warn when member centroids are too similar to separate cleanly.

## Next Steps

- Pair a winner with [Scores](./scores) when the resolved signal should still contribute to a weighted route score.
- Use [Mappings](./mappings) when decisions should read named routing bands instead of raw signal winners.
