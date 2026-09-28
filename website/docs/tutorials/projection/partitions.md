---
sidebar_position: 2
---

# Partitions

## Overview

`routing.projections.partitions` coordinates competing `domain` or `embedding` signals and keeps one winner.

## What Problem Does It Solve?

Without partitions, a request can match several nearby domain or embedding lanes at once. That is often undesirable for routing:

- a request should usually have one main domain winner, not four partially matched domains
- an intent lane should usually collapse to one best-fit embedding category before decisions evaluate it
- repeated route rules become harder to reason about when every decision has to defend against overlapping matches

Partitions solve that by coordinating the detector results after signal extraction but before decision evaluation.

## How Partitions Behave at Runtime

Partitions follow these rules:

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

## Configuration

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

Partitions make no additional model calls; they coordinate results produced by
their member signals. A configured default is a routing fallback, not evidence
that the default actually matched. See a complete example in the
[`config/recipes/balance/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml).
