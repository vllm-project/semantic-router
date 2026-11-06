# Confidence (removed)

## Overview

The Looper `confidence` algorithm was removed. Its behavior is the Chat path of
[Escalate](./escalate).

## What Problem Does It Solve?

It tried candidates in order and stopped when an answer's token confidence
cleared a threshold. `escalate` with `accept.confidence` does the same, and
`accept.systemone` replaces its `automix_entailment` verifier server with the
decision model's `answer_check` question.

## When to Use

Do not configure it: a configuration that still names it fails at load. Use
[Escalate](./escalate).

## Configuration

Run `vllm-sr config migrate` to rewrite it. A decision such as

```yaml
algorithm:
  type: confidence
  confidence:
    confidence_method: hybrid
    threshold: 0.72
    escalation_order: small_to_large
```

becomes

```yaml
algorithm:
  type: escalate
  escalate:
    order: size
    accept:
      confidence: {method: hybrid, gte: 0.72}
```

See [Migrating from confidence and cascade](./escalate#migrating-from-confidence-and-cascade)
for the other fields.
