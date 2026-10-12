# Random

## Overview

`random` picks one model uniformly from a decision's eligible candidates. The
shared eligibility filter runs first, so the draw only ever sees models that
are already valid for the request; the selector itself adds no filtering,
fallback, or recovery of its own. Every candidate has the same chance, and no
state carries between requests.

## Key Advantages

- Produces an unbiased traffic split with no metrics, artifacts, or training.
- Has no selector model, storage, or external dependency.
- Gives other selectors a control group to be measured against.

## What Problem Does It Solve?

Comparing two selection policies is only meaningful against a baseline that
carries no preference of its own. Any ranked selector — even a deliberately
naive one — encodes an assumption about which model should win. `random`
encodes none, so differences you measure downstream belong to the policy under
test rather than to the baseline.

It is also the simplest way to spread load evenly across interchangeable
candidates when you have no signal that would justify preferring one.

## When to Use

Use Random when running a routing experiment or evaluation that needs a
control arm, when collecting unbiased samples across several models, or when
candidates are genuinely interchangeable and any choice is acceptable. Prefer
`static` when you want a deterministic, auditable choice, and `multi_factor`
or `latency_aware` when you have quality, cost, or latency signals worth
acting on.

## Configuration

```yaml
algorithm:
  type: random
```

The type takes no configuration block. Setting `minimum_candidates` is
worthwhile — it makes the decision fail in a typed, visible way when the
eligibility filter leaves too few models to draw from, rather than quietly
degrading to a single choice.
See a complete example:
[`config/fragments/algorithm/selection/random.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/random.yaml).

## Dependencies and Limitations

- No external dependencies and no request-content processing beyond ordinary
  decision matching.
- There is deliberately no seed setting. Routing stays non-deterministic in
  production; reproducing a specific sequence of choices is not supported. If
  you need to replay routing decisions, record them rather than trying to
  regenerate them.
- Every `modelRefs` entry is one equal slot. `weight` is not read, so this is
  not weighted traffic splitting, and two LoRA variants of the same base model
  count as two independent candidates.
- It does not fail over to another candidate, react to load, or learn from
  outcomes. Backend availability remains the responsibility of the normal
  provider path.
