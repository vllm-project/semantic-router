# Static

## Overview

`static` provides deterministic model choice without metrics or learned state.
It selects the first entry in a decision's `modelRefs` by default. A matched
domain can supply fixed `model_scores`; the selector uses the highest score
when at least one candidate has a configured score. Explicit scores of `1.0`
participate in ranking like any other configured score.

## Key Advantages

- Deterministic and easy to audit.
- Has no selector model, metrics, or storage dependency.
- Provides a stable baseline for other selection policies.

## What Problem Does It Solve?

Some routes already have an intentional model order or fixed per-domain scores
and do not need an online ranking policy. Static makes that choice explicit.

## When to Use

Use Static for deterministic routing, as a baseline when comparing selectors,
or when an external process owns candidate ordering. If a decision has only
one candidate, you can usually omit the algorithm entirely.

## Configuration

```yaml
algorithm:
  type: static
```

Place the intended fallback winner first in `modelRefs`. To rank with domain
`model_scores`, score every candidate. Unscored candidates retain the default
score of `1.0`; if no candidate has a configured score, the selector falls back
to the first candidate. Equal highest scores also keep the first candidate in
the tied set.
See a complete example:
[`config/fragments/algorithm/selection/static.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/static.yaml).

## Dependencies and Limitations

- No external dependencies and no request-content processing beyond ordinary
  decision matching.
- It does not fail over to later candidates, react to load, or learn from
  outcomes. Backend availability remains the responsibility of the normal
  provider path.
