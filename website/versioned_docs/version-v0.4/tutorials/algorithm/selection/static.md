# Static

## Overview

`static` provides deterministic model choice without metrics or learned state.
It selects the first entry in a decision's `modelRefs` by default. A matched
domain can supply fixed `model_scores`; the selector uses the highest score
when it differs from the default score sentinel of `1.0`.

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
`model_scores`, score every candidate and avoid `1.0`, which is reserved by the
selector's first-candidate fallback.
See a complete example:
[`config/fragments/algorithm/selection/static.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/static.yaml).

## Dependencies and Limitations

- No external dependencies and no request-content processing beyond ordinary
  decision matching.
- It does not fail over to later candidates, react to load, or learn from
  outcomes. Backend availability remains the responsibility of the normal
  provider path.
