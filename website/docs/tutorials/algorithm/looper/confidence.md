# Confidence

## Overview

`confidence` is a looper algorithm that escalates across candidate models until confidence is high enough.

It aligns to `config/algorithm/looper/confidence.yaml`.

## Key Advantages

- Supports small-to-large escalation instead of a fixed winner.
- Makes stopping conditions explicit.
- Lets one route trade extra latency for higher confidence only when needed.

## What Problem Does It Solve?

Some routes should try a cheaper candidate first and only escalate when the answer is not confident enough. `confidence` gives that escalation policy a dedicated algorithm instead of embedding it in application code.

## When to Use

- a route should escalate across several candidate models
- confidence should decide whether to continue
- the route should stop as soon as one response is good enough

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: confidence
  confidence:
    confidence_method: hybrid
    threshold: 0.72
    escalation_order: small_to_large
    on_error: skip
```
