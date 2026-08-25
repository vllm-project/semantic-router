# AND Decisions

## Overview

An `AND` decision matches only when every child condition matches. Use it for
narrow routes that require several independent facts.

## Key Advantages

- Reduces false positives by requiring multiple detectors.
- Works well for escalation and premium routes.
- Keeps compound requirements explicit instead of hidden in one signal.
- Produces predictable route boundaries.

## What Problem Does It Solve?

A single signal often matches too broadly. Domain alone may be insufficient without urgency, safety, or complexity context.

`AND` solves that by requiring all required signals to agree before the route becomes eligible.

## When to Use

Use `AND` when:

- domain and urgency must both be present
- domain and safety clearance must both pass
- preference and complexity should cooperate before escalation

## Configuration

```yaml
routing:
  decisions:
    - name: urgent_business_route
      description: Match only when business intent and urgent language appear together.
      priority: 140
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
          - type: keyword
            name: urgent_keywords
```

Use `AND` when a model should only activate for a narrow, high-confidence slice of traffic.

Every referenced signal must be declared in the same recipe. `AND` reduces
broad matches but does not make probabilistic signals authoritative. See a
complete example:
[`config/fragments/decision/and/urgent-business.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/and/urgent-business.yaml).
