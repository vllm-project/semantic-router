# AND Decisions

## Overview

Use `config/decision/and/` when multiple signals must all match before the route is valid.

`AND` is the standard shape for narrow, high-confidence routes.

## Key Advantages

- Reduces false positives by requiring multiple detectors.
- Works well for escalation and premium routes.
- Keeps compound requirements explicit instead of hidden in one signal.
- Produces predictable route boundaries.

## What Problem Does It Solve?

A single signal often matches too broadly. Domain alone may be insufficient without urgency, safety, or complexity context.

`AND` solves that by requiring all required signals to agree before the route becomes eligible.

## When to Use

Use `and/` when:

- domain and urgency must both be present
- domain and safety clearance must both pass
- preference and complexity should cooperate before escalation

## Configuration

Source fragment: `config/decision/and/urgent-business.yaml`

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
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

Use `AND` when a model should only activate for a narrow, high-confidence slice of traffic.

For token-count routing, keep low-context and high-context routes mutually guarded. A route named for low-token traffic should include the matching context signal; otherwise a broad domain-only route with higher priority can still win for long prompts that also match the domain:

```yaml
routing:
  decisions:
    - name: low_token_code_route
      description: Route short code prompts to the code model.
      priority: 180
      rules:
        operator: AND
        conditions:
          - type: domain
            name: code_generation
          - type: context
            name: low_token_count
      modelRefs:
        - model: code-lite
          use_reasoning: false

    - name: high_token_route
      description: Route long prompts to the long-context model.
      priority: 170
      rules:
        operator: AND
        conditions:
          - type: context
            name: high_token_count
      modelRefs:
        - model: long-context
          use_reasoning: false
```
