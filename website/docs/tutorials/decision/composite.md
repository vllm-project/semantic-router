# Composite Decisions

## Overview

Use `config/decision/composite/` when the policy needs nested `AND`, `OR`, and `NOT` logic in one route.

This is the right shape for realistic production policies where business logic and safety logic have to coexist.

## Key Advantages

- Supports nested logic without flattening policy into unreadable conditions.
- Keeps business, operational, and safety constraints in one route.
- Makes complex eligibility rules explicit and reviewable.
- Avoids duplicating related routes that only differ by one branch.

## What Problem Does It Solve?

Flat boolean rules stop scaling once a route depends on multiple independent branches, exclusions, and escalation paths.

`composite/` solves that by letting you encode a real match tree instead of forcing complex policy into a simplistic shape.

## When to Use

Use `composite/` when:

- domain-specific routing needs urgency or complexity escalation
- production safety policy must exclude unsafe traffic
- one route combines business logic and security logic in the same match tree

## Configuration

Source fragment: `config/decision/composite/priority-safe-escalation.yaml`

```yaml
routing:
  decisions:
    - name: priority_safe_escalation_route
      description: Combine AND, OR, and NOT for a realistic multi-signal routing case.
      priority: 160
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
          - operator: OR
            conditions:
              - type: keyword
                name: urgent_keywords
              - type: complexity
                name: needs_reasoning
          - operator: NOT
            conditions:
              - type: jailbreak
                name: prompt_injection
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: true
```

If a decision needs nested logic, prefer a `composite/` fragment instead of stretching one flat rule block until it becomes unreadable.
