# Composite Decisions

## Overview

A composite decision nests `AND`, `OR`, and `NOT` groups in one route. Use it
when business, operational, and safety requirements must be evaluated together.

## Key Advantages

- Supports nested logic without flattening policy into unreadable conditions.
- Keeps business, operational, and safety constraints in one route.
- Makes complex eligibility rules explicit and reviewable.
- Avoids duplicating related routes that only differ by one branch.

## What Problem Does It Solve?

Flat boolean rules stop scaling once a route depends on multiple independent branches, exclusions, and escalation paths.

Composite decisions encode the policy as a readable match tree instead of
forcing it into a flat list of conditions.

## When to Use

Use a composite decision when:

- domain-specific routing needs urgency or complexity escalation
- production safety policy must exclude unsafe traffic
- one route combines business logic and security logic in the same match tree

## Configuration

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
                name: needs_reasoning:hard
          - operator: NOT
            conditions:
              - type: jailbreak
                name: prompt_injection
```

If a decision needs nested logic, keep the groups explicit instead of
stretching one flat rule block until it becomes unreadable.

Keep nesting shallow enough to review and test each branch. Signal results can
be probabilistic, so a complex tree is not a substitute for authorization or
backend policy. See a complete example:
[`config/fragments/decision/composite/priority-safe-escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/composite/priority-safe-escalation.yaml).
