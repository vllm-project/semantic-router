# Single Condition Decisions

## Overview

Use `config/decision/single/` when one signal is enough to pick a route.

This is the cleanest entry point for a route that has one authoritative detector.

## Key Advantages

- Smallest possible decision shape.
- Easy to read and easy to audit.
- Good baseline before adding more boolean logic.
- Lets one strong signal own a route without extra nesting.

## What Problem Does It Solve?

Some routes do not need a boolean tree. Forcing them into a larger `AND` or `OR` structure adds noise and makes simple policy harder to review.

`single/` solves that by keeping the route focused on one decisive match.

## When to Use

Use `single/` when:

- one domain signal is authoritative
- one safety signal should block immediately
- one preference signal chooses a dedicated model

## Configuration

Source fragment: `config/decision/single/domain-business.yaml`

```yaml
routing:
  decisions:
    - name: business_route
      description: Route business and management questions.
      priority: 110
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

Even for a single condition, keep the route named and reusable. If the policy becomes more complex later, you can promote it to `and/`, `or/`, or `composite/` without changing the surrounding config layout.
