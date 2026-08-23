# Single Condition Decisions

## Overview

A single-condition decision is the simplest route policy: one signal or
projection output determines whether the route is eligible.

## Key Advantages

- Smallest possible decision shape.
- Easy to read and easy to audit.
- Good baseline before adding more boolean logic.
- Lets one strong signal own a route without extra nesting.

## What Problem Does It Solve?

Some routes do not need a boolean tree. Forcing them into a larger `AND` or `OR` structure adds noise and makes simple policy harder to review.

A single-condition decision keeps the route focused on one decisive match.

## When to Use

Use a single-condition decision when:

- one domain signal is authoritative
- one safety signal should block immediately
- one preference signal chooses a dedicated model

## Configuration

```yaml
document:
  decisions:
    - name: business_route
      description: Route business and management questions.
      priority: 110
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
```

Even for a single condition, keep the route named and reusable. If the policy
becomes more complex later, add explicit boolean groups without changing the
surrounding route structure.

The referenced signal must be declared in the same recipe. A single learned
signal remains probabilistic, so use trusted identity or deterministic policy
for authorization-sensitive routing. See a complete example:
[`config/fragments/decision/single/domain-business.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/single/domain-business.yaml).
