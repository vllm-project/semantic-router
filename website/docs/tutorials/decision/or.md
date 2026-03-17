# OR Decisions

## Overview

Use `config/decision/or/` when one route should handle several equivalent signal matches.

`OR` is the right shape when multiple independent signals lead to the same route outcome.

## Key Advantages

- Avoids duplicating the same route across multiple decisions.
- Keeps fallback or shared-policy routes compact.
- Makes equivalent matches explicit.
- Works well when one model policy spans several topics or signals.

## What Problem Does It Solve?

Without `OR`, teams often duplicate the same route logic several times just to support different match conditions. That creates drift and makes later policy changes risky.

`OR` solves that by collapsing equivalent triggers into one route.

## When to Use

Use `or/` when:

- two domains share the same model policy
- several signal variants map to one fallback route
- one operational plugin should run for several independent cases

## Configuration

Source fragment: `config/decision/or/business-or-law.yaml`

```yaml
routing:
  decisions:
    - name: business_or_law_route
      description: Share one route across either business or law traffic.
      priority: 100
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
          - type: domain
            name: law
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

Use `OR` when the route outcome is the same, but several signals should be allowed to trigger it.
