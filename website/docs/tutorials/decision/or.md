# OR Decisions

## Overview

An `OR` decision matches when any child condition matches. Use it when several
independent request types should share the same route outcome.

## Key Advantages

- Avoids duplicating the same route across multiple decisions.
- Keeps fallback or shared-policy routes compact.
- Makes equivalent matches explicit.
- Works well when one model policy spans several topics or signals.

## What Problem Does It Solve?

Without `OR`, teams often duplicate the same route logic several times just to support different match conditions. That creates drift and makes later policy changes risky.

`OR` solves that by collapsing equivalent triggers into one route.

## When to Use

Use `OR` when:

- two domains share the same model policy
- several signal variants map to one fallback route
- one operational plugin should run for several independent cases

## Configuration

```yaml
document:
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
```

Use `OR` when the route outcome is the same, but several signals should be allowed to trigger it.

Any child can make the route eligible, so audit each child as if it were a
standalone route condition. See a complete example:
[`config/fragments/decision/or/business-or-law.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/or/business-or-law.yaml).
