# Decision Adaptations

## Overview

Decision adaptations let a matched decision control whether a global Router
Learning adaptation can adjust its proposed model.

## Key Advantages

- Keeps global learning enabled by default.
- Lets hard policy boundaries opt out in one small block.
- Supports observe-only rollout without changing final routing.
- Avoids duplicating learning rules inside semantic decision conditions.

## What Problem Does It Solve?

Most decisions can allow Router Learning to make stay-vs-switch adjustments.
Some decisions are policy boundaries: privacy, security, or local-only routes.
Those decisions should make their base selection final even if an ongoing
session previously used another model.

## When to Use

- Sensitive traffic must stay on a local model.
- You want to measure learning behavior before enforcing it.
- A decision has a non-negotiable backend boundary.
- A small number of decisions need stricter scope or tuning than the global
  adaptation default.

## Configuration

Default behavior is `apply`, so most decisions need no local block.

Use `bypass` for hard boundaries:

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      modelRefs:
        - model: local-private-model
      adaptations:
        session_aware:
          mode: bypass
```

Use `observe` for rollout:

```yaml
adaptations:
  bandit:
    mode: observe
```

`observe` records what the adaptation would have done, but the base decision
model remains final.

Use sparse local overrides only for exceptional routes:

```yaml
adaptations:
  session_aware:
    mode: apply
    scope: session
    tuning:
      switch_margin: 0.10
```

Unset fields inherit from
`global.router.learning.adaptations.session_aware`. The router validates
decision adaptation names and fields strictly: an unknown adaptation name or an
unknown adaptation field fails config validation instead of being ignored.
