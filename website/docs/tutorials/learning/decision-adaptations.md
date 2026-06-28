# Decision Adaptations

## Overview

Decision adaptations let a matched decision control whether global Router
Learning can adjust its proposed model.

Most decisions inherit global learning behavior. Add `adaptations` only when a
decision needs a hard boundary, observe-only rollout, or small protection
tuning.

## Key Advantages

- Keeps policy boundaries close to the decision that owns them.
- Lets sensitive routes bypass learning with one small block.
- Supports observe-only rollouts before adaptation or protection can affect
  traffic.
- Lets one decision use a narrower or broader adaptation candidate set than the
  global default.
- Lets one decision tune the stability trade-off without changing global
  defaults.

## What Problem Does It Solve?

Global learning is convenient, but not every decision should be adjusted by
online state. Privacy, local-only, security, compliance, and operational routes
often need hard boundaries. Decision adaptations give the matched decision the
final say on whether learning may apply, observe, or bypass.

## When to Use

- A matched decision must not be changed by online learning.
- You want to compare learning diagnostics before allowing route changes.
- One decision should search the whole route tier while most decisions stay
  inside their own `modelRefs`.
- One decision needs a stronger or weaker protection margin than the default.
- Adaptation and protection need different modes for the same decision.

## Configuration

Use `bypass` for hard boundaries:

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      modelRefs:
        - model: local-private-model
      adaptations:
        mode: bypass
```

Use component-level controls when adaptation and protection should behave
differently:

```yaml
adaptations:
  adaptation:
    mode: observe
    candidate_set: tier
  protection:
    mode: apply
```

`adaptation.candidate_set` is optional. When omitted, the decision inherits
`global.router.learning.adaptation.candidate_set`.

Allowed modes:

| Mode | Meaning |
| --- | --- |
| `apply` | The component may affect the final route. |
| `observe` | The component records diagnostics but cannot change the final route. |
| `bypass` | The component does not adjust this decision. |

`adaptations.mode: bypass` overrides component-level modes and prevents both
adaptation and protection from changing the route.

## Protection Tuning

Use decision-local protection tuning only when a decision needs a different
stability trade-off than the global default:

```yaml
adaptations:
  protection:
    stability_weight: 1.5
    switch_margin: 0.10
```

Higher `protection.stability_weight` favors stability. Lower
`protection.stability_weight` makes it easier for adaptation to switch models.
`switch_margin` is the minimum model advantage required before switching for
this decision.
