# Content Safety

## Overview

`content-safety` is a reusable route-local safety bundle that combines multiple safety plugins in one fragment.

It aligns to `config/plugin/content-safety/hybrid.yaml`.

## Key Advantages

- Reuses a consistent multi-plugin safety chain across routes.
- Keeps route-local safety readable even when several plugins are required.
- Makes the bundle explicit instead of scattering separate plugin snippets by hand.

## What Problem Does It Solve?

Some routes need more than one safety control at once. Instead of repeatedly hand-writing jailbreak, PII, and response screening plugins together, `content-safety` packages that chain into one reusable fragment.

## When to Use

- a route needs several safety plugins together
- you want one reusable moderation chain for multiple routes
- the route should apply both request-side and response-side checks

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: jailbreak
    configuration:
      enabled: true
      threshold: 0.6
  - type: pii
    configuration:
      enabled: true
      pii_types_allowed: []
  - type: response_jailbreak
    configuration:
      enabled: true
      threshold: 0.8
      action: annotate
```
