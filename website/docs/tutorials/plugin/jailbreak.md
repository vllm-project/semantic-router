# Jailbreak

## Overview

`jailbreak` is a route-local plugin for screening matched requests or outputs with a jailbreak threshold.

It aligns to `config/plugin/jailbreak/guard.yaml`.

## Key Advantages

- Adds route-specific jailbreak enforcement after a decision matches.
- Keeps thresholds local to the traffic that needs them.
- Complements global model loading without forcing global route behavior.

## What Problem Does It Solve?

Some routes need stricter jailbreak enforcement than others. `jailbreak` makes that post-match safety policy explicit for the routes that require it.

## When to Use

- one route needs stricter jailbreak controls than the rest of the router
- a matched route should block or annotate suspicious traffic
- you want route-local safety in addition to global model availability

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: jailbreak
  configuration:
    enabled: true
    threshold: 0.85
```
