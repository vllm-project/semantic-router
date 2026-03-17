# Fast Response

## Overview

`fast_response` is a route-local plugin that returns a deterministic fallback message immediately.

It aligns to `config/plugin/fast-response/busy.yaml`.

## Key Advantages

- Short-circuits expensive routes when a lightweight fallback is enough.
- Keeps overload behavior local to the route that needs it.
- Makes the fallback message explicit in config.

## What Problem Does It Solve?

Some routes should degrade gracefully instead of waiting for the full model path. `fast_response` gives those routes an immediate response path without changing global behavior.

## When to Use

- the route needs a cheap fallback under overload or maintenance conditions
- a deterministic response is acceptable for this traffic class
- fallback behavior should stay local to one route

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: fast_response
  configuration:
    message: The primary model is saturated, so a lightweight response was returned immediately.
```
