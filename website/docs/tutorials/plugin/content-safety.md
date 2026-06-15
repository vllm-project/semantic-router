# Content Safety

## Overview

`content-safety` is a reusable route-local safety bundle that combines supported safety-oriented plugins in one fragment.

It aligns to `config/plugin/content-safety/hybrid.yaml`.

## Key Advantages

- Reuses a consistent multi-plugin safety chain across routes.
- Keeps route-local safety readable even when several plugins are required.
- Makes the bundle explicit instead of scattering separate plugin snippets by hand.

## What Problem Does It Solve?

Some routes need more than one safety control at once. Instead of repeatedly hand-writing response screening, route-local guard prompts, and audit headers together, `content-safety` packages that chain into one reusable fragment.

## When to Use

- a route needs several safety plugins together
- you want one reusable moderation chain for multiple routes
- the route should apply both route-local guidance and response-side screening

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: system_prompt
    configuration:
      enabled: true
      mode: insert
      system_prompt: Apply the platform safety policy before answering and clearly note when a request needs additional review.
  - type: header_mutation
    configuration:
      add:
        - name: X-Safety-Profile
          value: standard
  - type: response_jailbreak
    configuration:
      enabled: true
      threshold: 0.8
      action: header
```
