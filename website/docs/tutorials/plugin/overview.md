# Plugin

## Overview

Use plugins when a matched decision needs extra route-local behavior after model selection.

In canonical v0.3 YAML, plugins live under `routing.decisions[].plugins`.

## Key Advantages

- Keeps route-local behavior attached to the route that needs it.
- Avoids pushing all behavior into `global:` defaults.
- Lets one route opt into caching, mutation, retrieval, or safety controls without affecting others.
- Maps directly to the fragment tree under `config/plugin/`.

## What Problem Does It Solve?

Not every route needs the same post-selection behavior. Some need semantic cache, some need system-prompt mutation, some need route-local safety enforcement.

Plugins solve that by making route-local processing explicit instead of overloading global runtime settings.

## When to Use

Use `plugin/` when:

- only one route or route family needs extra processing
- behavior should happen after a route matches
- shared backing services live in `global:`, but per-route behavior must stay local
- you want reusable route-local fragments under `config/plugin/`

## Configuration

Canonical placement:

```yaml
routing:
  decisions:
    - name: cached_support
      plugins:
        - type: semantic-cache
          configuration:
            enabled: true
```

The plugin docs mirror `config/plugin/`:

| Plugin family | Fragment examples | Purpose | Doc |
|---------------|-------------------|---------|-----|
| response and mutation | `fast-response`, `header-mutation`, `system-prompt`, `image-gen` | change or short-circuit route output | [Response and Mutation](./response-and-mutation) |
| retrieval and memory | `semantic-cache`, `rag`, `memory`, `router-replay` | reuse context, retrieval, or state | [Retrieval and Memory](./retrieval-and-memory) |
| safety and generation | `content-safety`, `jailbreak`, `pii`, `hallucination`, `response-jailbreak` | apply route-local safety or generation controls | [Safety and Generation](./safety-and-generation) |
