# Plugin

## Overview

Use plugins when a matched decision needs extra route-local behavior after model selection.

In canonical v0.3 YAML, plugins live under `routing.decisions[].plugins`.

## Key Advantages

- Keeps route-local behavior attached to the route that needs it.
- Avoids pushing all behavior into `global:` defaults.
- Lets one route opt into caching, mutation, retrieval, or safety controls without affecting others.
- Maps directly to the fragment tree under `config/plugin/`, with one tutorial page per plugin or plugin bundle.

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

The plugin docs now mirror `config/plugin/` one page at a time.

### Response and Mutation

- [Fast Response](./fast-response)
- [Header Mutation](./header-mutation)
- [Image Generation](./image-gen)
- [System Prompt](./system-prompt)
- [Tools](./tools)

### Retrieval and Memory

- [Memory](./memory)
- [RAG](./rag)
- [Router Replay](./router-replay)
- [Semantic Cache](./semantic-cache)

### Safety and Generation

- [Content Safety](./content-safety)
- [Hallucination](./hallucination)
- [Jailbreak](./jailbreak)
- [PII](./pii)
- [Response Jailbreak](./response-jailbreak)
