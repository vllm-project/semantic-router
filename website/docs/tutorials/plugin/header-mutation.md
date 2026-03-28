# Header Mutation

## Overview

`header_mutation` is a route-local plugin for adding, updating, or deleting downstream headers.

It aligns to `config/plugin/header-mutation/tenant-routing.yaml`.

## Key Advantages

- Keeps downstream header policy local to the matched route.
- Supports add, update, and delete operations in one plugin.
- Useful for tenant routing, debugging, and downstream policy hints.

## What Problem Does It Solve?

Some routes need different downstream headers than the rest of the router. `header_mutation` makes that transformation explicit instead of hiding it in proxies or app code.

## When to Use

- one route should stamp tenant or plan metadata into headers
- downstream services expect route-specific headers
- debug or provenance headers should be added only for selected traffic

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: header_mutation
  configuration:
    add:
      - name: X-Tenant-Tier
        value: premium
    update:
      - name: X-Route-Source
        value: semantic-router
    delete:
      - X-Debug-Trace
```
