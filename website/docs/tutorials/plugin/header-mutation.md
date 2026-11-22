# Header Mutation

## Overview

`header_mutation` is a route-local plugin for adding, updating, or deleting downstream headers.

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

Add the plugin under `recipes[].decisions[].plugins`:

```yaml
plugins:
  - type: header_mutation
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

Header values are static configuration, not templates. Do not use untrusted
request metadata to manufacture identity or authorization headers.

The plugin cannot write a header the Router owns: any `x-vsr-` or `x-envoy-`
header, and the credential and other internal headers of the
[header registry](../../api/headers), `Authorization` among
them. Each one fails at load with `reserved_header`. The retry and timeout
headers Envoy reads belong to the decision's `reliability` block. See a
complete example:
[`config/fragments/plugin/header-mutation/tenant-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/header-mutation/tenant-routing.yaml).
