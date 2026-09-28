# Content Safety

## Overview

Content Safety combines supported route-local safety plugins into one reusable
policy. It is a configuration bundle, not a separate plugin type.

## Key Advantages

- Reuses a consistent multi-plugin safety chain across routes.
- Keeps route-local safety readable even when several plugins are required.
- Makes the bundle explicit instead of scattering separate plugin snippets by hand.

## What Problem Does It Solve?

Some routes need more than one safety control at once. The bundle keeps
response screening, route-local guard prompts, and audit headers consistent
across those routes.

## When to Use

- a route needs several safety plugins together
- you want one reusable moderation chain for multiple routes
- the route should apply both route-local guidance and response-side screening

## Configuration

Add the bundled plugin entries under `routing.decisions[].plugins`:

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

This is a composition example, not a `content_safety` plugin type. The
`system_prompt` adds request-side guidance, `header_mutation` adds a policy
label, and `response_jailbreak` evaluates the generated response. The bundle
does not run a request-side content classifier, and the header is not proof
that content is safe. Calibrate response screening and decide whether
header-only handling is sufficient.

See the complete bundle:
[`config/fragments/plugin/content-safety/hybrid.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/content-safety/hybrid.yaml).
