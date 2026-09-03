# System Prompt

## Overview

`system_prompt` is a route-local plugin for inserting or modifying the system prompt on matched traffic.

## Key Advantages

- Keeps instruction shaping local to the route.
- Makes prompt mode explicit instead of hiding it in application code.
- Works well for expert, persona, or workflow-specific routes.

## What Problem Does It Solve?

Some routes need a different instruction layer than the router default. `system_prompt` lets those routes attach the extra prompt context without affecting unrelated traffic.

## When to Use

- one route needs an expert or persona-specific instruction layer
- prompt insertion should happen after the decision matches
- prompt policy should stay visible in the route config

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: system_prompt
    configuration:
      enabled: true
      mode: insert
      system_prompt: You are a domain expert. Answer precisely and state tradeoffs.
```

The inserted text is sent to the selected model and can change cache identity
and model behavior. Keep secrets and untrusted caller text out of it. See a
complete example:
[`config/fragments/plugin/system-prompt/expert.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/system-prompt/expert.yaml).
