# System Prompt

## Overview

`system_prompt` is a route-local plugin for inserting or modifying the system prompt on matched traffic.

It aligns to `config/plugin/system-prompt/expert.yaml`.

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

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: system_prompt
  configuration:
    enabled: true
    mode: insert
    system_prompt: You are a domain expert. Answer precisely, state tradeoffs, and keep the response actionable.
```
