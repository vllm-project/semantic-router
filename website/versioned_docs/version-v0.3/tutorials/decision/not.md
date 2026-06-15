# NOT Decisions

## Overview

Use `config/decision/not/` when a route should match only if a risky or disallowed signal is absent.

`NOT` is the simplest exclusion rule in the decision catalog.

## Key Advantages

- Makes negative policy explicit.
- Works well for safety gates and premium-route exclusions.
- Keeps exclusion logic in the route graph instead of hidden downstream.
- Makes audits easier because the denied signal is named directly.

## What Problem Does It Solve?

Some routes should only run when a known risk signal is absent. If that exclusion is implicit, reviewers have to infer it from downstream behavior.

`NOT` solves that by putting the exclusion directly in the route definition.

## When to Use

Use `not/` when:

- known jailbreak or PII-bearing traffic must be excluded
- premium routes should stay away from unsafe inputs
- a conflicting signal must be absent before escalation

## Configuration

Source fragment: `config/decision/not/exclude-jailbreak.yaml`

```yaml
routing:
  decisions:
    - name: safe_only_route
      description: Match only when the known prompt-injection signal is absent.
      priority: 70
      rules:
        operator: NOT
        conditions:
          - type: jailbreak
            name: prompt_injection
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

Use `NOT` sparingly and keep the excluded signal explicit, otherwise the decision becomes hard to audit.
