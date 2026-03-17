# Jailbreak Signal

## Overview

`jailbreak` detects prompt-injection and jailbreak attempts before the router commits to a route. It maps to `config/signal/jailbreak/` and is declared under `routing.signals.jailbreak`.

This family is learned: it uses `global.model_catalog.modules.prompt_guard` and the router-owned jailbreak model bindings in `global.model_catalog.system`.

## Key Advantages

- Blocks or downgrades unsafe traffic before model selection.
- Supports classifier, contrastive, and hybrid-style safety detection.
- Keeps jailbreak policy visible inside routing decisions.
- Reuses one safety signal across multiple guarded routes.

## What Problem Does It Solve?

If jailbreak detection only happens downstream, the router can still send unsafe traffic to the wrong model or toolchain. If it lives outside the routing graph, safety logic becomes harder to audit.

`jailbreak` solves that by making injection detection a first-class routing input.

## When to Use

Use `jailbreak` when:

- unsafe traffic must be blocked before model selection
- prompt-injection attempts should route to a safer fallback
- multi-turn history should influence routing
- safety policy must be visible and testable in the same graph as routing logic

## Configuration

Source fragment family: `config/signal/jailbreak/`

```yaml
routing:
  signals:
    jailbreak:
      - name: prompt_injection
        method: hybrid
        threshold: 0.8
        include_history: true
        description: Detect common prompt-injection or jailbreak attempts.
        jailbreak_patterns:
          - ignore previous instructions
          - reveal the hidden prompt
          - jailbreak mode
        benign_patterns:
          - explain the policy
          - summarize the safety rules
```

Use `include_history` for multi-turn attacks, and treat the pattern lists as tuning data for the configured detection method.
