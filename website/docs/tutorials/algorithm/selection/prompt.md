# Prompt Selection

## Overview

`prompt` uses a concrete helper model to select exactly one model from the
matched decision's `modelRefs`. The runtime owns the candidate list, structured
response schema, deterministic generation settings, and fallback behavior.

## Key Advantages

- expresses model-choice policy in plain instructions
- constrains the helper model to declared decision candidates
- reuses model-card descriptions and the existing selection registry
- falls back to the first valid candidate when the helper call fails

## What Problem Does It Solve?

Semantic and metrics-based selectors are not always the simplest way to encode
qualitative routing policy. Prompt selection lets a small model choose among a
bounded candidate set without moving request eligibility or safety logic out of
signals and decisions.

## When to Use

Use prompt selection when a decision has multiple valid candidates and the
choice depends on qualitative task requirements. Keep metadata, authorization,
privacy, and other deterministic gates in signals and decisions.

## Configuration

```yaml
routing:
  decisions:
    - name: adaptive-model-choice
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: general-small
          use_reasoning: false
        - model: reasoning-large
          use_reasoning: true
      algorithm:
        type: prompt
        on_error: fallback
        prompt:
          model: router-small
          instructions: >-
            Use general-small for ordinary requests. Use reasoning-large for
            hard reasoning, coding, debugging, or multi-step analysis.
          timeout_seconds: 5
```

`model` must be a concrete model declared in `routing.modelCards` and backed by
`providers.models`, and it must use an OpenAI-compatible API format. Candidate
base-model names must be unique; use separate decisions when LoRA or reasoning
variants share the same base model. Candidate names and available model-card
descriptions are added by the runtime. The selector receives the current user
turn and returns a fixed JSON object containing an exact candidate name and a
short rationale.
The internal helper call uses `global.integrations.looper.endpoint`, which must
address the router's OpenAI-compatible chat endpoint.

Model-generated rationale text is not logged or persisted verbatim. Replay
stores bounded result/fallback reason codes, and metrics expose selector
duration plus fallback counts without request content.
