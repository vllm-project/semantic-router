# Prompt Selection

## Overview

`prompt` selects exactly one Model from the matched decision's Entrypoint
assignment. The runtime owns the candidate list, structured response schema,
deterministic generation settings, and fallback behavior.

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
      description: Let a helper model choose the best eligible candidate.
      priority: 100
      rules:
        operator: AND
        conditions: []
      algorithm:
        type: prompt
        on_error: fallback
        prompt:
          instructions: >-
            Use general-small for ordinary requests. Use reasoning-large for
            hard reasoning, coding, debugging, or multi-step analysis.
          timeout_seconds: 5
```

The Entrypoint assigns the candidate Models to the decision. Candidate Model
IDs must be unique; LoRA and reasoning overrides stay on those assignments.
Candidate names and Model descriptions are added by the runtime. The selector
receives the current user turn and returns a fixed JSON object containing an
exact candidate name and a short rationale.
The internal helper call uses `global.integrations.looper.endpoint`, which must
address the router's OpenAI-compatible chat endpoint.

Model-generated rationale text is not logged or persisted verbatim. Replay
stores bounded result/fallback reason codes, and metrics expose selector
duration plus fallback counts without request content.

The helper model receives the current user turn and candidate descriptions, so
its provider must be allowed by the route's data policy. Its choice is bounded
to declared candidates but remains model-generated; deterministic privacy,
authorization, and safety gates belong in signals and decisions. See a
complete example:
[`config/fragments/algorithm/selection/prompt.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/prompt.yaml).
