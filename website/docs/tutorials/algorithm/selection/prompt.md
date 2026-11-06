# Prompt (removed)

## Overview

The `prompt` selector was removed. The [System One](./systemone) selector asks
the decision model the same question.

## What Problem Does It Solve?

It let a helper model choose one of the decision's candidates by following the
operator's instructions. The `systemone` selector puts the same instructions to
the decision model as a typed choice question over the candidates.

## When to Use

Do not configure it: a configuration that still names it fails at load. Use
[System One](./systemone).

## Configuration

Run `vllm-sr config migrate` to rewrite it. The instructions are kept,
`timeout_seconds` becomes `timeout_ms`, and the helper model no longer chooses:
set `systemone.deployment` to ask a System One deployment other than the
decision model.

```yaml
algorithm:
  type: systemone
  systemone:
    instructions: Choose the model that best fits the request.
    timeout_ms: 5000
```
