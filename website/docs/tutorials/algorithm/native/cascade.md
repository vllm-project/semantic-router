# System One Cascade (removed)

## Overview

The native `cascade` algorithm was removed. Its behavior is the System One path
of [Escalate](../looper/escalate#on-system-one).

## What Problem Does It Solve?

It tried declared decision models in order and returned a complete System One
response when its acceptance rules passed. `escalate` with `escalate.stages`
does the same, with the same budget, acceptance evidence, calibration and
optional judge.

## When to Use

Do not configure it: a configuration that still names it fails at load. Use
[Escalate](../looper/escalate#on-system-one).

## Configuration

Run `vllm-sr config migrate` to rewrite it. A decision such as

```yaml
algorithm:
  type: cascade
  budget: {deadline: 10s, max_calls: 2}
  quality:
    type: uncalibrated
    acceptance:
      rules:
        - {question_type: choice, field: top_probability, predicate: {gte: 0}}
  stages:
    - {name: fast, kind: native, model: kai}
    - {name: strong, kind: native, model: vega}
```

becomes

```yaml
algorithm:
  type: escalate
  budget: {deadline: 10s, max_calls: 2}
  escalate:
    quality:
      type: uncalibrated
      accept:
        native:
          - {question_type: choice, field: top_probability, gte: 0}
    stages:
      - {name: fast, model: kai}
      - {name: strong, model: vega}
```
