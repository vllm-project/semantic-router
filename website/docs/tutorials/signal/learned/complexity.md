# Complexity Signal

## Overview

`complexity` estimates whether a prompt needs a harder reasoning path or a cheaper easy path. It maps to `config/signal/complexity/` and is declared under `routing.signals.complexity`.

This family is learned: the classifier compares requests against hard and easy examples using embedding similarity, and can optionally use multimodal candidates.

## Key Advantages

- Separates reasoning escalation from domain classification.
- Reuses one complexity policy across multiple decisions.
- Supports hard/easy examples that are easy to tune over time.
- Lets simple prompts stay on cheaper models while hard prompts escalate.

## What Problem Does It Solve?

Topic alone does not tell you whether a prompt needs strong reasoning. Two questions in the same domain can have very different reasoning depth.

`complexity` solves that by estimating task difficulty directly from example-driven signal rules.

## When to Use

Use `complexity` when:

- some prompts need stronger reasoning or longer chains of thought
- easy traffic should stay on cheaper models
- you want escalation policies that are independent of domain
- multimodal reasoning requests need different handling from simple prompts

## Configuration

Source fragment family: `config/signal/complexity/`

```yaml
routing:
  signals:
    complexity:
      - name: needs_reasoning
        threshold: 0.75
        description: Escalate multi-step reasoning or synthesis-heavy prompts.
        hard:
          candidates:
            - solve this step by step
            - compare multiple tradeoffs
            - analyze the root cause
        easy:
          candidates:
            - answer briefly
            - quick summary
            - simple rewrite
```

Use `complexity` with representative hard and easy examples so the learned boundary matches your real routing cost profile.
