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
global:
  model_catalog:
    modules:
      complexity:
        prototype_scoring:
          enabled: true
          cluster_similarity_threshold: 0.9
          max_prototypes: 8
          best_weight: 0.75
          top_m: 2
          margin_threshold: 0.0
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

Use `complexity` with representative hard and easy examples so the learned boundary matches your real routing cost profile. `prototype_scoring` is a family-level config under `global.model_catalog.modules.complexity`, so every complexity rule shares the same prototype-bank construction and label-scoring policy. Each rule still builds separate hard and easy prototype banks before computing the hard-vs-easy margin.

## Routing on a complexity rule

A complexity rule classifies each request into one of three difficulty levels — `hard`, `easy`, or `medium` — and emits the match as `<rule>:<level>`. Decision conditions must therefore reference the rule **with the difficulty suffix**, not by the bare rule name:

```yaml
routing:
  decisions:
    - name: escalate_hard_prompts
      priority: 160
      rules:
        operator: OR
        conditions:
          - type: complexity
            name: needs_reasoning:hard # required form: <rule>:<hard|easy|medium>
      modelRefs:
        - model: qwen3-30b
          use_reasoning: true
```

A bare `name: needs_reasoning` never matches at runtime — the classifier only ever emits `needs_reasoning:hard|easy|medium` — so config validation rejects the suffix-less form with a clear error.

## GPT-5.6 cost-tier example

GPT-5.6 model cards use the same open model-card contract as other providers; no router code or model registry entry is required. Configure the explicit Luna, Terra, and Sol model IDs so complexity decisions can select a stable cost tier:

```yaml
providers:
  defaults:
    default_model: gpt-5.6-luna
  models:
    - name: gpt-5.6-luna
      provider_model_id: gpt-5.6-luna
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 1.0
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 6.0
    - name: gpt-5.6-terra
      provider_model_id: gpt-5.6-terra
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 2.5
        cached_input_per_1m: 0.25
        cache_write_per_1m: 3.125
        completion_per_1m: 15.0
    - name: gpt-5.6-sol
      provider_model_id: gpt-5.6-sol
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 5.0
        cached_input_per_1m: 0.5
        cache_write_per_1m: 6.25
        completion_per_1m: 30.0

routing:
  modelCards:
    - name: gpt-5.6-luna
      context_window_size: 1050000
      description: Cost-efficient GPT-5.6 model for routine requests.
      capabilities: [chat, coding, reasoning, tools, long-context]
      modality: ar
      tags: [gpt-5.6, low-cost]
    - name: gpt-5.6-terra
      context_window_size: 1050000
      description: Balanced GPT-5.6 model for moderately complex requests.
      capabilities: [chat, coding, reasoning, tools, long-context]
      modality: ar
      tags: [gpt-5.6, balanced]
    - name: gpt-5.6-sol
      context_window_size: 1050000
      description: Highest-capability GPT-5.6 model for complex requests.
      capabilities: [chat, coding, reasoning, tools, long-context]
      modality: ar
      tags: [gpt-5.6, premium]
  signals:
    complexity:
      - name: request_complexity
        threshold: 0.70
        hard:
          candidates:
            - compare architecture tradeoffs under multiple failure modes
            - debug a subtle distributed systems incident
            - synthesize research into a technical recommendation
        easy:
          candidates:
            - write a small helper function
            - summarize this text briefly
            - update a simple configuration example
  decisions:
    - name: gpt_5_6_hard
      priority: 300
      rules:
        operator: AND
        conditions:
          - type: complexity
            name: request_complexity:hard
      modelRefs:
        - model: gpt-5.6-sol
    - name: gpt_5_6_medium
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: complexity
            name: request_complexity:medium
      modelRefs:
        - model: gpt-5.6-terra
    - name: gpt_5_6_easy
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: complexity
            name: request_complexity:easy
      modelRefs:
        - model: gpt-5.6-luna
```

Retain the `backend_refs` required by your deployment, such as an agentgateway endpoint. These are OpenAI's standard text-token rates from the [GPT-5.6 model page](https://openai.com/index/gpt-5-6/). The current pricing contract stores one flat rate per token class and does not model context-dependent pricing tiers.
