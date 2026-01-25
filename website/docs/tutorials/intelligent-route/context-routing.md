---
sidebar_position: 8
---

# Context Routing Tutorial

This tutorial shows you how to use **Context Signals** (Token Count) to route requests based on their length.

This is useful for:

- Routing short queries to faster, smaller models
- Routing long documents/prompts to models with large context windows
- Optimizing cost by using cheaper models for short tasks

## Scenario

We want to:

1. Route short requests (< 4K tokens) to a fast model (`llama-3-8b`)
2. Route medium requests (4K - 32K tokens) to a standard model (`llama-3-70b`)
3. Route long requests (32K - 128K tokens) to a large-context model (`claude-3-opus`)

## Step 1: Define Context Signals

Add `context_rules` to your `signals` configuration:

```yaml
signals:
  context:
    - name: "short_context"
      min_tokens: "0"
      max_tokens: "4K"
      description: "Short queries suitable for fast models"

    - name: "medium_context"
      min_tokens: "4K"
      max_tokens: "32K"
      description: "Medium length context"

    - name: "long_context"
      min_tokens: "32K"
      max_tokens: "128K"
      description: "Long context requiring specialized handling"
```

## Step 2: Define Decisions

Create decisions that trigger based on these context signals:

```yaml
decisions:
  - name: "fast_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "context"
          name: "short_context"
    modelRefs:
      - model: "llama-3-8b"

  - name: "standard_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "context"
          name: "medium_context"
    modelRefs:
      - model: "llama-3-70b"

  - name: "long_context_route"
    priority: 10
    rules:
      operator: "AND"
      conditions:
        - type: "context"
          name: "long_context"
    modelRefs:
      - model: "claude-3-opus"
```

## Step 3: Combined Logic (Advanced)

You can combine context signals with other signals (like domain or keyword).

**Example**: Route long **coding** tasks to a specialized long-context coding model:

```yaml
decisions:
  - name: "long_code_analysis"
    priority: 20  # Higher priority
    rules:
      operator: "AND"
      conditions:
        - type: "context"
          name: "long_context"
        - type: "domain"
          name: "computer_science"
    modelRefs:
      - model: "deepseek-coder-v2"
```

## How Token Counting Works

- The router counts tokens **before** making a routing decision.
- It uses a fast tokenizer compatible with most LLMs.
- Suffixes like "K" (1000) and "M" (1,000,000) are supported for readability.
- If a request matches multiple ranges (e.g., overlapping rules), all matching signals are active.

## Monitoring

You can monitor token distribution using the Prometheus metric:
`llm_context_token_count`

This helps you tune your ranges based on actual traffic patterns.
