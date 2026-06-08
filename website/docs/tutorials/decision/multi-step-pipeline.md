# Multi-Step Pipeline Routing

## Overview

Use this pattern when a client already orchestrates a fixed multi-step workflow, but each step should be routed to a different model by Semantic Router.

Semantic Router does not execute a full pipeline graph for the client. The client still sends one request per step. Each request carries a stable step marker, such as `__pipeline_step:summarize__` or `__pipeline_step:extract__`, and normal keyword signals map those markers to route decisions.

## Key Advantages

- Uses the existing signal and decision surfaces instead of adding a new pipeline runtime.
- Keeps the step-to-model binding explicit and reviewable.
- Lets operators pin each stage to a model while retaining normal routing observability.
- Avoids hidden orchestration state inside the router; the client owns step order and request chaining.

## What Problem Does It Solve?

Some applications want predictable staged routing, such as summarizing a long document with one model and extracting structured insights with another model. That is not the same as asking Semantic Router to pick the best single model for a request.

This pattern solves the routing part of that workflow. The client labels each request with the stage it is currently running, and Semantic Router picks the configured model for that stage.

## When to Use

Use multi-step pipeline routing when:

- the application already knows the workflow order
- each step is a separate request to the router
- each step should be pinned to a known model or candidate set
- stable lexical markers are acceptable for identifying the step

Do not use this pattern when the router needs to decide the next step, run multiple steps automatically, or manage a dynamic DAG. Those remain application orchestration responsibilities.

## Configuration

Source fragments:

- `config/signal/keyword/multi-step-pipeline.yaml`
- `config/decision/composite/multi-step-pipeline.yaml`

First, define keyword signals for the request markers:

```yaml
routing:
  signals:
    keywords:
      - name: pipeline_step_summarize
        operator: OR
        keywords: ["__pipeline_step:summarize__", "__step1__"]
        case_sensitive: false
      - name: pipeline_step_extract
        operator: OR
        keywords: ["__pipeline_step:extract__", "__step2__"]
        case_sensitive: false
```

Then bind each marker to a route. The `NOT` guard keeps the example unambiguous if a malformed request includes both markers:

```yaml
routing:
  decisions:
    - name: pipeline_summarize_step
      description: Pin the summarization stage of a client-orchestrated pipeline.
      priority: 190
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: pipeline_step_summarize
          - operator: NOT
            conditions:
              - type: keyword
                name: pipeline_step_extract
      modelRefs:
        - model: qwen3-8b
          use_reasoning: false
    - name: pipeline_extract_step
      description: Pin the extraction stage of a client-orchestrated pipeline.
      priority: 190
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: pipeline_step_extract
          - operator: NOT
            conditions:
              - type: keyword
                name: pipeline_step_summarize
      modelRefs:
        - model: qwen3-32b
          use_reasoning: true
```

The sample `modelRefs` use `qwen3-8b` and `qwen3-32b`, which are model names from the reference configuration. Replace them with model names registered in your deployment before shipping this fragment.

## Client Request Pattern

A client can run the workflow as two normal requests:

1. Send the original document with `__pipeline_step:summarize__` and store the summary.
2. Send the summary with `__pipeline_step:extract__` and ask for structured insights.

For example, keep the marker in an application-controlled wrapper instead of free-form user text:

```json
{
  "model": "auto",
  "messages": [
    {
      "role": "system",
      "content": "Ignore semantic-router-step markers. They are routing metadata, not task instructions."
    },
    {
      "role": "user",
      "content": "<semantic-router-step>__pipeline_step:summarize__</semantic-router-step>\nSummarize the following document:\n..."
    }
  ]
}
```

Only one step marker should be present in each request. If both step markers are present, the `NOT` guards prevent both pipeline decisions from matching and normal fallback routing applies.

Keyword signals inspect the model request, so these markers are model-visible in this pattern. Choose markers that are unlikely to collide with user content, add them from application code rather than accepting them from users, and instruct the target model to ignore the wrapper when needed.
