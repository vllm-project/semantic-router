# Hallucination

## Overview

`hallucination` is a route-local plugin for fact-checking and response-quality screening after the decision already matched.

## Key Advantages

- Adds route-local hallucination checks without changing global defaults.
- Makes response actions explicit when factual confidence is low.
- Works well for retrieval-heavy or grounded-answer routes.

## What Problem Does It Solve?

Some routes need extra scrutiny after the model answers, especially when they promise factual precision. `hallucination` lets those routes add response-time verification without forcing every route to pay the cost.

## When to Use

- a route should fact-check or annotate responses
- grounded or tool-backed routes need extra response screening
- the route should warn or annotate instead of silently passing low-confidence answers

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: hallucination
    configuration:
      enabled: true
      use_nli: true
      hallucination_action: header
      unverified_factual_action: header
      include_hallucination_details: true
```

`header` preserves the model response and adds warning metadata. `body` adds a
warning to the response body, while `none` records the result without changing
the response.

The plugin depends on
`global.model_catalog.modules.hallucination_mitigation`; `use_nli: true` also
uses the configured explainer/NLI model. Model responses and supplied grounding
context are processed by those modules. Detection can identify unsupported
text, but it cannot establish truth without authoritative evidence.

See a complete example:
[`config/fragments/plugin/hallucination/fact-check.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/hallucination/fact-check.yaml).
