# Hallucination

## Overview

`hallucination` is a route-local plugin for fact-checking and response-quality screening after the decision already matched.

It aligns to `config/plugin/hallucination/fact-check.yaml`.

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

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: hallucination
  configuration:
    enabled: true
    use_nli: true
    hallucination_action: annotate
    unverified_factual_action: warn
    include_hallucination_details: true
```
