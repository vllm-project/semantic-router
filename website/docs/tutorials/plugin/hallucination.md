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
      hallucination_action: header
      unverified_factual_action: header
      include_hallucination_details: true
```

`header` preserves the model response and adds warning metadata. `body` adds a
warning to the response body, while `none` records the result without changing
the response.

With a [hallucination signal](../signal/learned/hallucination) declared under
`routing.signals.hallucination`, detection runs as a response-stage signal and
this plugin only enforces on it: it applies `hallucination_action` when the rule
matched, `unverified_factual_action` when the answer had no grounding context
to be checked against, and nothing when the rule was unavailable or did not
apply. Without a rule the plugin classifies the answer itself, which is the
compatibility path and is reported at load as such.

The plugin depends on
`global.model_catalog.modules.hallucination_mitigation`, whose detector (Vela
2.0 0.3B by default, with Vela Halu available as a specialist) runs in the [model runtime](../../model-runtime/guides/hallucination.md).
Model responses and supplied grounding context are processed by that module.
NLI explanations (`use_nli`) are retired; `vllm-sr config migrate` removes the
setting. Input handling depends on the selected detector and deployment budget;
a general decision task requires complete admitted context and reports an
unavailable result if it cannot cover it. See the runtime guide before
choosing a truncation or window policy. Detection can identify unsupported text, but it cannot establish
truth without authoritative evidence.

See a complete example:
[`config/fragments/plugin/hallucination/fact-check.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/hallucination/fact-check.yaml).
