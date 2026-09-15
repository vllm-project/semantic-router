# Knowledge Routing Recipe Model Card

## Overview

Knowledge Routing uses measured domain evidence to decide whether a question
should stay on a small local model or move to a larger frontier model. The
included knowledge base is seeded from MMLU domain results, but the policy is
designed to work with an organization's own benchmarks and feedback.

## Model details

| Lane | Reference alias | Purpose |
| --- | --- | --- |
| Local | `local/small-7b` | Lower-cost default for domains with little measured uplift. |
| Frontier | `cloud/frontier-72b` | Escalation lane for domains with meaningful measured uplift. |
| Visual | `local/omni` | Dedicated image-understanding lane. |

Clients use `vllm-sr/auto`. Both aliases are examples and should be rebound to
the models represented by the deployment's own evidence.

## Intended use

Use this recipe when:

- model selection should be justified by benchmark or feedback data;
- a small model handles many domains well enough;
- frontier capacity should be reserved for domains with measurable benefit; or
- operators need an explainable knowledge label behind each escalation.
- image-bearing questions require a model with explicit visual capability.

It is not suitable when the knowledge base is stale, unrelated to production
traffic, or too small to support the routing decision.

## Routing behavior

| Evidence | Route | Behavior |
| --- | --- | --- |
| Image content | `omni` | Send the request to the visual-language alias. |
| High-uplift knowledge label | `escalate_72b` | Send the request to the frontier alias. |
| Lower-uplift or unmatched label | `keep_7b` | Keep the request on the local alias. |
| Known semantic boundary ambiguity | Deterministic guard | Correct a small set of terms that otherwise map to a neighboring label. |

The deterministic guards handle known boundary cases; they do not replace the
knowledge-base score.

## Requirements

- Reachable OpenAI-compatible endpoints for the local, frontier, and visual aliases.
- The versioned `mmlu_kb` asset referenced by [`config.yaml`](config.yaml).
- Production benchmark or feedback data before using the policy for a
  materially different workload.

## Data handling and safety

The checked-in recipe does not enable Router Replay or response caching. It
uses a local knowledge-base asset to choose a route, then sends the request to
one selected provider. The knowledge base contains routing evidence rather
than user documents.

Provider-side logging and retention still apply. Replace the seed knowledge
base carefully if an organization's benchmark or feedback data is sensitive.

## Quick start

```bash
vllm-sr serve
```

In the Dashboard, connect the physical Models, choose **Knowledge Routing** in
**Recipes**, assign the local and frontier decisions, and publish the resulting
Mixture-of-Model Entrypoint.

## Evaluation

The probes cover each escalation and local-domain family, the fallback metric,
and the small set of boundary guards. They validate KB lookup and route
selection without generating model responses. See [`probes.yaml`](probes.yaml)
and the [conformance guide](../CONFORMANCE.md).

## Limitations

- MMLU is only seed evidence and may not represent a production workload.
- Benchmark uplift can change after model, prompt, or serving updates.
- A domain label does not guarantee factual correctness.
- The reference prices and aliases are examples, not billing or performance
  claims.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
