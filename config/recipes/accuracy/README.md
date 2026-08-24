# Accuracy Routing Recipe Model Card

## Overview

Accuracy Routing uses multi-model orchestration only when a request is likely
to benefit from it. Ordinary questions and long-context requests stay on a
single backend, while explicit planning, evidence gathering, and adversarial
review can use bounded workflow or fusion strategies.

This is a routing recipe, not a model checkpoint. It exposes the configured
`vllm-sr/auto` entrypoint and expects its provider backends to be available
before serving.

## Model details

| Item | Value |
| --- | --- |
| Request-facing model | `vllm-sr/auto` |
| Coordinator | `qwen-coordinator` |
| Review workers | `opus48-worker`, `gemini31-worker`, `gpt55-worker` |
| Default behavior | One direct request to `gpt55-worker` |
| Orchestration styles | Dynamic workflow and fusion |

The worker names are configuration aliases. Replace their provider bindings
when adapting the recipe to another model pool.

## Intended use

Good fits include:

- research or implementation tasks that explicitly ask for a plan, tools, or
  evidence gathering;
- reviews that benefit from independent hypotheses or adversarial checking;
- long documents that need a strong single-model answer without fan-out; and
- general requests where a direct high-quality answer is sufficient.

This recipe is not intended for latency-first or cost-first traffic. Use a
simpler recipe when every request must have predictable single-call cost.

## Routing behavior

| Request pattern | Route | Behavior |
| --- | --- | --- |
| Explicit planning, tools, or evidence gathering | `accuracy_workflow` | A coordinator creates a bounded workflow and uses up to three workers. |
| Long context | `accuracy_long_context_direct` | One long-context worker answers directly. |
| Competing hypotheses or adversarial review | `accuracy_deliberation` | Independent worker responses are fused into one answer. |
| Everything else | `accuracy_direct` | One worker answers directly. |

Workflow takes precedence over long-context handling, which takes precedence
over deliberation and the direct fallback. Orchestrated routes tolerate one
worker failure as long as enough workers remain to complete the request.

## Requirements

- An OpenAI-compatible endpoint for the local coordinator.
- An `OPENROUTER_API_KEY` for the three reference review workers, or replacement
  bindings to equivalent backends.
- Enough provider capacity for up to three concurrent worker calls on
  orchestrated routes.
- The Looper integration configured in [`config.yaml`](config.yaml).

Keep credentials outside the recipe and pass them by environment variable.

## Data handling and safety

The checked-in recipe does not enable Router Replay or response caching.
Direct routes send the request to one provider. Workflow and Fusion routes can
send request content or derived task prompts to the coordinator and several
review workers, so one user request may cross multiple provider boundaries.

Use only providers approved for the request's data. Provider-side logging and
retention remain outside the Router's control.

## Quick start

```bash
vllm-sr validate --config config/recipes/accuracy/config.yaml
vllm-sr serve --config config/recipes/accuracy/config.yaml \
  --recipe-env OPENROUTER_API_KEY
```

## Evaluation

The maintained probes cover direct, long-context, workflow, and deliberation
routes, including priority collisions. They evaluate routing decisions without
calling the configured generation backends. See [`probes.yaml`](probes.yaml)
for the scenarios and the [conformance guide](../CONFORMANCE.md) for the public
evaluation workflow.

## Limitations

- Orchestration increases latency and token use compared with a direct call.
- Route evaluation does not prove that a provider can generate successfully;
  verify each configured backend separately.
- The reference worker IDs and limits are deployment choices, not universal
  recommendations.
- Long-context admission is ultimately enforced by the selected backend's
  tokenizer and configured context limit.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
