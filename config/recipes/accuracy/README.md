# Accuracy Routing Recipe Model Card

## Overview

Accuracy Routing uses multi-model orchestration only when a request is likely
to benefit from it. In one call, the router's decision model judges the way of
working each request asks for (a direct answer, workstreams or an independent
review) and how much reasoning it needs. Ordinary questions and long-context
requests stay on a single backend, which the decision model picks, while
multi-step workstreams and reviews can use bounded workflow or fusion
strategies.

This is a routing recipe, not a model checkpoint. It exposes the configured
`vllm-sr/auto` entrypoint and expects its provider backends to be available
before serving.

## Model details

| Item | Value |
| --- | --- |
| Request-facing model | `vllm-sr/auto` |
| Workflow planner | `gpt55-worker` |
| Fusion coordinator | `qwen-coordinator` |
| Review workers | `opus48-worker`, `gemini31-worker`, `gpt55-worker` |
| Visual understanding | `local/omni` |
| Default behavior | One direct request to the worker the decision model picks |
| Orchestration styles | Dynamic workflow and fusion |
| Routing judgments | The decision model the recipe declares, vela3-lite (Vela 3.0 0.8B) |

The worker names are configuration aliases. Replace their provider bindings
when adapting the recipe to another model catalog.

## Intended use

Good fits include:

- research or implementation tasks that explicitly ask for a plan, tools, or
  evidence gathering;
- reviews that benefit from independent hypotheses or adversarial checking;
- image-bearing requests that require visual understanding;
- long documents that need a strong single-model answer without fan-out; and
- general requests where a direct high-quality answer is sufficient.

This recipe is not intended for latency-first or cost-first traffic. Use a
simpler recipe when every request must have predictable single-call cost.

## Routing behavior

| Request pattern | Route | Behavior |
| --- | --- | --- |
| Image content | `omni` | One visual-language model handles the request directly. |
| Workstreams such as research, diagnosis or evidence gathering, with multi-step reasoning | `accuracy_workflow` | A coordinator creates a bounded workflow and uses up to three workers. |
| Long context | `accuracy_long_context_direct` | One long-context worker answers directly. |
| Independent opinions, a debate or an adversarial review, with multi-step reasoning | `accuracy_deliberation` | Independent worker responses are fused into one answer. |
| Everything else | `accuracy_direct` | The decision model picks one of the three workers from their descriptions, and that worker answers directly. |

The decision model answers two named questions about the latest user message
in one call:

| Question | Type | Asks | Decides |
| --- | --- | --- | --- |
| `accuracy_mode` | Choice of 3 | Which way of working does this request ask for: a single direct answer, several workstreams, or an independent review? | Whether a request may use a workflow (`workstreams`) or fusion (`independent_review`); the lane needs that answer with probability 0.6 or more. |
| `accuracy_difficulty` | Score from 0 to 4, at 1.25 | How much reasoning does a strong expert need to answer this request well? | Whether the work is hard enough to pay for orchestration. |

The `direct` label includes a request not to split the work, compare options
or use several models, so "answer the question directly" or "do not debate"
keeps the request on one model without a keyword list. Unknown answers keep
it on `accuracy_direct`. The direct lane asks the decision model one more
question to pick its worker and uses `gpt55-worker` when that answer is late
or invalid. Long context is a request fact: inputs of 16K tokens or more take
the long-context lane unless they ask for workstreams.

Workflow takes precedence over long-context handling, which takes precedence
over deliberation and the direct fallback. Orchestrated routes tolerate one
worker failure as long as enough workers remain to complete the request.

## Requirements

- An OpenAI-compatible endpoint for the local fusion coordinator.
- An OpenAI-compatible visual-language endpoint for `local/omni`.
- An `OPENROUTER_API_KEY` for the three reference review workers, or replacement
  bindings to equivalent backends.
- Enough provider capacity for up to three concurrent worker calls on
  orchestrated routes.
- The Looper integration configured in [`config.yaml`](config.yaml).
- The decision model for the routing questions. The recipe declares it as the
  `primary` deployment of vela3-lite, which
  `global.model_catalog.system.decision_model` names. The deployment sets no
  device, so the model runtime uses a GPU when the router has one and the CPU
  otherwise. On a CPU a long latest message takes minutes: the questions read
  all of it, once per question, and with the router limited to 4 CPUs two
  55K-token messages took 13 minutes each and passed the decision deadline.
  In the CPU conformance run, with the router limited to 4 CPUs and 4
  requests in flight, the median request took 11.9 s and the 95th
  percentile 16.3 s; the long-context probes send their document in an
  earlier message.
  The worker pick may take up to 30 seconds (`timeout_ms: 30000`), set for a
  CPU; on a GPU it can be lowered, for example to 1000.
  Serve it on a GPU for live traffic. With no device set, the runtime uses a
  GPU when the router has one; set `device: rocm` or `device: cuda` on the
  `primary` deployment to require one. To use another decision model, change
  its `artifact` or point
  `global.model_catalog.system.decision_model.deployment` at another
  deployment, and rerun the probes, because the thresholds were calibrated on
  vela3-lite; `accuracy_difficulty` is a score question, which vela3-fast
  does not answer.

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
vllm-sr serve
```

In the Dashboard, connect approved physical Models, choose **Accuracy
Routing** in **Recipes**, assign the coordinator and review lanes, and publish
the resulting Mixture-of-Model Entrypoint. Provider credentials belong to the
Model resources rather than the Recipe or launch command.

## Evaluation

The maintained probes cover direct, long-context, workflow, and deliberation
routes, including priority collisions and a request not to fan out. They
evaluate routing decisions without calling the configured generation backends,
and CPU conformance runs them with the declared vela3-lite, which reaches
the expected decision for all 14. On that model the lane-deciding
`accuracy_mode` answers have probability 0.96 or more, and workflow probes
score 1.37 or more for `accuracy_difficulty` and review probes 1.61 or more.
A review request after a long document asks for independent opinions (0.96)
on hard work (1.98) and keeps its long-context lane by priority. See [`probes.yaml`](probes.yaml)
for the scenarios and the [conformance guide](../CONFORMANCE.md) for the public
evaluation workflow.

## Limitations

- Orchestration increases latency and token use compared with a direct call.
- Routing depends on the decision model's judgments. It reads the whole latest
  message, up to 262,144 tokens, and a long document in the same message
  weakens its answers wherever the request sits in it, so send a long
  document in an earlier message and the request in the latest one.
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
