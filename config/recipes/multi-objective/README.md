# Multi-Objective Routing Recipe Model Card

## Overview

Multi-Objective Routing exposes five request-facing virtual models over one
shared provider pool. Clients choose the objective they care about—balance,
speed, cost, accuracy, or privacy—while each objective keeps its own signals,
decisions, algorithms, and plugins.

The virtual model IDs select routing policies. They do not identify or start a
physical model server.

## Model details

| Virtual model | Objective | Main trade-off |
| --- | --- | --- |
| `vllm-sr/mom-v1-blend` | Adapt quality, speed, and efficiency to the request. | No single metric is always minimized. |
| `vllm-sr/mom-v1-flash` | Prefer low observed latency. | Heavy requests may still use a slower capable lane. |
| `vllm-sr/mom-v1-lite` | Keep work on economical local models. | Lower peak capability than frontier routing. |
| `vllm-sr/mom-v1-ultra` | Use direct frontier or bounded orchestration for accuracy. | Higher latency and compute use. |
| `vllm-sr/mom-v1-vault` | Keep sensitive and suspicious traffic local. | Reduced provider diversity and no cloud escalation. |

The reference configuration shares nine logical provider models across these
five policies. Operators can replace that pool while keeping the public
entrypoint names.

## Intended use

Use this recipe when:

- one deployment serves clients with different service objectives;
- provider capacity should be shared without mixing routing state;
- applications need stable model IDs instead of client-side policy flags; or
- operators want to compare objective-specific behavior on the same workload.

It is not a universal model benchmark. The outcome depends on the configured
provider pool, live health, latency, load, and cost metadata.

## Routing behavior

| Objective | High-level behavior |
| --- | --- |
| Balanced | Recovers from poor answers, spends more effort on difficult work, and otherwise weighs quality, latency, cost, and load. |
| Speed | Uses observed latency for interactive work and keeps a bounded capable lane for heavier requests. |
| Cost | Uses economical local replicas and enables bounded reasoning only when the request calls for it. |
| Accuracy | Keeps ordinary work direct; explicit verification, expert comparison, deep reasoning, or staged workflows can use confidence, fusion, ReMoM, or Router Flow. |
| Privacy | Contains attempts to override the assistant locally, keeps sensitive work on the local model, and defaults to a local model. |

Every objective routes on named questions to the router's decision model,
answered together in one call per request, and on facts read from the request
itself: an image, an active tool loop, the input length, and, for Balanced, a
repeated request.

| Objective | Question (threshold) | Decides |
| --- | --- | --- |
| Balanced | `correction` (0.6) | The user says an answer was wrong: the recovery lane. A repeated request also recovers; the built-in re-ask judgment compares the last two user turns. |
| Balanced | `deliberate` (0.3), `verification` (0.7) | Careful reasoning about trade-offs, a root cause or several exact constraints, or an explicit request to verify or cite: the deliberate lane. Inputs of 12K tokens or more take it too. |
| Speed | `heavy` (0.5) | A detailed design, a thorough review or a deep analysis: the heavy lane. Inputs of 16K tokens or more take it too. |
| Cost | `depth` (0.6) | A hard problem that needs investigation or careful multi-step reasoning: bounded local reasoning. Inputs of 16K tokens or more take it too. |
| Accuracy | `workflow` (0.7) | A task carried out in stages, such as research, implement and validate, or by several agents: Router Flow. A request for an expert panel does not count. |
| Accuracy | `mode` = `experts` (0.7) | Several independent models or experts answering the same question: fusion. |
| Accuracy | `verification` (0.9) | An explicit request to fact-check, verify or cite sources: the confidence route. |
| Accuracy | `depth` (0.5) | A hard problem that needs investigation or careful multi-step reasoning: ReMoM. |
| Privacy | `attack` (0.6) | The message tells the assistant to override its rules, reveal hidden instructions or secrets, or exfiltrate data: local containment with tools disabled. Quoting such a phrase for review does not count. |
| Privacy | `confidential` (0.5), `personal_data` (0.5) | Confidential or internal material, a request to keep the data local, or personal data about a specific person: the local sensitive lane. |

In Accuracy, text that the user gives to translate, quote, define or explain
(`quoted`, 0.5) never starts orchestration, and an active client tool loop is
synthesized directly unless the request is a staged workflow.

Every objective includes an `omni` decision. Image-bearing requests use the
shared local visual-language model; the privacy objective still gives attack
containment higher priority.

Workflow owns tool interruption and resume. Fusion and confidence routes may
return a tool call only from the final selected model. Multi-response routes do
not attempt to merge independent tool trajectories.

When a question goes unanswered, each objective falls back conservatively.
Balanced's deliberate lane, Speed's heavy lane and Cost's local reasoning
match an unanswered question, so a decision-model outage never lowers quality.
Accuracy's orchestration lanes need an answer, so the request stays on the
direct lane. Privacy's sensitive lane matches an unanswered question, so the
request stays on the local model with tools disabled. Preview lists an
unanswered question under `signal_errors`, and the router counts it in
`vsr_model_runtime_unknown_answers_total` by deployment and reason.

The questions read the start of the latest user message, up to the decision
model's input limit (16,384 tokens for Vela 2.0 0.8B): put the request before
a long attachment.

## Requirements

- Reachable OpenAI-compatible endpoints for every provider alias used by the
  selected objectives.
- Live latency and load observations for objectives that use them.
- The decision model for the questions. The recipe declares it as the
  `primary` deployment of `vllm-sr/Vela-2.0-0.8B`, which
  `global.model_catalog.system.decision_model` names. The deployment sets no
  device, so the model runtime uses a GPU when the router has one and the CPU
  otherwise. In the CPU conformance run, with the router limited to 4 CPUs
  and 8 requests in flight, the 129 probes took 584 s (0.22 requests per
  second); the median request took 28 s and the 95th percentile 101 s.
  Serve it on a GPU for live traffic: set `device: rocm` or `device: cuda` on
  the `primary` deployment to require one. For a smaller model, change its
  `artifact` (for example to `vllm-sr/Vela-2.0-0.3B`) or point
  `global.model_catalog.system.decision_model.deployment` at another
  deployment, and rerun the probes, because the thresholds were calibrated on
  the 0.8B.
- Redis for the response API and startup-status service.
- Postgres for Router Replay. Local `vllm-sr serve` provisions managed Redis
  and Postgres services with development-only runtime defaults. Direct Router,
  Kubernetes, and production deployments must supply explicit connections and
  secret-managed credentials in a user-owned config.
- Looper support for orchestration routes.

## Data handling and safety

The privacy objective's questions read the latest user message, so an attack
or personal data placed only in an earlier turn, or past the first 16,384
tokens, is not seen. Containment and the sensitive lane both stay local with
tools disabled, and an unanswered question keeps the request on the sensitive
lane. A city or country name alone is not personal data; an email or street
address with it is.

Each objective has isolated routing state, but providers and supporting stores
are shared infrastructure. The privacy objective keeps model traffic local and
disables inappropriate tool use; operators must still configure storage,
logging, and network boundaries accordingly.

The checked-in config enables Postgres-backed Router Replay for 30 days. Unless
a route overrides it, replay captures up to 4 KiB each from request and
response bodies. This also applies to the privacy objective: its provider path
is local, but its replay records use the shared store. Restrict and protect that
database, or disable replay or body capture when content must not be retained.

## Quick start

```bash
vllm-sr serve
```

In the Dashboard, connect the physical Models, choose **Multi-Objective
Routing** in **Recipes**, assign every objective lane, and publish the desired
Mixture-of-Model Entrypoints. Clients then send one of those published model
IDs in the OpenAI-compatible request body.

## Evaluation

The probe set covers every objective, route priority, multilingual and negative
cases, tool and multi-turn shapes, privacy signals, and long-input boundaries.
It validates recipe isolation and selection without requiring generation from
the provider pool. With the declared Vela 2.0 0.8B and the router limited to
4 CPUs, all 129 probes reach their expected decision. See [`probes.yaml`](probes.yaml) and the
[conformance guide](../CONFORMANCE.md).

## Limitations

- Virtual model names do not guarantee a fixed physical backend or latency.
- Multi-model orchestration can multiply latency and compute use.
- Cost and quality metadata must be recalibrated for each deployment.
- Tool execution remains the client's responsibility.
- The reference provider aliases and context limits are deployment examples,
  not architectural limits of the underlying checkpoints.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
