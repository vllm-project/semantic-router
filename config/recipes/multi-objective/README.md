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

The reference configuration shares eight logical provider models across these
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
| Accuracy | Keeps ordinary work direct; explicit verification, expert comparison, multi-path reasoning, or workflows can use confidence, fusion, ReMoM, or Router Flow. |
| Privacy | Contains jailbreak and sensitive-data signals locally and defaults to a local model. |

Workflow owns tool interruption and resume. Fusion and confidence routes may
return a tool call only from the final selected model. Multi-response routes do
not attempt to merge independent tool trajectories.

## Requirements

- Reachable OpenAI-compatible endpoints for every provider alias used by the
  selected objectives.
- Live latency and load observations for objectives that use them.
- The embedding, feedback, fact-check, PII, jailbreak, language, complexity,
  and privacy-KB assets referenced by [`config.yaml`](config.yaml).
- Redis for the response API and startup-status service.
- Postgres for Router Replay. Local `vllm-sr serve` provisions managed Redis
  and Postgres services with development-only runtime defaults. Direct Router,
  Kubernetes, and production deployments must supply explicit connections and
  secret-managed credentials in a user-owned config.
- Looper support for orchestration routes.

## Data handling and safety

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
vllm-sr validate --config config/recipes/multi-objective/config.yaml
vllm-sr serve --config config/recipes/multi-objective/config.yaml
```

Clients then send one of the five virtual model IDs in the OpenAI-compatible
request body.

## Evaluation

The probe set covers every objective, route priority, multilingual and negative
cases, tool and multi-turn shapes, privacy signals, and long-input boundaries.
It validates recipe isolation and selection without requiring generation from
the provider pool. See [`probes.yaml`](probes.yaml) and the
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
