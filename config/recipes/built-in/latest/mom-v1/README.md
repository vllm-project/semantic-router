# vLLM-SR MoM V1 Model Card

*Many models. One intelligence.*

## Overview

vLLM-SR MoM V1 is a family of five virtual models backed by a shared pool
of specialized, OpenAI-compatible inference services. Each public model ID
selects a different routing objective while hiding backend placement and
checkpoint details from clients.

MoM is a routing policy, not a checkpoint or a model installer. The Router,
Envoy, and Dashboard can be started by `vllm-sr`, but the seven inference
backends must already be running and reachable.

## Model details

| Public model ID | Best for | Main trade-off |
| --- | --- | --- |
| `vllm-sr/mom-v1-blend` | General-purpose multi-objective routing across quality, latency, cost, and answer recovery. | No single metric is always minimized. |
| `vllm-sr/mom-v1-lite` | Economical local responses with optional bounded reasoning. | Lower peak capability. |
| `vllm-sr/mom-v1-flash` | Interactive and tool-oriented traffic where latency matters most. | Heavy work may use a slower capable lane. |
| `vllm-sr/mom-v1-ultra` | High-accuracy answers, verification, expert comparison, and bounded orchestration. | Higher latency and compute use. |
| `vllm-sr/mom-v1-vault` | Sensitive, private, or suspicious workloads that must remain local. | Tools are disabled and provider choice is deliberately narrow. |

The public IDs are stable within MoM V1. Internal recipe names and backend
aliases may evolve with a new catalog version.

## Intended use

MoM V1 is a good fit when:

- one endpoint must serve several quality, cost, speed, and privacy profiles;
- clients should choose a stable virtual model instead of a physical backend;
- multimodal, tool-bearing, long-context, and multi-turn requests need explicit
  capability guards; or
- accuracy-sensitive tasks may benefit from bounded multi-model orchestration.

It is not a substitute for provisioning, monitoring, or securing the
underlying inference services. It also does not guarantee that the reference
model pool is optimal for another workload or hardware platform.

## Routing behavior

| Virtual model | High-level policy |
| --- | --- |
| MoM Blend | Protect modality and context limits first, recover after a weak answer, spend more effort on difficult or verification-heavy work, and otherwise balance quality, latency, cost, and load. |
| MoM Lite | Keep ordinary work on the economy model and enable bounded reasoning only when the request asks for it. |
| MoM Flash | Preserve tools and images, choose from low-latency candidates for ordinary work, and retain a capable lane for heavier requests. |
| MoM Ultra | Keep ordinary work direct; explicit planning, expert comparison, factual verification, or multi-path exploration can use Workflow, Fusion, Confidence, or ReMoM. |
| MoM Vault | Keep every request local, strip client tools and prior tool history, contain attacks, and use stronger local privacy handling when sensitive data is detected. |

Capability checks run before semantic routing. Images are kept on multimodal
backends, and long inputs move to backends with a larger configured context
window. The selected backend remains responsible for its tokenizer-specific
context limit.

## Requirements

The reference pool contains seven logical provider models:

| Provider alias | Reference role | Configured context limit |
| --- | --- | ---: |
| `local/qwen3.5-9b` | Economy and interactive traffic | 262,144 |
| `local/qwen3.6-35b` | Fast long-context, code, tools, and images | 262,144 |
| `local/step-3.7-flash` | Fast reasoning and multimodal analysis | 65,536 |
| `local/qwen3.5-122b` | Image understanding and high-quality synthesis | 131,072 |
| `local/mistral-small-4` | Diverse analysis and review | 131,072 |
| `local/gpt-oss-120b` | Local reasoning and security containment | 131,072 |
| `local/glm-5.2` | Text-only synthesis, judging, and very long context | 524,288 |

Every service must expose its configured alias through an OpenAI-compatible
API. The routing policy also uses semantic embedding, domain, PII, jailbreak,
fact-check, feedback, language, and privacy knowledge-base assets. Ultra
orchestration requires the Looper integration; confidence escalation requires
participating backends to return token log probabilities.

The context values above are operating limits chosen for the reference
deployment, not claims about each checkpoint's architectural maximum.

## Data handling and safety

MoM Vault keeps provider traffic on the local pool, removes callable tools
and prior tool history, and disables replay capture on every Vault route. The
other four profiles use Redis-backed Router Replay with a seven-day retention
period and capture up to 4 KiB each from request and response bodies. Operators
should review replay access and retention, or disable capture when application
content must not be stored.

“Local” describes the configured backend path. End-to-end privacy still
depends on network isolation, backend ownership, logs, caches, and supporting
stores.

## Quick start

Inspect the installed Model Card and backend requirements:

```bash
vllm-sr model show vllm-sr/mom-v1-blend
```

After starting or binding the required provider services, serve one or more
virtual models:

```bash
vllm-sr serve vllm-sr/mom-v1-blend
vllm-sr serve vllm-sr/mom-v1-lite vllm-sr/mom-v1-flash
```

To change the provider pool or routing policy, create a user-owned config:

```bash
vllm-sr model fork vllm-sr/mom-v1-blend mom-v1.yaml
vllm-sr model validate mom-v1.yaml
vllm-sr serve --config mom-v1.yaml
```

See the
[built-in model catalog](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/built-in/README.md)
for catalog versions and selection behavior.

## Evaluation

MoM V1 publishes versioned end-to-end evaluation scorecards under the
`vllm-sr/mom-evaluation/v1` contract. Decision-level routing probes remain in
[`probes.yaml`](probes.yaml); the scorecards below measure complete system
quality, operational behavior, baseline comparison, and objective-specific
extension packs.

Evaluation contract: `vllm-sr/mom-evaluation/v1`

See [evaluation-scorecard.md](evaluation-scorecard.md) for the aggregate launch
scorecard summary, baseline comparison, and known limitations.

[`probes.yaml`](probes.yaml) covers all five virtual models and their route
families. The scenarios include multilingual requests, negative and collision
cases, tools, images, multi-turn history, feedback recovery, privacy signals,
and context boundaries. They validate routing independently of backend answer
quality.

Per-entrypoint launch scorecards:

| Entrypoint | Scorecard |
| --- | --- |
| `vllm-sr/mom-v1-blend` | [blend/1.0.0](../../../evaluation/scorecards/mom-v1/mom-v1-blend/1.0.0/scorecard.md) |
| `vllm-sr/mom-v1-lite` | [lite/1.0.0](../../../evaluation/scorecards/mom-v1/mom-v1-lite/1.0.0/scorecard.md) |
| `vllm-sr/mom-v1-flash` | [flash/1.0.0](../../../evaluation/scorecards/mom-v1/mom-v1-flash/1.0.0/scorecard.md) |
| `vllm-sr/mom-v1-ultra` | [ultra/1.0.0](../../../evaluation/scorecards/mom-v1/mom-v1-ultra/1.0.0/scorecard.md) |
| `vllm-sr/mom-v1-vault` | [vault/1.0.0](../../../evaluation/scorecards/mom-v1/mom-v1-vault/1.0.0/scorecard.md) |

The checked-in image is a small synthetic fixture used only to exercise
multimodal request shape; probes do not grade its visual content. Large context
fixtures are generated in memory so the Model Card and probe source remain
readable. Contributor commands and coverage rules live in the
[conformance guide](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/CONFORMANCE.md).

## Limitations

- `vllm-sr serve` does not download or start the seven provider models.
- Ultra orchestration can make several provider calls and therefore has higher
  latency and compute cost than a direct route.
- Classifier or knowledge-base errors can affect route selection.
- Configured cost coefficients are relative routing inputs, not public prices
  or billing guarantees.
- Tool execution remains the client's responsibility; Workflow can pause and
  resume around tool results but does not run client tools.
- Model quality, context capacity, and hardware requirements must be verified
  for each deployment.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Generated routing projection](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Built-in model catalog](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/built-in/README.md)
- [Recipe authoring and conformance](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/CONFORMANCE.md)
