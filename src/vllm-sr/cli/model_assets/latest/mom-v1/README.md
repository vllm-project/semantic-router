# vLLM-SR MoM V1 Model Card

*Many models. One intelligence.*

## Overview

vLLM-SR MoM V1 exposes five stable virtual models over six complementary
OpenAI-compatible backends. Each entrypoint applies a distinct objective while
keeping physical model placement and checkpoint details out of client code.

MoM is a routing policy, not a model installer. Start or bind the provider
services before serving these entrypoints.

## Model details

| Public model ID | Best for | Decisions |
| --- | --- | --- |
| `vllm-sr/mom-v1-blend` | General-purpose traffic across quality, latency, cost, and workload complexity. | `simple`, `medium`, `complex`, `agentic` |
| `vllm-sr/mom-v1-flash` | Interactive applications, tools, and visual requests. | `instant`, `heavy`, `visual`, `tooling`, `extended` |
| `vllm-sr/mom-v1-lite` | High-volume traffic with bounded spend on harder work. | `economy`, `reasoning`, `visual`, `extended` |
| `vllm-sr/mom-v1-ultra` | Verification, expert synthesis, and bounded orchestration. | `direct`, `verify`, `experts`, `orchestrate`, `extended`, `resume` |
| `vllm-sr/mom-v1-vault` | Sensitive workloads that must remain on the local pool. | `private`, `restricted_tools`, `containment`, `sensitive` |

The public IDs are stable within MoM V1. Decision and provider assignments can
evolve with a new policy version.

## Intended use

MoM V1 is designed for applications that want one stable API across mixed
interactive, reasoning, coding, multimodal, private, and long-context traffic.
It is most useful when clients should select a behavior rather than a physical
checkpoint and when operators can validate a co-designed provider pool.

It is not a substitute for provisioning, monitoring, network isolation, or
backend safety controls.

## Reference model pool

| Provider alias | Co-designed role | Configured context limit |
| --- | --- | ---: |
| `local/qwen3.6-35b-flash` | Default low-latency chat, tools, structured output, and vision | 262,144 |
| `local/gemma4-26b-balanced` | Architecture-diverse multilingual and multimodal balance | 131,072 |
| `local/qwen3.6-27b-coder` | Coding, tool use, planning, and structured agentic work | 262,144 |
| `local/qwen3.5-122b-frontier` | Local frontier synthesis, review, privacy, and vision | 262,144 |
| `local/deepseek-v4-flash-analyst` | Independent text analysis, code, and long-context review | 262,144 |
| `remote/glm-5.2` | Remote frontier synthesis, judging, and terminal text context | 524,288 |

The `local/` and `remote/` prefixes describe placement contracts, not
checkpoint vendors. Operators may bind equivalent backends when their
capabilities, context limits, API dialects, and tool behavior match the card.

## Routing behavior

Blend maps ordinary work to the fast local pair, multimodal and conversational
work to a broader local pool, difficult text synthesis to the remote frontier,
and tool-driven work to coding and analysis specialists.

Flash chooses from latency-aware local pools and reserves the remote frontier
for text beyond the conservative 240K boundary. Lite weights cost most heavily
while retaining dedicated visual, reasoning, and extended-context lanes.

Ultra uses direct local frontier inference by default. Explicit workflow,
expert-panel, or verification intent activates bounded Workflow, Fusion, or
Confidence algorithms. A completed tool turn uses the `resume` lane so the
router does not open another tool or multi-model loop.

Vault never references the remote model. Every Vault decision disables client
tools, removes prior tool history, and disables replay capture. Suspicious,
sensitive, multimodal, and long-context requests use the strongest local
boundary.

## Requirements

Every provider must expose its configured alias through an OpenAI-compatible
API. Tool-capable backends must support automatic tool choice and return
OpenAI-compatible tool calls. Ultra confidence escalation additionally requires
token log probabilities from participating backends.

The policy uses semantic embedding, PII, jailbreak, fact-check, feedback,
language, conversation-shape, structure, and privacy knowledge-base signals.
Ultra orchestration requires the Looper integration.

Configured context values are operating limits for the reference pool, not
claims about checkpoint architectural maxima. The selected backend remains
responsible for tokenizer-specific validation.

## Data handling and safety

The Vault entrypoint provides the strictest built-in data boundary: it keeps
all inference on models marked local, removes callable tools and prior tool
history, and disables replay capture on every route. Other entrypoints may use
the remote frontier model and the globally configured replay policy. Operators
must align those bindings, stores, logs, and retention settings with their own
data-handling requirements.

No built-in decision injects a system prompt. Application instructions remain
owned by the caller.

## Quick start

Inspect the installed card:

```bash
vllm-sr model show vllm-sr/mom-v1-blend
```

After the provider services are reachable:

```bash
vllm-sr serve vllm-sr/mom-v1-blend
vllm-sr serve vllm-sr/mom-v1-lite vllm-sr/mom-v1-flash
```

Fork before changing provider bindings or routing policy:

```bash
vllm-sr model fork vllm-sr/mom-v1-blend mom-v1.yaml
vllm-sr model validate mom-v1.yaml
vllm-sr serve --config mom-v1.yaml
```

## Evaluation

[`probes.yaml`](probes.yaml) covers every decision across multilingual
paraphrases, benign negatives, priority collisions, tool and image shapes,
multi-turn histories, privacy signals, and context boundaries. Probes validate
routing independently of backend answer quality; live generation checks are a
separate deployment gate.

See the [conformance guide](../../../CONFORMANCE.md) for contributor commands and
coverage rules.

## Limitations

- `vllm-sr serve` does not download or start the six provider models.
- Ultra orchestration can make multiple provider calls and has higher latency
  and compute cost than direct routing.
- Classifier and knowledge-base errors can affect selection.
- Cost coefficients are relative routing inputs, not prices or billing terms.
- Tool execution remains the client's responsibility.
- Operators must validate model quality, context capacity, privacy boundaries,
  and hardware requirements for their deployment.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Generated DSL projection](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Built-in model catalog](../catalog.yaml)
