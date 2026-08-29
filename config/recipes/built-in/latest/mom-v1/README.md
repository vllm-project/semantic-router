# vLLM-SR MoM V1 Model Card

*Many models. One intelligence.*

## Overview

MoM V1 contains five reusable routing Recipes. Each Recipe turns request
signals into human-readable decisions while leaving Model selection to the
Entrypoint that uses it.

## Model details

| Recipe | Best for | Decisions |
| --- | --- | --- |
| `balance` | General traffic across quality, latency, and workload complexity | `simple`, `medium`, `complex`, `agentic`, `omni` |
| `speed` | Interactive applications, tools, and visual requests | `instant`, `heavy`, `omni`, `tooling`, `extended` |
| `cost` | High-volume traffic with bounded escalation | `economy`, `reasoning`, `omni`, `extended` |
| `accuracy` | Verification, expert synthesis, and bounded orchestration | `direct`, `verify`, `experts`, `orchestrate`, `extended`, `resume`, `omni` |
| `vault` | Sensitive workloads with local and tool-isolation policy | `private`, `restricted_tools`, `containment`, `sensitive`, `omni` |

These decision names form each Recipe's assignment contract. A published
Entrypoint must bind every reachable decision to one or more configured Models.

## Intended use

MoM V1 is designed for applications that serve mixed interactive, reasoning,
coding, multimodal, private, and long-context traffic through one API. It is
most useful when operators want routing policy to evolve independently from
provider connections and credentials.

It does not provision Models, choose a provider, or publish a callable model
name on its own.

## Routing behavior

Balance separates simple, medium, complex, agentic, and image-bearing work.
Speed favors instant responses while reserving explicit lanes for tools,
images, heavy work, and extended context. Cost uses an economy lane by default
and escalates only when request evidence calls for more capability.

Accuracy activates bounded verification, expert fusion, or workflow
orchestration only when the matching evidence is present. A completed tool turn
uses `resume` so the Router does not start another loop. Vault applies local,
tool-history, and replay constraints for sensitive traffic.

Every Recipe includes an `omni` decision for image-bearing requests. No
built-in decision injects a system prompt.

## Requirements

Assign Models whose cards satisfy the capabilities implied by each decision.
Image lanes require vision input; tool and orchestration lanes require tool-call
support; confidence-based algorithms require token log probabilities from the
assigned Models. Entrypoint validation rejects missing decisions and invalid
fallback tiers before publication.

The Recipes use semantic embedding, PII, jailbreak, fact-check, feedback,
language, conversation-shape, structure, and privacy knowledge-base signals.
Workflow and multi-model algorithms require their corresponding Router
integrations.

## Data handling and safety

Vault expresses the strictest built-in data boundary: its decisions disable
client tools, remove prior tool history, and disable replay capture where the
policy requires it. The control plane must assign Models and connections whose
placement and retention properties satisfy that boundary.

Other Recipes do not imply a placement boundary. Operators remain responsible
for provider credentials, network isolation, logs, stores, and retention.

## Quick start

```bash
vllm-sr serve
```

Connect Models in the Dashboard, open **Recipes**, choose a profile, and create
a **Mixture of Models**. Assign configured Models to every decision, configure
fallback only where intended, then publish the Entrypoint.

An independent control plane can perform the same lifecycle through the Router
Management API.

## Evaluation

[`probes.yaml`](probes.yaml) covers every decision across multilingual
paraphrases, benign negatives, priority collisions, tool and image shapes,
multi-turn histories, privacy signals, and context boundaries. Probes validate
routing policy independently from any physical Model assignment.

See the [conformance guide](../../../CONFORMANCE.md) for the validation contract.

## Limitations

- A Recipe cannot prove that an assigned Model is reachable or capable.
- Multi-model algorithms can add latency and compute cost.
- Classifier and knowledge-base errors can affect selection.
- Tool execution remains the client's responsibility.
- Deployment-specific quality, cost, context, and privacy claims require
  end-to-end evaluation after Models are assigned.

## References

- [Recipe metadata](metadata.yaml)
- [Recipe configuration](config.yaml)
- [DSL projection](recipe.dsl)
- [Evaluation probes](probes.yaml)
