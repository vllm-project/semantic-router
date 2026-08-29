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
Balance and Speed treat a declared tool schema as available capability, not as
proof that the current turn wants to execute a tool. Their tool lanes require
explicit execution intent, a protocol-level required or named tool choice, or
an active tool loop. An explicit `tool_choice: none` suppresses fresh textual
tool intent while an already active loop retains continuity. Speed optimizes
the tooling lane for first-token latency and the heavy lane for generation
latency. Cost uses an economy lane by default; context size contributes bounded
evidence but does not trigger reasoning escalation by itself.

Accuracy activates bounded verification, expert fusion, or workflow
orchestration only when explicit matching evidence is present. Long context,
quoted routing phrases, and semantic difficulty alone do not fan out a request.
An ordinary completed tool turn uses `resume`; a trailing Flow-owned tool result
returns to `orchestrate` so the managed workflow can continue.

Vault evaluates PII across the conversation, applies local containment, strips
tool history, disables client tool execution, memory, response cache, replay,
and learning adaptation for every decision, and emits a drop-retention policy.

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

Before model selection, the Router removes candidates whose known context
window cannot hold the request. Models without context metadata remain eligible
for compatibility. If every assigned candidate has a known insufficient
window, the request is rejected instead of being sent to a backend that cannot
serve it.

When Router learning is enabled, ordinary single-model Accuracy decisions keep
adaptation inside the matched decision. Multi-model decisions and Vault bypass
adaptation. `resume` bypasses model-choice adaptation while retaining session
stability protection, which avoids changing lanes in the middle of a client
tool loop without pinning unrelated future decisions.

## Data handling and safety

Vault expresses the strictest built-in data boundary: all of its decisions
disable client tools, remove prior tool history, disable memory, response cache,
replay capture, and learning adaptation, and request immediate retention drop.
The control plane must still assign Models and connections whose placement and
retention properties satisfy that boundary.

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
