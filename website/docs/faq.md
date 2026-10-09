---
sidebar_position: 999
title: FAQ
description: What vLLM Semantic Router is, how it differs from llm-d and AI gateways, how to measure routing value, and how to debug a bad route.
---

# FAQ

## Is it an AI gateway, or something else?

It combines a frontend, an optional Decision Engine, and a model runtime. The
frontend accepts requests; the Decision Engine selects models or bounded
multi-model workflows from signals and policy. The default standalone gateway
forwards requests itself. Use `--gateway extproc` to integrate with Envoy.

The model runtime serves the Router's own judgments, embeddings, and other model
tasks. Chat backends run independently. TLS termination and cluster-wide Pod
scheduling remain deployment responsibilities.

See [Component Architecture](overview/component-architecture) for the architecture and
[Mixture of Models](overview/mom-model-family) for multi-model execution.

## When should I use Router or Engine mode?

Use Router when you want recipes to select or combine Chat backends. Use Engine
when your application only needs native judgments and you do not need Chat
backends or a routing recipe:

```bash
# Router: the positional model supplies routing judgments
vllm-sr serve vllm-sr/Vela-2.0-0.3B --config config.yaml

# Engine: the same frontend and runtime, without Chat routing
vllm-sr serve vllm-sr/Vela-2.0-0.3B --engine
```

`serve` defaults to Router; a positional model does not change the mode.
`--engine` also has the short form `-e`. Mode is chosen at startup. The Dashboard
manages model deployments and tasks in either mode.

## Can Router also expose System One?

Yes. Publish the native model through the listener's `systemone.models`
grant and call `/v1/systemone`. `/v1/decisions` is an alias. Native model discovery
uses `/v1/systemone/models`; Chat model discovery uses `/v1/models`.

Current System One requests name a concrete served model. Automatic native
routing through `vllm-sr/auto` is a roadmap item; the default Chat entrypoint
already uses that name. See the [System One guide](model-runtime/guides/decisions)
and [Quickstart](model-runtime/quickstart) for complete requests and access rules.

## How does it work with an agent harness?

The harness owns the agent loop, tools, and task state. It calls a stable model
entrypoint; the Router applies its recipe, and inference backends execute the
selected model path.

[Connect a harness](installation/agent-harness), or
[use an agent to install the Router](installation/agent).

## How do we measure routing accuracy and business value?

Classifier accuracy is not the number that decides this. The question is whether a routing recipe **keeps answer quality while reducing cost and latency** against a strong single-model baseline — and that claim needs paired evidence, not a before-and-after anecdote.

The repository's answer is **sr-bench**: freeze the cases, grader versions, request parameters and prices; run the strongest single model and the current mixture on the same aggregate; then read quality together with `100 × (1 − candidate cost / baseline cost)`. Two disciplines keep the number honest:

- the baseline is the best observed single **on the same cases**, not a per-question oracle, and quality ties resolve to the lower total cost;
- coverage is reported with the score — missing or ungraded results are not zeros, and a small dev win or a zero observed difference does not establish equivalence.

[sr-bench 1.0](benchmarking/sr-bench) defines the full measurement protocol, from freezing cases to reading the outcome.

## Why Semantic Router and not just llm-d?

Because the two systems answer different questions.

| | Semantic Router | llm-d |
| --- | --- | --- |
| Decides | which logical model or **pool** — *what* | which healthy **replica** inside that pool — *which / how / where* |
| Reads | request content, policy, semantic evidence | load, prefix-cache locality, replica health |
| Layer | request-time model selection | scheduling and placement |

llm-d should not decide business policy, and Semantic Router is not meant to choose a Pod. The project states the rule directly: *"Do not configure both systems to make the same decision."*

The [llm-d integration guide](installation/k8s/llm-d) states this boundary in its deployment context.

The Router's own Serving Engine also dispatches requests among replicas of its
task models. Those runtime pools are separate from an external Chat backend's
llm-d scheduling.

## How do we avoid conflicts with llm-d?

They avoid them by **not overlapping** rather than by arbitration. Semantic Router resolves the pool first; llm-d then picks a replica inside it. Because the two decisions live at different layers, a disagreement between "the best model for this request" and "the best replica for cache locality" resolves itself — the first decision constrains the second.

Two operational rules follow from that shape:

- **Deploy and verify llm-d independently before adding Semantic Router**, and use one supported llm-d release rather than mixing copied manifests.
- **Make the final decision visible.** The Router writes `x-vsr-selected-model` as its receipt and `x-selected-model` as the value a gateway matches on; `vllm-sr route probe --expect-selected-model` asserts the receipt. Read the decision from those rather than inferring it after the fact.

See [Integrate with llm-d](installation/k8s/llm-d) for the shared deployment and decision rules.

## How do we avoid hurting multi-turn / agentic workloads?

Switching models mid-conversation has a real cost — prefix-cache loss, tool drift and broken continuity. Treat it as a deliberate trade rather than an automatic optimization:

- **Prefer stickiness where continuity matters.** Session-scoped learning protects an established choice by default (built-in initialization defaults to protection on and online adaptation off), but only when the Router can identify the conversation: the default `scope: conversation` requires **both** `x-session-id` and `x-conversation-id`. Sending either header alone leaves the request without a retained model. Session-only identification applies to an explicit `scope: session`.
- **Switch only on an explicit escalation policy.** "Cheaper" is not by itself a reason to move a conversation.
- **Verify continuity, not just delivery.** Check the model actually used on continuation, tool completion, correction, model failure and conversation reset — an observed recommendation is not an applied hold.

[Session identification](api/session-identification) defines the header contract, and [Recipes](tutorials/global/recipes) shows where protection and escalation are configured.

## How do operators debug a bad outcome?

Walk the decision chain and stop at the first stage that does not match expectations. Each stage has its own surface:

```
signal → policy / decision → algorithm / model → plugin → endpoint scheduling → fallback → final model
```

1. **Preview before you generate.** `vllm-sr route preview` evaluates signals, projections and the decision without calling a backend, and reports the trace evidence.
2. **Read the provenance.** Learning-enabled preview exposes `selection_provenance`, including the config and state identity and the sampling seed.
3. **Then probe for real.** `vllm-sr route probe` sends an actual request through the configured inference listener. Check `response.body.delivery` and the final assistant output as well as HTTP status: empty output, reasoning-only output and `finish_reason: length` all fail delivery.
4. **Check the UI path separately.** Exercise the same entrypoint in the Dashboard and report API-only coverage apart from UI coverage.

Delivery, route correctness and answer quality are three separate outcomes; qualify the ones you did not measure instead of inferring success.

The [CLI reference](api/cli) documents every flag shown here, [VSR headers](troubleshooting/vsr-headers) lists the receipt headers, and [API and Observability](tutorials/global/api-and-observability) covers the telemetry surfaces.
