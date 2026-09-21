---
sidebar_position: 6
title: FAQ
description: What vLLM Semantic Router is, how it differs from llm-d and AI gateways, how to measure routing value, and how to debug a bad route.
---

# FAQ

Short answers to the questions that come up most often when evaluating vLLM Semantic
Router. Each answer links to the page that carries the detail, so this page stays short
as those pages grow.

## What is vLLM Semantic Router? Is it an AI gateway, or something else?

It is a **content- and policy-aware control plane** for Mixture-of-Models serving. It
reads the request — signals, projections and declared evidence — decides *what* should
run, and applies the decision through Envoy as an ExtProc filter.

That makes it a decision layer above a gateway rather than a gateway itself: it does not
terminate TLS, does not load-balance providers, and does not schedule pods. An AI
gateway answers "how do I reach N backends"; Semantic Router answers "which model,
recipe and policy should serve this request" — and then runs through whatever gateway
or listener you already have.

→ [System Overview](overview/semantic-router-overview) · [Mixture of Models](overview/mom-model-family)

## How do we measure routing accuracy and business value?

Classifier accuracy is not the number that decides this. The question is whether a
routing recipe **keeps answer quality while reducing cost and latency** against a
strong single-model baseline — and that claim needs paired evidence, not a
before-and-after anecdote.

The repository's answer is **sr-bench**: freeze the cases, grader versions, request
parameters and prices; run the strongest single model and the current mixture on the
same aggregate; then read quality together with
`100 × (1 − candidate cost / baseline cost)`. Two disciplines keep the number honest:

- the baseline is the best observed single **on the same cases**, not a per-question
  oracle, and quality ties resolve to the lower total cost;
- coverage is reported with the score — missing or ungraded results are not zeros, and
  a small dev win or a zero observed difference does not establish equivalence.

→ [sr-bench 1.0](benchmarking/sr-bench)

## If llm-d already does multi-model routing, why VSR?

Because the two systems answer different questions.

| | Semantic Router | llm-d |
| --- | --- | --- |
| Decides | which logical model or **pool** — *what* | which healthy **replica** inside that pool — *which / how / where* |
| Reads | request content, policy, semantic evidence | load, prefix-cache locality, replica health |
| Layer | control-plane decision | scheduling and placement |

llm-d should not decide business policy, and Semantic Router is not meant to choose a Pod.
The project states the rule directly: *"Do not configure both systems to make the same
decision."*

→ [Integrate with llm-d](installation/k8s/llm-d)

## How do VSR and llm-d Endpoint Picker avoid conflicts?

They avoid them by **not overlapping** rather than by arbitration. Semantic Router
resolves the pool first; llm-d then picks a replica inside it. Because the two
decisions live at different layers, a disagreement between "the best model for this
request" and "the best replica for cache locality" resolves itself — the first decision
constrains the second.

Two operational rules follow from that shape:

- **Deploy and verify llm-d independently before adding Semantic Router**, and use one
  supported llm-d release rather than mixing copied manifests.
- **Make the final decision visible.** Record it through headers, traces and replay
  rather than inferring it after the fact.

→ [Integrate with llm-d](installation/k8s/llm-d)

## How do we avoid hurting multi-turn / agentic workloads?

Switching models mid-conversation has a real cost — prefix-cache loss, tool drift and
broken continuity. Treat it as a deliberate trade rather than an automatic
optimization:

- **Prefer stickiness where continuity matters.** Session-scoped learning protects an
  established choice by default (built-in initialization defaults to protection on and
  online adaptation off), so send a stable session or conversation identity for it to hold.
- **Switch only on an explicit escalation policy.** "Cheaper" is not by itself a reason
  to move a conversation.
- **Verify continuity, not just delivery.** Check the model actually used on
  continuation, tool completion, correction, model failure and conversation reset — an
  observed recommendation is not an applied hold.

→ [Recipes](tutorials/global/recipes)

## How do operators debug a bad outcome?

Walk the decision chain and stop at the first stage that does not match expectations.
Each stage has its own surface:

```
signal → policy / decision → algorithm / model → plugin → endpoint scheduling → fallback → final model
```

1. **Preview before you generate.** `route preview` evaluates signals, projections and
   the decision without calling a backend, and reports the trace evidence.
2. **Read the provenance.** Learning-enabled preview exposes `selection_provenance`,
   including the config and state identity and the sampling seed.
3. **Then probe for real.** `route probe` sends an actual request through Envoy — but a
   successful HTTP status is not sufficient: check `response.body.delivery` and the
   final assistant output, because empty output, reasoning-only output and
   `finish_reason: length` all fail delivery.
4. **Check the UI path separately.** Exercise the same entrypoint in the Dashboard and
   report API-only coverage apart from UI coverage.

Delivery, route correctness and answer quality are three separate outcomes; qualify the
ones you did not measure instead of inferring success.

→ [API and Observability](tutorials/global/api-and-observability)
