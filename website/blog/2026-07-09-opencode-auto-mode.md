---
slug: opencode-auto-mode
title: "Adding Cursor-Style Auto Model Selection to OpenCode with vLLM Semantic Router"
description: Cursor picks the right model for you automatically. OpenCode doesn't - yet. A complete guide to adding intelligent auto model selection to OpenCode (or any OpenAI-compatible agent) using vLLM Semantic Router and AgentGateway.
authors: [anupsharma,aayushsaini101,shivjikumarjha]
tags: [opencode, routing, agentgateway, mixture-of-models, tutorial, community, vllm, semantic-router]
image: /img/blog/opencode-auto-mode-architecture.png
---

## The Feature Everyone Wants and Almost Nobody Has

If you've used Cursor, you know its **Auto** mode: you don't pick a model, you just type, and the IDE quietly decides whether your request deserves a frontier model or something faster and cheaper. It's one of those features you stop noticing until it's gone — and the moment you move to an open tool, it's gone.

<!-- truncate -->

This matters more than it sounds. Companies everywhere are standing up **local and internal model serving** — a fine-tuned coder on their own GPUs, a frontier API for the hard stuff, a fast cheap model for everything else. The models exist. The serving works. What's missing is the *decision layer*: something that reads each request and sends it to the right model, automatically, without every developer hand-picking from a dropdown all day.

[OpenCode](https://opencode.ai) — the open-source AI coding agent — has no Auto mode. But it has something better: an open provider interface. And that's all we need.

This post is a **one-stop guide** to building Auto mode for OpenCode — or honestly, for *any* OpenAI-compatible client — using [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) and [AgentGateway](https://github.com/agentgateway/agentgateway). One endpoint, one model name, and an ML router behind it choosing among your model fleet per request. Everything below runs today; every config is complete; every failure mode I hit is documented so you don't have to hit it.

The spark for this came from a presentation by [Johnu George](https://www.linkedin.com/in/johnu-george/) recently on using OpenCode with custom AI providers. I've been working in the vLLM Semantic Router codebase for a while — contributing a CLI shell completion command ([#2232](https://github.com/vllm-project/semantic-router/pull/2232)), performance benchmark and Hugging Face CLI fixes ([#2300](https://github.com/vllm-project/semantic-router/pull/2300)), auto-discovery for mmbert32k merged models ([#2302](https://github.com/vllm-project/semantic-router/pull/2302)), and SSE streaming for modality responses ([#2381](https://github.com/vllm-project/semantic-router/pull/2381)) — so connecting the two felt inevitable.

<iframe
  width="100%"
  height="400"
  src="https://www.youtube.com/embed/_BUGwgXTpag"
  title="Demo"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  allowfullscreen>
</iframe>
*[📹 Demo video: three prompts from one OpenCode session, routed live to three different models]*

---

## First, Know Your Options: The Routing Algorithms in vLLM Semantic Router

Before building anything, it's worth understanding what the Semantic Router actually offers — because "auto mode" means different things to different teams, and the project ships **more than a dozen selection algorithms**, each answering a different business question. This is the map I wish I'd had on day one:

| Algorithm | What it optimizes | Reach for it when... |
|---|---|---|
| `router_dc` | Prompt ↔ model-description similarity (dual-contrastive embeddings) | Your models are **specialists** — a coder, a reasoner, a generalist — and the prompt's *intent* should decide. The closest thing to Cursor's Auto. |
| `multi_factor` | Weighted quality / latency / cost / load score with SLO filters | Your models are **interchangeable** (same capability, different deployments) and you're balancing budget and latency guardrails across a fleet. |
| `latency_aware` | Live TTFT/TPOT percentiles | You have hard latency SLAs — user-facing chat where p95 time-to-first-token is the metric that gets you paged. |
| `automix` | Cost, via cascade + self-verification (POMDP, from the AutoMix paper) | You want maximum savings and can tolerate escalation: try the cheap model, verify its own answer, escalate only on low confidence. Great for batch/offline work. |
| `elo` | Feedback-driven ranking | You collect user feedback (thumbs, regenerations) and want rankings that improve continuously in production. |
| `knn` / `svm` / `mlp` / `kmeans` | Learned routing from labeled examples | You have historical data of "this prompt type → this model worked" and want a trained policy instead of a hand-written one. |
| `rl_driven` | Long-run reward | You can define a reward signal and want the router to optimize it over time. |
| `hybrid` | Intent + operational signals combined | Big fleets: specialists *and* replicas, where both "what is this prompt" and "which deployment is healthy" matter. |
| `static` | Determinism | Compliance and predictability: category X always goes to model Y, auditable, no surprises. |

And one thing that isn't an algorithm but matters to enterprises: decisions can also be gated by **signal rules** — the router ships classifier models for intent, PII, and jailbreak detection. "Anything containing PII stays on the on-prem model, no exceptions" is a routing rule here, not a policy document. If your business need is *governance* rather than *optimization*, that's the feature to explore.

**The honest takeaway:** if your models are clones of each other, you want the signal-based family (`multi_factor`, `latency_aware`). If your models have *different jobs* — which is exactly the local-coder-plus-frontier-API setup most teams are building — you want prompt-aware routing. That's `router_dc`, and that's what we'll use.

---

## The Design: One Virtual Model Called MoM

The Semantic Router's core abstraction is a **decision**: a named bundle of candidate models, a selection algorithm, and a *virtual model name* clients call. Mine is `MoM` — Mixture of Models:

```yaml
decisions:
  - name: MoM
    description: "Mixture of Models router"
    priority: 100
    rules:
      operator: AND        # empty AND = catch-all (see gotcha #1 below!)
    modelRefs:
      - model: qwen-coder
      - model: gpt-4o
      - model: gemini-flash
    algorithm:
      type: router_dc
```

With `router_dc`, you don't write rules about prompts at all. You write **descriptions of what each model is good at, in plain English**:

```yaml
modelCards:
  - name: qwen-coder
    description: >
      Specialized coding model optimized for programming tasks.
      Excellent at writing code, debugging, algorithms, ...

  - name: gpt-4o
    description: >
      Frontier reasoning model with exceptional analytical capability.
      Best for complex multi-step reasoning, strategic analysis, ...

  - name: gemini-flash
    description: >
      Fast general-purpose model. Ideal for simple factual questions,
      quick lookups, summarization, casual conversation, ...
```

At request time, an embedded **mmBERT** model (CPU, ~130MB) embeds the incoming prompt and compares it to those descriptions by cosine similarity. Real logs from my setup:

```plaintext
[RouterDC]   qwen-coder:   similarity=0.9998   ← "Write me a Python function..."
[RouterDC]   gpt-4o:       similarity=1.0000   ← "Compare utilitarian and deontological ethics..."
[RouterDC]   gemini-flash: similarity=0.9939   ← "What is the capital of Japan?"
```

Routing decision cost: **1–18ms, on CPU**. The descriptions *are* the routing policy — adding a fourth model (a SQL specialist, a legal model) is a new model card, not new code.

Why this design earns the name "auto mode" rather than "clever hack":

1. **The client stays dumb.** OpenCode holds no routing logic and no upstream API keys. Swap models, retune descriptions, add candidates — the client config never changes.
2. **Cost control is structural.** In a coding agent, the bulk of traffic is code-shaped and lands on your $0 local model. Only prompts that genuinely need frontier reasoning pay frontier prices.
3. **It fails soft.** The gateway policy is `failureMode: failOpen` — if the router process dies, traffic falls through to a default route. Users see answers, not incidents.

---

## The Architecture

<div align="center">

![OpenCode Auto mode architecture: OpenCode TUI sends requests to AgentGateway, which pauses traffic for ExtProc semantic routing and forwards to Local Ollama, OpenAI Cloud, or Gemini Cloud](/img/blog/opencode-auto-mode-architecture.png)

</div>

The Semantic Router runs as an **Envoy ExtProc sidecar** to AgentGateway — no extra proxy hop. The gateway pauses each request, streams the body to the router over gRPC, gets back a header mutation, and resumes. Routes then match on that header:

```yaml
binds:
- port: 3000
  listeners:
  - routes:
    - matches:
      - headers:
        - name: x-selected-model
          value:
            exact: qwen-coder
      backends:
      - ai:
          provider:
            openAI: {}
          name: ollama
          hostOverride: localhost:11434
    # ... gpt-4o → OpenAI (with backendAuth),
    #     gemini-flash → Gemini (with backendAuth),
    #     plus a failOpen fallback route
```

One design choice worth stealing even if you steal nothing else: **the router never touches API keys.** It classifies and sets a header; AgentGateway owns `backendAuth` and injects credentials per upstream. The component making ML decisions on untrusted input holds zero secrets. If this endpoint ever serves a whole team instead of one laptop, that separation is the difference between a routing bug and a credentials incident.

---

## Wiring OpenCode: The Whole Integration Is One Config Block

OpenCode accepts any OpenAI-compatible endpoint as a custom provider. In `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "provider": {
    "auto_sr": {
      "name": "Auto (Semantic Router)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:3000/v1"
      },
      "models": {
        "MoM": {
          "name": "MoM",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

A satisfying detail: the Semantic Router intercepts `/v1/models` and **advertises the virtual model**, so OpenCode's discovery finds `MoM` as if it were a real model. To the client, it is one.

Select the provider, pick `MoM`, and you have Auto mode: every prompt is classified server-side and lands on the best model in your pool.

<!-- Add screenshot at website/static/img/blog/opencode_gateway.png before publish:
![AgentGateway UI — one OpenCode session fanning out across qwen2.5-coder, gpt-4o, and gemini-2.5-flash, with per-request tokens and cost](/img/blog/opencode_gateway.png)
-->

For visibility, AgentGateway ships a built-in UI (build with `--features ui`, served at `:15000/ui`) and can persist every request to SQLite with cost attribution:

```yaml
config:
  modelCatalog:
  - file: base-costs.json
  database:
    url: sqlite://agentgateway.db
```

Keep that database enabled. It's about to earn its place.

---

## The Gotchas (So You Don't Spend My Week)

Four things broke on the way. Each one is general — if you build this, you *will* meet some of them.

### Gotcha 1: The catch-all rule that caught nothing

Older examples show `rules: {}` as "match everything." On current `main`, an empty rules block evaluates as an OR over zero conditions — which matches **nothing**. Every request silently fell back to the default model. No errors, plausible answers, zero actual routing. The only tell was `"decision":""` in the routing logs.

The documented catch-all is an empty **AND**:

```yaml
rules:
  operator: AND
```

*If your router "works" but never routes, check that your decision matches at all before blaming the algorithm.*

### Gotcha 2: Picking an algorithm that reads the prompt

My first attempt used `multi_factor` and everything went to one model — no matter how I rebalanced the weights. Reading the selector source explained it: `multi_factor` scores configured quality/pricing plus live latency signals. It's a fleet-balancing algorithm; **the prompt never enters the equation.** With no quality/pricing data configured, every factor is neutral and you get the first candidate forever.

The algorithm-to-business-need table earlier in this post exists because of this mistake. For specialist pools, use `router_dc`.

### Gotcha 3: A virtual model inherits its *weakest* backend's limits

First cloud-routed prompt from OpenCode:

```plaintext
max_tokens is too large: 32000. This model supports at most 16384
completion tokens, whereas you provided 32000.
```
r
OpenCode can't know what's behind `MoM`, so it asked for a generous budget; GPT-4o caps completions at 16,384. The virtual model's advertised limits must be the **intersection** across all candidates — smallest context window, safest output ceiling:

```jsonc
"limit": { "context": 32768, "output": 8192 }
```

The same intersection rule applies to every capability: context, output, tool calling, vision. Configure for your least capable model, not your average one.

### Gotcha 4: Agentic clients + small local models = JSON soup

OpenCode is an *agent*: every request carries a ~10KB system prompt ("use the tools available to you") and a catalog of tools (`write`, `edit`, `webfetch`...). Frontier models return structured tool calls. Small local models often return a broken *imitation* of a tool call, as plain text:

```plaintext
> what is python language

{"name":"write","arguments":{"content":"Python is a high-level,
interpreted programming language...","filePath":"python_language.md"}}
```

And it compounds: OpenCode replays history with every request, so the model's malformed JSON becomes part of its own context, and it concludes JSON is the house style. The session poisons itself. (Your local model size is your call — mine is small purely due to hardware constraints — but every team running compact local models next to an agentic client will eventually see some version of this.)

How I diagnosed it instead of guessing: **payload logging at the gateway**. AgentGateway can capture the full messages array:

```yaml
frontendPolicies:
  http:
    accessLog:
      database:
        add:
          gen_ai.prompt: llm.prompt
          gen_ai.completion: 'llm.completion.map(c, {"role":"assistant", "content": c})'
```

One SQL query showed me the exact prompt the model received — the giant agent preamble, and its own previous JSON replies staring back from the history. Debugging LLM pipelines without payload capture is astrology.

The fix that keeps Auto mode clean for *any* model pool — a custom OpenCode agent whose prompt **replaces** the built-in agentic one:

```jsonc
"agent": {
  "chat": {
    "description": "Plain chat through the semantic router",
    "mode": "primary",
    "model": "auto_sr/MoM",
    "prompt": "You are a helpful assistant. Answer directly and concisely in clean markdown. Use fenced code blocks for any code. Never output JSON tool calls.",
    "permission": { "edit": "deny", "bash": "deny", "webfetch": "deny" }
  }
}
```

Plus `"tool_call": false` on the model entry. Press **Tab** in the OpenCode TUI to switch to the `chat` agent, start a fresh session (poisoned history stays poisoned), and every model in the pool answers in clean markdown. Keep the default `build` agent for full agentic work with your strongest tool-calling models.

---

## The Payoff

Three prompts, one OpenCode session, one provider, one model name:

| Prompt | Auto-routed to | Where it ran | Cost |
|---|---|---|---|
| "Write me a Python function to compute fibonacci numbers using memoization" | `qwen2.5-coder` | Local Ollama | $0 |
| "Compare utilitarian and deontological ethics for AI decision making..." | `gpt-4o` | OpenAI | ~$0.03 |
| "What is the capital of Japan?" | `gemini-2.5-flash` | Google | ~$0.001 |

Routing overhead: 1–18ms per request, on CPU. The gateway UI shows every hop with tokens and cost; the SQLite log holds the payload trail for the day something looks off.

What I like most is what's *absent*. No model dropdown. No "oops, I burned frontier tokens on a trivial question." No routing code in any client. Just English descriptions of what each model is good at — and an embedding model that takes them seriously.

---

## Takeaways

- **Auto mode is a gateway feature, not a client feature.** Build it once behind an OpenAI-compatible endpoint and *every* tool you use inherits it — OpenCode today, whatever you adopt next year.
- **Match the algorithm to the business need.** Specialist models → `router_dc`. Interchangeable replicas → `multi_factor` / `latency_aware`. Maximum savings with escalation → `automix`. Governance → signal rules (PII, jailbreak). The table above is the decision I wish someone had written down for me.
- **A virtual model inherits the weakest backend's limits.** Context, output, tool calling — intersection, not average.
- **Payload logging is non-negotiable.** My two hardest bugs (a rule that never matched, a session that poisoned itself) were invisible until I could read the actual bytes on the wire.
- **Keep keys out of the router.** The component reading untrusted prompts should hold zero secrets; let the gateway own auth.

If your team is standing up internal model serving and wishing your tools had a Cursor-style Auto mode — this is that, in two YAML files and a JSON block, with [AgentGateway](https://github.com/agentgateway/agentgateway) and [vLLM Semantic Router](https://github.com/vllm-project/semantic-router).

And yes — mine runs in my living room. 🏠

## Get Started

- [Install with agentgateway](/docs/installation/k8s/agentgateway) — Kubernetes integration guide
- [Router DC selection algorithm](/docs/tutorials/algorithm/selection/router-dc) — prompt-to-model-description routing
- [Giving AgentGateway a Semantic Brain](/blog/agentgateway-semantic-brain-homelab) — homelab walkthrough with ExtProc routing
- [Ollama local setup](/docs/installation/ollama) — run local models behind the gateway

---

*Published for [issue #2424](https://github.com/vllm-project/semantic-router/issues/2424). Have a community story about Semantic Router? Open a blog issue and the team will help get it published.*
