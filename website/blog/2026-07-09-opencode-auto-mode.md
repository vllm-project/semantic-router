---
slug: opencode-auto-mode
title: "Adding Cursor-Style Auto Model Selection to OpenCode with vLLM Semantic Router"
description: A practical guide to adding intelligent auto model selection to OpenCode or any OpenAI-compatible agent using vLLM Semantic Router and AgentGateway — one endpoint, one virtual model, and semantic routing across a mixed local-and-cloud fleet.
authors: [anupsharma,aayushsaini101,shivjikumarjha]
tags: [opencode, routing, agentgateway, mixture-of-models, tutorial, community, vllm, semantic-router]
image: /img/blog/opencode-auto-mode-hero.png
---

<div align="center">

![OpenCode with vLLM Semantic Router: open provider interface, AgentGateway integration layer, and semantic routing hub](/img/blog/opencode-auto-mode-hero.png)

</div>

## The Feature Everyone Wants and Almost Nobody Has

Cursor's **Auto** mode is deceptively simple: the developer types, and the IDE chooses whether a prompt deserves a frontier model or something faster and cheaper. It is easy to stop noticing — until moving to an open tool where every request starts with a model dropdown.

<!-- truncate -->

That gap matters more than it sounds. Teams everywhere are standing up **local and internal model serving** — a fine-tuned coder on owned GPUs, a frontier API for hard problems, a fast cheap model for everything else. The models exist. The serving works. What is often missing is the *decision layer*: something that reads each request and sends it to the right backend automatically.

[OpenCode](https://opencode.ai) ships without a built-in Auto mode, but it does expose an open provider interface — enough to bolt intelligent routing on behind a single OpenAI-compatible endpoint.

This guide walks through building Auto mode for OpenCode (or any OpenAI-compatible client) with [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) and [AgentGateway](https://github.com/agentgateway/agentgateway): one endpoint, one model name, and an ML router choosing among the fleet per request. The fragments below cover the routing path; a production deployment must also enforce the pre-routing header boundary described in the architecture section.

<iframe
  width="100%"
  height="400"
  src="https://www.youtube.com/embed/_BUGwgXTpag"
  title="Demo"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  allowfullscreen>
</iframe>
*[Demo video: three prompts from one OpenCode session, routed live to three different models]*

---

## First, Know Your Options: The Routing Algorithms in vLLM Semantic Router

"Auto mode" means different things to different teams. The current public decision surface has **10 single-model selection algorithms** and **5 Looper algorithms** for multi-model orchestration. The status below comes from the same catalog enforced by configuration validation.

### Single-model selection (10)

| Algorithm | Status | What it does | Reach for it when... |
|---|---|---|---|
| `static` | Supported | Always chooses the first configured candidate | Determinism and curated ordering matter more than dynamic ranking. |
| `router_dc` | Supported | Compares the prompt with model descriptions in a dual-contrastive embedding space | Models are **specialists** — a coder, a reasoner, a generalist — and prompt intent should decide. This is the closest analogue to Cursor's Auto. |
| `multi_factor` | Supported | Scores quality, latency, cost, and load, with optional SLO filters | Models are interchangeable deployments and operational trade-offs should decide. |
| `latency_aware` | Supported | Selects from observed TTFT/TPOT percentiles | A hard responsiveness SLO dominates the routing decision. |
| `hybrid` | Supported | Blends several request-time ranking signals | One signal is insufficient and the fleet needs an auditable composite score. |
| `automix` | Experimental | Uses a cost-quality cascade with self-verification | Spending more only after a cheaper attempt lacks confidence is acceptable. |
| `knn` | Experimental | Routes from nearby labeled examples | An interpretable offline-trained example policy is available. |
| `kmeans` | Experimental | Maps embedding-space clusters to models | Prompt traffic forms stable clusters with known model winners. |
| `svm` | Experimental | Applies an offline-trained decision boundary | A compact classifier can separate prompt-to-model assignments. |
| `mlp` | Experimental | Applies a non-linear neural selector | Offline data supports more complex decision boundaries and the extra runtime dependency is acceptable. |

### Looper orchestration (5)

| Algorithm | Status | What it does |
|---|---|---|
| `confidence` | Supported | Escalates from a smaller model when confidence is insufficient. |
| `ratings` | Supported | Runs a bounded candidate panel and aggregates ratings. |
| `remom` | Supported | Performs multi-round parallel reasoning followed by synthesis. |
| `fusion` | Experimental | Runs parallel deliberation with judge analysis and final synthesis. |
| `workflows` | Experimental | Executes Router Flow static role plans or dynamic planner-generated workflows. |

Legacy `elo` and `rl_driven` selectors are **not** public configuration options: current config loading rejects both algorithm types. Long-running feedback and adaptation belong in [Router Learning](/docs/tutorials/learning/adaptations), where the current `routing_sampling` adaptation strategy operates over candidates supplied by a supported base selection algorithm such as `router_dc` or `multi_factor`.

Decisions can also be gated by **signal rules** — classifiers for intent, PII, and jailbreak detection. "Anything containing PII stays on the on-prem model" is enforceable as routing policy, not just documentation. When the primary need is *governance* rather than *optimization*, signal rules are the right layer.

**Practical rule of thumb:** interchangeable replicas → `multi_factor` or `latency_aware`. Specialist pools (local coder + frontier API) → prompt-aware routing with `router_dc`. The rest of this guide uses `router_dc`.

---

## The Design: One Virtual Model Called MoM

The Semantic Router's core abstraction is a **decision**: a named bundle of candidate models, a selection algorithm, and a *virtual model name* the client calls. A typical Auto-mode setup exposes `MoM` — Mixture of Models:

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

With `router_dc`, routing rules about prompt keywords are unnecessary. Instead, configure **plain-English descriptions of what each model is good at**:

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

At request time, an embedded **mmBERT** model (CPU, ~130MB) embeds the incoming prompt and compares it to those descriptions by cosine similarity. Example routing logs:

```plaintext
[RouterDC]   qwen-coder:   similarity=0.9998   ← "Write me a Python function..."
[RouterDC]   gpt-4o:       similarity=1.0000   ← "Compare utilitarian and deontological ethics..."
[RouterDC]   gemini-flash: similarity=0.9939   ← "What is the capital of Japan?"
```

Routing decision cost is typically **1–18ms on CPU**. The descriptions *are* the routing policy — adding a fourth model (a SQL specialist, a legal model) is a new model card, not new application code.

Why this pattern qualifies as true Auto mode:

1. **The client stays simple.** OpenCode holds no routing logic and no upstream API keys. Swap models, retune descriptions, add candidates — the client config stays fixed.
2. **Cost control is structural.** In a coding agent, code-shaped traffic lands on the $0 local model; only prompts that genuinely need frontier reasoning incur frontier pricing.
3. **It fails closed by default.** If the router is unavailable, the gateway rejects the request instead of accepting an untrusted model-selection header. This protects the routing and credential boundary even though it trades some availability for integrity.

---

## The Architecture

<div align="center">

![OpenCode Auto mode architecture: OpenCode TUI sends requests to AgentGateway, which pauses traffic for ExtProc semantic routing and forwards to Local Ollama, OpenAI Cloud, or Gemini Cloud](/img/blog/opencode-auto-mode-architecture.png)

</div>

The Semantic Router runs as an **Envoy ExtProc sidecar** to AgentGateway — no extra proxy hop. The gateway pauses each request, streams the body to the router over gRPC, receives a header mutation, and resumes. Routes match on that header:

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
    #     gemini-flash → Gemini (with backendAuth)
```

Treat `x-selected-model` as an internal control-plane header, never as client input. The secure default is fail-closed ExtProc handling. At the listener or another true **pre-routing** stage, unconditionally remove every client-supplied occurrence of `x-selected-model` before route matching; only a successful ExtProc call may reconstruct it. A normal Gateway API `HTTPRoute` `RequestHeaderModifier` runs after route matching and is therefore too late to sanitize a header that participates in selecting the route.

An operator may explicitly choose fail-open only after that pre-routing strip-and-rebuild invariant is enforced and tested with spoofed, duplicated, and differently cased headers as well as an ExtProc outage. Without that invariant, fail-open lets an external caller choose an upstream and potentially cross the gateway's credential boundary.

A critical security split: **the router never holds API keys.** It classifies and sets a header; AgentGateway owns `backendAuth` and injects credentials per upstream. The component making ML decisions on untrusted input should hold zero secrets — especially when the endpoint serves a team rather than a single laptop.

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

The Semantic Router intercepts `/v1/models` and **advertises the virtual model**, so OpenCode's discovery finds `MoM` as if it were a real backend. Select the provider, pick `MoM`, and every prompt is classified server-side before it reaches an upstream LLM.

<!-- Add screenshot at website/static/img/blog/opencode_gateway.png before publish:
![AgentGateway UI — one OpenCode session fanning out across qwen2.5-coder, gpt-4o, and gemini-2.5-flash, with per-request tokens and cost](/img/blog/opencode_gateway.png)
-->

For observability, AgentGateway ships a built-in UI (build with `--features ui`, served at `:15000/ui`). Keep routine diagnostics **content-free**: an opaque request ID and timestamp, selected decision and model, status, latency, token counts, and derived cost are normally enough to explain routing and spend. Do not include request bodies, prompts, completions, authorization headers, or tool arguments in the default log path.

If SQLite backs the UI, treat it as a **bounded metadata trail**, not a payload archive. Apply least-privilege file access, an approved encrypted volume or database layer, a short retention window, size or row limits, and automatic pruning. A database URL alone is not a retention or data-classification policy; define and test those controls before enabling persistence in production.

---

## Common Pitfalls

Four issues show up repeatedly when wiring this stack. Each applies broadly beyond OpenCode.

### Pitfall 1: The catch-all rule that matches nothing

Older examples use `rules: {}` as "match everything." On current `main`, an empty rules block evaluates as an OR over zero conditions — which matches **nothing**. Requests silently fall back to the default model: no errors, plausible answers, zero actual routing. The tell in logs is `"decision":""`.

The documented catch-all is an empty **AND**:

```yaml
rules:
  operator: AND
```

If the router "works" but never routes, verify the decision matches at all before tuning the algorithm.

### Pitfall 2: Choosing an algorithm that ignores the prompt

`multi_factor` scores configured quality, pricing, and live latency signals. It is a fleet-balancing algorithm; **the prompt never enters the equation.** Without quality or pricing data configured, every factor is neutral and traffic sticks on the first candidate indefinitely.

For specialist pools (coder + reasoner + generalist), use `router_dc`. Reserve `multi_factor` for interchangeable replicas where cost, latency, and load are the primary tradeoffs.

### Pitfall 3: A virtual model inherits its weakest backend's limits

A typical first failure after routing to a cloud model:

```plaintext
max_tokens is too large: 32000. This model supports at most 16384
completion tokens, whereas you provided 32000.
```

OpenCode cannot see individual backends behind `MoM`, so it may request a generous token budget. The virtual model's advertised limits must be the **intersection** across all candidates — smallest context window, safest output ceiling:

```jsonc
"limit": { "context": 32768, "output": 8192 }
```

The same intersection rule applies to tool calling, vision, and every other capability flag. Configure for the least capable backend, not the average one.

### Pitfall 4: Agentic clients + small local models = malformed tool JSON

OpenCode is an *agent*: every request carries a large system prompt and a tool catalog (`write`, `edit`, `webfetch`, ...). Frontier models return structured tool calls. Smaller local models often emit a broken *imitation* as plain text:

```plaintext
> what is python language

{"name":"write","arguments":{"content":"Python is a high-level,
interpreted programming language...","filePath":"python_language.md"}}
```

OpenCode replays history on every request, so malformed JSON from earlier turns becomes part of the context and the session self-reinforces the wrong format. Teams running compact local models alongside agentic clients should expect some variant of this behavior.

Start diagnosis with content-free metadata: correlate one request ID across OpenCode, the gateway, and the router; verify the selected decision and model, advertised tool capability, response status, token counts, and any structured-output parse error code. Reproduce in a fresh session before inspecting content, because poisoned history often disappears there.

Full payload capture is an exceptional incident-response tool, not a normal observability setting. If metadata cannot isolate the fault, use a separately reviewed, time-bounded capture that:

- is explicitly opted in and allowlisted to the minimum set of request IDs;
- is accessible only to the incident responders who need it and is encrypted in transit and at rest;
- redacts PII, credentials, authorization data, secrets, and sensitive tool arguments **before** persistence;
- expires through a short TTL with automatic deletion;
- audits capture activation, reads, exports, and deletion; and
- is disabled as soon as the incident is resolved.

Do not publish a reusable production configuration that stores complete prompts and completions by default.

**Mitigation:** define a custom OpenCode agent whose prompt **replaces** the built-in agentic system prompt:

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

Also set `"tool_call": false` on the model entry. Switch to the `chat` agent in the OpenCode TUI (**Tab**), start a fresh session (poisoned history does not self-heal), and models in the pool respond in clean markdown. Keep the default `build` agent for full agentic work with backends that handle tool calls reliably.

---

## Expected Results

Three prompts, one OpenCode session, one provider, one model name:

| Prompt | Auto-routed to | Where it ran | Cost |
|---|---|---|---|
| "Write me a Python function to compute fibonacci numbers using memoization" | `qwen2.5-coder` | Local Ollama | $0 |
| "Compare utilitarian and deontological ethics for AI decision making..." | `gpt-4o` | OpenAI | ~$0.03 |
| "What is the capital of Japan?" | `gemini-2.5-flash` | Google | ~$0.001 |

Routing overhead stays in the **1–18ms** range on CPU. The gateway UI surfaces per-hop tokens and cost; a bounded SQLite metadata trail supports post-incident correlation without turning normal diagnostics into a content store.

The outcome that matters: no model dropdown, no accidental frontier spend on trivial prompts, and no routing logic in the client — only English model descriptions and an embedding model that routes on semantic fit.

---

## Takeaways

- **Auto mode is a gateway feature, not a client feature.** Build it once behind an OpenAI-compatible endpoint and every compatible tool inherits it.
- **Match the algorithm to the business need.** Specialist models → `router_dc`. Interchangeable replicas → `multi_factor` / `latency_aware`. Maximum savings with escalation → `automix`. Governance → signal rules (PII, jailbreak).
- **A virtual model inherits the weakest backend's limits.** Context, output, tool calling — intersection, not average.
- **Content-free telemetry is non-negotiable; full payload capture is exceptional.** Correlate decisions, models, latency, tokens, cost, and errors by request ID. Capture content only under the narrow, short-lived incident controls described above.
- **Fail closed at the model-selection boundary.** Strip client-supplied `x-selected-model` before route matching and let only successful ExtProc processing rebuild it.
- **Keep keys out of the router.** The component reading untrusted prompts should hold zero secrets; let the gateway own auth.

Teams standing up internal model serving and wanting Cursor-style Auto mode can get there with two YAML files and a JSON block, using [AgentGateway](https://github.com/agentgateway/agentgateway) and [vLLM Semantic Router](https://github.com/vllm-project/semantic-router).

## Get Started

- [Install with agentgateway](/docs/installation/k8s/agentgateway) — Kubernetes integration guide
- [Router DC selection algorithm](/docs/tutorials/algorithm/selection/router-dc) — prompt-to-model-description routing
- [Giving AgentGateway a Semantic Brain](/blog/agentgateway-semantic-brain-homelab) — homelab walkthrough with ExtProc routing
- [Ollama local setup](/docs/installation/ollama) — run local models behind the gateway

---

*Published for [issue #2424](https://github.com/vllm-project/semantic-router/issues/2424). Have a community story about Semantic Router? Open a blog issue and the team will help get it published.*
