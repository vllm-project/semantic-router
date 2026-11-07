---
slug: aegra-multi-model-langgraph-agents
title: "Build Multi-Model LangGraph Agents with Aegra and vLLM Semantic Router"
description: Run self-hosted LangGraph agents on Aegra, then point every LLM call at vLLM Semantic Router so deep agents get the right model per turn—without rewriting the graph.
authors: [aayushsaini101]
tags: [aegra, langgraph, agents, deep-agents, routing, semantic-router, enterprise, community]
image: /img/blog/aegra-multi-model-langgraph-agents/hero.jpg
---

# Build Multi-Model LangGraph Agents with Aegra and vLLM Semantic Router

<div align="center">

<p className="aegra-sr-brand-row">
  <a href="https://www.aegra.dev/" className="aegra-sr-brand-link">
    <img
      className="aegra-sr-brand-logo aegra-sr-brand-logo--light-bg"
      src="/img/blog/aegra-multi-model-langgraph-agents/aegra-logo.svg"
      alt=""
      width={36}
      height={30}
    />
    <img
      className="aegra-sr-brand-logo aegra-sr-brand-logo--dark-bg"
      src="/img/blog/aegra-multi-model-langgraph-agents/aegra-logo-light.svg"
      alt=""
      width={36}
      height={30}
    />
    <span className="aegra-sr-brand-text">aegra</span>
  </a>
  <strong className="aegra-sr-brand-x">×</strong>
  <a href="https://vllm-sr.ai/" className="aegra-sr-brand-link">
    <img
      className="aegra-sr-brand-mark"
      src="/img/blog/aegra-multi-model-langgraph-agents/vllm-sr-mark.svg"
      alt=""
      width={36}
      height={36}
    />
    <span className="aegra-sr-brand-text">vLLM Semantic Router</span>
  </a>
</p>

<p><em>Aegra × vLLM Semantic Router — agent runtime meets multi-model routing</em></p>

</div>

{/* Brand logos must stay small — blog CSS forces width:100% on content images. */}
<style>{`
.aegra-sr-brand-row {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin: 0.5rem 0 0.75rem;
}
.aegra-sr-brand-row img.aegra-sr-brand-logo,
.aegra-sr-brand-row img.aegra-sr-brand-mark {
  width: auto !important;
  max-width: none !important;
  height: 28px !important;
  margin: 0 !important;
  display: inline-block !important;
  object-fit: contain;
  border-radius: 0 !important;
  background: transparent !important;
}
.aegra-sr-brand-row img.aegra-sr-brand-mark {
  height: 32px !important;
  width: 32px !important;
}
.aegra-sr-brand-row img.aegra-sr-brand-logo--dark-bg { display: none !important; }
html[data-theme='dark'] .aegra-sr-brand-row img.aegra-sr-brand-logo--light-bg { display: none !important; }
html[data-theme='dark'] .aegra-sr-brand-row img.aegra-sr-brand-logo--dark-bg { display: inline-block !important; }
.aegra-sr-brand-x { font-size: 1.15rem; line-height: 1; opacity: 0.7; }
.aegra-sr-brand-link {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  text-decoration: none;
  color: inherit;
}
.aegra-sr-brand-text {
  font-weight: 700;
  font-size: 1.15rem;
  letter-spacing: -0.02em;
  line-height: 1;
}
`}</style>

Most LangGraph agents hardcode a single LLM. That forces bad tradeoffs: easy turns burn frontier spend, hard turns fail on a small model, and changing the fleet means editing graph code. Enterprises need something different—an **agent runtime they own**, plus a **model decision layer** that can classify each call and send it to the right backend.

This post shows how those layers fit together: **[Aegra](https://www.aegra.dev/)** for self-hosted Agent Protocol / LangGraph serving, and **[vLLM Semantic Router](https://vllm-sr.ai/)** for programmable multi-model routing.

<!-- truncate -->

## The problem: agents and models are different layers

An enterprise agent has to do more than call `chat.completions` once:

- Keep a long conversation alive across tools and human approvals
- Survive process crashes and redeploys
- Scope state to a user or tenant
- Plan, delegate, and work over many steps

Separately, each LLM call inside that agent has its own economics:

- Greeting vs. debugging vs. legal analysis need different models
- Local GPUs, private vLLM, and cloud APIs have different cost and residency rules
- Quality, latency, and spend should be policy—not hardcoded in a node

Hardcoding one model into the graph conflates those problems. Aegra owns the **agent lifecycle**. vLLM Semantic Router owns **which model answers each request**.

<div align="center">

![Your app talks Agent Protocol to Aegra; Aegra checkpoints to Postgres and sends LLM calls to vLLM Semantic Router for Ollama, Gemini, or vLLM](/img/blog/aegra-multi-model-langgraph-agents/architecture.svg)

</div>

Aegra stays the agent server (threads, workers, checkpoints). The one extra hop is **vLLM Semantic Router**: every `vllm-sr/auto` call is classified and sent to the right backend.

## What is Aegra?

[Aegra](https://www.aegra.dev/) is an open-source **Agent Protocol server**—a self-hosted alternative to managed LangGraph / LangSmith Deployments. You keep the same LangGraph SDK; you change the URL to your own infrastructure.

Out of the box it gives you:

| Capability | Why it matters |
| --- | --- |
| **Your Postgres** | Threads, runs, and checkpoints live in a database you control—region, backups, SQL access |
| **Agent Protocol API** | `threads`, `runs/stream`, state, assistants, store, crons—the surface Agent Chat UI and LangGraph Studio already speak |
| **Workers that survive restarts** | Redis queue with leases and crash recovery; scale by adding instances |
| **Custom auth in Python** | JWT, OAuth, Firebase; threads scoped to owners |
| **Streaming** | SSE token / value / event streams, including Agent Protocol v2 thread-scoped streaming |
| **OTLP tracing** | Langfuse, Phoenix, or your collector |

Aegra does **not** decide whether turn N should use Gemini Flash or a local coder. That is the Router’s job.

## How Aegra helps companies (and who uses it)

Companies adopt Aegra when they want LangGraph-class agents **without** handing thread state, auth, and data residency to a vendor-only control plane.

Public production stories from the [Aegra homepage](https://www.aegra.dev/) include:

| Organization | How they use Aegra |
| --- | --- |
| **[Mercanis](https://www.aegra.dev/)** | Procurement / AI platform team; chose Aegra to escape managed-platform lock-in and contribute upstream |
| **[tmarket.io](https://www.aegra.dev/)** | Runs Aegra on Kubernetes for their AI stack; cites concurrent load and reliability for inventory mapping at scale |
| **Diag n'Grow** | Data-collection chatbot (replacing fixed questionnaires) and a software audit agent |
| **ROQ Markets / ROQ Capital** | Core component of agentic AI infrastructure for an AI-powered investment platform—flexibility and control while staying in the LangGraph ecosystem |
| **Ingenix.AI** and others | Production agents with strong developer experience on self-hosted stacks |

The pattern is the same: **agents in production**, on **infra they operate**, with **portable SDK clients**.

## What are Deep Agents?

A shallow agent is “LLM + tools in a loop.” It works for short tasks and fails when the work is long, branched, or context-heavy.

**[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)** (LangChain’s agent harness on LangGraph) add the pieces that make long-horizon work reliable:

- **Planning** — decompose goals and track progress  
- **Subagents** — delegate with isolated context windows  
- **Filesystem / skills / memory** — persist artifacts and recall across steps  
- **Context management** — summarize, offload large tool results  
- **Human-in-the-loop** — pause for approval on risky actions  

Deep Agents still need a **runtime** (threads, checkpoints, streaming) and still emit many LLM calls. Aegra is a natural host for that runtime. vLLM Semantic Router is a natural front door for those LLM calls: each planning step, tool-loop turn, or subagent prompt can hit a different physical model under one logical name like `vllm-sr/auto`.

## Why threads and checkpoints matter

In Aegra, a **thread** is the durable conversation unit: messages, graph state, interrupts, and ownership. Runs execute *inside* threads. See [Threads and state](https://docs.aegra.dev/guides/threads-and-state).

A **checkpoint** is a snapshot after graph steps. That is what makes enterprise agents operable:

| Need | What checkpoints / threads enable |
| --- | --- |
| Crash recovery | Resume from the last durable step instead of restarting the whole agent |
| Human-in-the-loop | Interrupt, wait for approval, continue the same thread |
| Time travel / branch | Replay or fork from an earlier `checkpoint_id` |
| Audit & compliance | Inspect history: who ran what, with which state |
| Multi-tenant isolation | Auth-scoped threads; users only see their own work |
| Cost control | TTL / prune policies so long agent histories do not grow forever |

Without threads and checkpoints, “enterprise agent” is a demo script. With them, you get **durable, inspectable, recoverable** work—the difference between a chat toy and a production system.

Durability modes (`sync`, `async`, `exit`) let you trade step-level recoverability for throughput. Deep Agents’ delta-channel state has special TTL considerations; Aegra documents those explicitly so operators do not silently drop history they still need.

## Why connect Aegra to vLLM Semantic Router?

<div align="center">

<p className="aegra-sr-brand-row">
  <a href="https://www.aegra.dev/" className="aegra-sr-brand-link">
    <img
      className="aegra-sr-brand-logo aegra-sr-brand-logo--light-bg"
      src="/img/blog/aegra-multi-model-langgraph-agents/aegra-logo.svg"
      alt=""
      width={36}
      height={30}
    />
    <img
      className="aegra-sr-brand-logo aegra-sr-brand-logo--dark-bg"
      src="/img/blog/aegra-multi-model-langgraph-agents/aegra-logo-light.svg"
      alt=""
      width={36}
      height={30}
    />
    <span className="aegra-sr-brand-text">aegra</span>
  </a>
  <strong className="aegra-sr-brand-x">→</strong>
  <a href="https://vllm-sr.ai/" className="aegra-sr-brand-link">
    <img
      className="aegra-sr-brand-mark"
      src="/img/blog/aegra-multi-model-langgraph-agents/vllm-sr-mark.svg"
      alt=""
      width={36}
      height={36}
    />
    <span className="aegra-sr-brand-text">vLLM Semantic Router</span>
  </a>
</p>

</div>

Aegra answers: *How do we run and persist the agent?*  
Semantic Router answers: *Which model should serve this call under our policy?*

Point the agent’s OpenAI-compatible client at the Router listener (for example `http://localhost:8899`) and use a logical model such as `vllm-sr/auto`. The graph stays simple; the fleet stays flexible.

### What the Router adds

1. **Signals** — domain, complexity, safety, PII, conversation features, and more  
2. **Decisions** — named policies with priority (for example “code → Gemini”, “chat → local”)  
3. **Algorithms** — static, cascade, hybrid, session-aware selection, and others  
4. **Backends** — Ollama, vLLM, Gemini, OpenAI, private endpoints—under one API  

A concrete local pattern (from the workflow we exercised while drafting this post):

| Domain signal | Decision | Backend |
| --- | --- | --- |
| `computer science` | `code-to-gemini` | Gemini Flash-Lite |
| `math` | `math-to-gemini` | Gemini Flash-Lite |
| fallback | `default-ollama` | Local Ollama |

Preview without generating an answer:

```bash
curl -s 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-sr/auto","text":"Help me debug this Python program."}' \
  | python3 -m json.tool
```

You should see something like `decision_name: code-to-gemini`, domain confidence near `1.0`, and `selected_model` pointing at your cloud coder—while a greeting falls through to the local model. Response headers on live chat (`x-vsr-selected-decision`, `x-vsr-selected-model`, `x-vsr-matched-domains`) make the same decision visible to Aegra-side traces and support tooling.

For long-running agents, pair this with [session-aware agentic routing](/blog/session-aware-agentic-routing): send a stable `x-session-id` (often the Aegra `thread_id`) so tool loops and provider state do not break when the policy would otherwise switch models mid-turn.

## End-to-end: the enterprise agent stack

Putting it together:

1. **Client** creates an Aegra thread (per user / ticket / deal).  
2. **Aegra** runs a LangGraph or Deep Agent: plan → tools → maybe HITL → continue.  
3. **Every LLM call** goes to Semantic Router as `vllm-sr/auto` (or a named virtual model).  
4. **Router** classifies the *nature* of that turn (domain, difficulty, safety), matches a **decision**, and selects a backend.  
5. **Checkpoints** persist agent state in Postgres; **Router headers / preview** persist *why* a model was chosen.  
6. **Operators** tune `config.yaml` (or the Dashboard) to change fleet policy without redeploying the graph.

That is an end-to-end enterprise agent: durable orchestration on infra you own, plus explainable multi-model inference under policy.

```python
# Aegra side: same LangGraph SDK, your URL
from langgraph_sdk import get_client

client = get_client(url="https://agents.yourcompany.com")
thread = await client.threads.create(metadata={"tenant": "acme", "case": "INC-2044"})

# Inside the graph's LLM client (conceptual):
#   base_url = "https://router.yourcompany.com"  # vLLM Semantic Router
#   model = "vllm-sr/auto"
#   extra_headers = {"x-session-id": thread["thread_id"]}
```

## Tradeoffs and caveats

- **Two control planes** — Aegra for agent ops; Router for model ops. Own both configs deliberately.  
- **Protocol quirks** — some providers’ OpenAI-compat responses need models/Router versions that tolerate vendor usage fields (we hit this with full Gemini Flash on older Router releases; Flash-Lite was a practical workaround).  
- **Reachability** — from Docker, use `host.docker.internal` (or service DNS) for Ollama/vLLM; cloud keys via `api_key_env`, never committed YAML.  
- **Deep Agents + TTL** — DeltaChannel state may skip certain prune strategies; read Aegra’s checkpoint TTL notes before aggressive compaction.  
- **Session continuity** — prefer Router session awareness when tool loops span multiple LLM calls.

## What to build next

1. Run Aegra against your Postgres ([aegra.dev](https://www.aegra.dev/)).  
2. Run `vllm-sr serve` with at least two backends and domain (or decision) signals.  
3. Point the agent LLM client at the Router; use `vllm-sr/auto`.  
4. Pass `thread_id` as `x-session-id` for multi-turn agents.  
5. Use Router preview + `x-vsr-*` headers in staging until the decision policy matches product intent.

### Worked example: support agent on Aegra + Router

Here is a minimal local path that matches the checklist—Aegra for the thread, Semantic Router for every LLM call.

**1. Router config (two backends + domain decisions)**

Keep a small fleet: local Ollama for chat, Gemini for code/math. Sketch:

```yaml
providers:
  models:
    - name: llama3.2:3b
      backend_refs:
        - name: local-ollama
          endpoint: host.docker.internal:11434
          provider: ollama
    - name: gemini-2.5-flash-lite
      backend_refs:
        - name: google-gemini
          provider: gemini
          base_url: https://generativelanguage.googleapis.com/v1beta/openai
          api_key_env: GEMINI_API_KEY

routing:
  signals:
    domains:
      - name: computer science
        mmlu_categories: [computer science]
      - name: math
        mmlu_categories: [math]
  decisions:
    - name: code-to-gemini
      priority: 200
      rules:
        conditions: [{ type: domain, name: computer science }]
      modelRefs: [{ model: gemini-2.5-flash-lite }]
    - name: default-ollama
      priority: 100
      rules: { conditions: [] }
      modelRefs: [{ model: llama3.2:3b }]
```

Then start the stack (listener on `8899`, management often on `8080`):

```bash
export GEMINI_API_KEY=...   # never commit this
vllm-sr serve --config config.yaml
```

**2. Preview the policy before wiring the agent**

```bash
# Should match code-to-gemini → gemini-2.5-flash-lite
curl -s 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-sr/auto","text":"Help me debug this Python race condition."}' \
  | python3 -m json.tool

# Should fall through to default-ollama → llama3.2:3b
curl -s 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-sr/auto","text":"Thanks — talk later."}' \
  | python3 -m json.tool
```

**3. Aegra thread + LLM client pointed at the Router**

Create a durable thread in Aegra, then make every graph LLM call use `vllm-sr/auto` with that `thread_id` as `x-session-id`:

```python
import os
from langgraph_sdk import get_client
from langchain_openai import ChatOpenAI

# Agent runtime you own
aegra = get_client(url=os.environ["AEGRA_URL"])  # e.g. http://localhost:8000
thread = await aegra.threads.create(
    metadata={"tenant": "acme", "case": "INC-2044"},
)
session_id = thread["thread_id"]

# Model brain: one logical name, many physical backends
llm = ChatOpenAI(
    base_url="http://localhost:8899/v1",
    api_key="not-needed",  # backends use their own keys via api_key_env
    model="vllm-sr/auto",
    default_headers={"x-session-id": session_id},
)

# Use `llm` inside your LangGraph / Deep Agent nodes.
# Aegra still owns checkpoints; Router owns which model answers each turn.
```

**4. Confirm live routing with response headers**

On a real chat completion through the Router, inspect:

| Header | What it tells you |
| --- | --- |
| `x-vsr-selected-decision` | Which decision fired (`code-to-gemini`, `default-ollama`, …) |
| `x-vsr-selected-model` | Physical model that answered |
| `x-vsr-matched-domains` | Domains the classifier saw |

In staging, iterate: change prompts → `preview` → adjust decisions → re-check headers until product intent matches. Only then point production Aegra graphs at the same Router URL.

**5. Smoke-test the loop end-to-end**

```text
Client → Aegra (thread INC-2044)
       → graph node calls ChatOpenAI(model=vllm-sr/auto, x-session-id=thread_id)
       → Semantic Router classifies + selects backend
       → Ollama or Gemini answers
       → Aegra checkpoints state in Postgres
```

If a tool loop spans many LLM calls, keep the same `x-session-id` for the life of the thread so session-aware routing does not thrash models mid-case.

## Closing

Aegra gives companies a **self-hosted LangGraph / Deep Agent runtime**—threads, checkpoints, auth, workers—used in production by teams like Mercanis, tmarket, Diag n'Grow, and ROQ Markets. vLLM Semantic Router gives those agents a **programmable model brain**: classify the nature of each request and route it to the right compute.

Connect them, and you stop choosing between “one expensive model everywhere” and “fragile if/else in the graph.” You get durable enterprise agents—and decisions you can explain.
