---
slug: runkite-ai-sdk-semantic-router
title: "One Conversation, Multiple Models: Runkite and vLLM Semantic Router"
description: "Connect a native AI SDK agent to user-configured model backends with Runkite and vLLM Semantic Router. Preserve chat history and enforce tool permissions."
authors: [sharanharsoor]
tags: [runkite, ai-sdk, agents, routing, semantic-router, community]
image: /img/blog/runkite-ai-sdk-semantic-router/hero.jpg
---

import { ArticleFigure } from '@site/src/components/ArticleMedia';

**[Runkite website](https://getrunkite.com/) · [GitHub](https://github.com/getrunkite/runkite) · [Runnable example and recorded results](https://gist.github.com/sharanharsoor/84f4dbd4913106d06f8b8156fa8f47a9)**

An incident investigation can begin with a small request: read the incident and tell me which service is affected. The next question asks why it is failing. Eventually, someone asks the agent to restart the service.

The conversation has moved from reading to reasoning to acting. Those requests can use different model routes, but changing the model should not change the agent's permission to act. And if the process running the agent restarts along the way, the next question still belongs to the same conversation.

That is the path I wanted to test by connecting **[Runkite](https://getrunkite.com/)** with **vLLM Semantic Router**. Runkite's existing managed Vercel AI SDK runner would execute the agent. Semantic Router would select the model backend for its inference requests. The SDK would keep its native tool loop, while model selection would live in the router configuration.

I followed one conversation through three turns: read an incident, restart the runner and ask for a diagnosis, then request a service restart the agent was forbidden to perform. The incident and tool service are fictional. The model calls, streamed tool requests and durable results are real.

To understand the result, start with what happens between a user's question and an agent's answer.

<!-- truncate -->

<ArticleFigure
  src="/img/blog/runkite-ai-sdk-semantic-router/hero.jpg"
  width={1200} height={630}
  alt="Runkite and vLLM Semantic Router: one conversation follows configured model routes while incident reads are allowed and the service restart is denied."
/>

## Start with the loop

When a user asks about an incident, the model may need information it does not have. It requests a tool, the SDK executes that tool, and the result joins the conversation. The SDK then asks the model what to do next. A single user turn can therefore make several inference calls before producing its answer.

That gives us two decisions at different boundaries:

- Which model should serve this inference request?
- May this agent execute the requested tool?

vLLM Semantic Router makes the first decision from the configured routing policy. The SDK keeps running its loop. In this example, Runkite enforces the second decision when a tool crosses its connector gate.

There is also a connection between one turn and the next: conversation state. If the process running the SDK restarts, the next model call still needs the previous messages and completed tool results. Routing, tool authority and conversation continuity need to work together. Giving each responsibility a clear home made the integration easier to reason about.

## Give each layer a job

[Runkite](https://github.com/getrunkite/runkite) is an Apache-2.0 self-hosted agent control plane. It gives dispatch, conversation records, connector authorization and audit a home outside the agent's logic. It represents the conversation as a **Thread** and each execution as a **Run**.

**vLLM Semantic Router** provides the inference decision layer. The agent uses one logical model name at one inference endpoint; the router evaluates request signals and routing rules to select a backend from your configured model pool. This keeps model-selection policy outside the agent loop.

For this experiment, the application starts a run through Runkite's HTTP API. The plane admits and dispatches it to the TypeScript runner, which executes the native AI SDK `ToolLoopAgent` and reports events and results.

Once Runkite has dispatched the run, the SDK's model provider calls Semantic Router's OpenAI-compatible endpoint, where the router chooses a backend. A tool request takes a return path through Runkite's run-bound connector proxy, where the requested action is checked before it can reach the tool service.

<ArticleFigure
  src="/img/blog/runkite-ai-sdk-semantic-router/architecture.svg"
  width={960} height={1200}
  alt="Managed execution from application to Runkite, then the native AI SDK runner, vLLM Semantic Router and configured LLM backends. A separate tool-call branch returns through Runkite's connector gate."
  diagram
>
  Read top to bottom: the app starts a run in Runkite, Runkite dispatches the agent, and the agent calls Semantic Router to reach a model. Tool requests return through Runkite's connector gate. The Go plane owns admission and dispatch; inference HTTP originates in the runner. Open the image for full size.
</ArticleFigure>

This is **managed execution**: the agent loop runs in a Runkite runner, and the application creates Runkite runs. An existing Next or Nuxt app that keeps its SDK loop inside the app remains a different integration. Runkite's decision-only governance API can govern submitted operations from that app, but it does not turn those calls into managed Threads/Runs.

Behind the router, you configure the LLM backends: **vLLM**, **Ollama**, or another compatible model-serving endpoint. The serving engine and model family are deployment choices. This integration uses **Chat Completions with streaming tool calls**; configure backend endpoints, credentials, provider dialect and model options to match the capabilities supported by your Semantic Router version and SDK provider.

## Connect the existing interfaces

With those responsibilities in place, the first connection was the model endpoint. The managed AI SDK runner already loads a per-run factory. Its context includes the thread, tenant, run, generation and cancellation signal, plus the configuration needed by the connector helpers. I used that factory to construct the model provider and two tool executors.

The provider points at the local Compose stack's inference listener:

```typescript
import { createOpenAI } from '@ai-sdk/openai';

const provider = createOpenAI({
  baseURL: 'http://127.0.0.1:18899/v1',
  apiKey: 'local-semantic-router',
});

const model = provider.chat('vllm-sr/runkite');
```

The explicit `.chat(...)` matters. The provider's default model factory uses the Responses API; this minimal stack enables Chat Completions. The placeholder key belongs to this private local listener. Runkite's client and runner credentials are separate, and a deployed inference listener needs its own access controls.

The [full factory](https://gist.github.com/sharanharsoor/84f4dbd4913106d06f8b8156fa8f47a9) also supplies opaque tenant/thread correlation headers and captures selected routing headers and per-call token observations. These identifiers provide correlation, not authentication.

The next step was to give that one endpoint a choice of backends. The model name `vllm-sr/runkite` stays constant in the SDK, while the router configuration defines the `local-fast` and `local-specialist` model aliases and their backend bindings. Those names describe roles in the example configuration. In your deployment, bind routing roles to your own compatible model pool.

In the example's pinned Semantic Router 0.4.0 configuration, `global.router.auto_model_names` registers `vllm-sr/runkite` as an automatic-routing alias. Keep that entry when changing the backend bindings; without it, the router treats the request's model name as a concrete model and returns HTTP 400 if no such model is configured.

To make the example's route choices reproducible, I used an explicit keyword signal. Requests containing the words `diagnose`, `specialist`, `code` or `algorithm` select the specialist route; other requests use the default model. The signal uses case-insensitive regular expressions with word boundaries. The alias names are not measured claims about speed or answer quality.

The alias registration and routing decision look like this:

```yaml
global:
  router:
    auto_model_names: [vllm-sr/runkite]

routing:
  decisions:
    - name: specialist-route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: specialist-request
      modelRefs:
        - model: local-specialist
          use_reasoning: false
      algorithm:
        type: static
```

This is an excerpt. Use the full checked-in configuration for the signal, backend bindings and disabled optional model consumers. The fixture exercises routing and SDK compatibility without requiring classifier downloads or claiming an evaluation of learned semantic routing.

### Put the tool call through the gate

Once the model could request a tool through its native SDK loop, the other path needed an explicit permission check. The agent exposes `read_incident` and `restart_service`. Both executors use the same existing connector helper, so each request returns through Runkite before reaching the incident service. The important part of the read executor is:

```typescript
import { proxyConnectorMcp } from 'runkite-runner';

const response = await proxyConnectorMcp(
  context.runkiteConfig,
  'incidents',
  {
    jsonrpc: '2.0',
    id: crypto.randomUUID(),
    method: 'tools/call',
    params: {
      name: 'read_incident',
      arguments: { incident_id: 'INC-1042' },
    },
  },
  { signal: context.signal, timeoutMs: 10_000 },
);
```

The factory's `context` binds the call to its managed execution. The full executor checks JSON-RPC errors and MCP `isError` before returning a successful tool result.

The generated Runkite configuration grants `read_incident` and denies `restart_service` on the `incidents` connector. I deliberately left the restart tool visible to the model so the experiment would exercise enforcement when an actual request arrived. Its upstream implementation is harmless and increments a counter if reached.

This is why the tool's execution path matters as much as its definition. A native tool that bypasses this proxy does not receive this connector authorization. With the model route and tool gate connected, I could test how they behaved together in a conversation.

## Follow one conversation through three runs

The conversation begins with a catalog service that is degraded after a deployment. In the fixture, HTTP concurrency increased from 10 to 40, all 20 database pool connections are active, and requests are timing out while acquiring a connection. These are fixed fictional observations, so the example is safe to reproduce.

For the recorded test below, I used two **Qwen** models served by **Ollama**. The tested backend bindings were:

- `local-fast` mapped to `qwen3:0.6b`.
- `local-specialist` mapped to `qwen3:1.7b`.

The three turns follow the same Thread. Each turn creates a new managed Run, giving us a durable record to compare with the model stream and the tool service's counters.

<ArticleFigure
  src="/img/blog/runkite-ai-sdk-semantic-router/conversation.svg"
  width={760} height={846}
  alt="Recorded local test, using Qwen as example models: a status read on Qwen 0.6B, a runner restart between completed runs, a diagnostic follow-up on Qwen 1.7B, and a denied restart request on Qwen 0.6B."
  diagram
>
  Recorded local test with Qwen: one Thread, three managed Runs. The observed model change happens between user turns; this test does not establish model switching between tool steps inside one turn. Open the image for full size.
</ArticleFigure>

### First, read the incident

The first request explicitly asks the agent to call `read_incident` for `INC-1042` and report its status and service. The model requests the tool, Runkite allows the connector call, and the fixture returns the incident record. The SDK uses that result in its next inference call and completes the turn.

Both inference calls in this run reported `qwen3:0.6b`, matching the router's `local-fast` response header. The native stream completed, and `/wait` agreed with the durable run output.

### Then restart the runner and ask a follow-up

The status request established that the agent could retrieve an incident and answer through the default route. The follow-up would ask more of the system: preserve that conversation while taking a different model route.

After the first run completed, the verifier stopped and restarted the TypeScript runner process. It then submitted a diagnostic follow-up on the same thread, asking the agent to read the incident again before explaining a likely cause.

The word `diagnose` selected `specialist-route`. Both model responses reported `qwen3:1.7b`, matching `local-specialist`. The saved native conversation retained the first user turn and its completed tool result, and the second read reached the fixture.

This establishes conversation continuity across a runner restart between completed turns. It is a different test from crashing halfway through a side effect, and it does not rank the models' diagnostic quality.

### Finally, request the forbidden operation

The conversation had now exercised both model routes and survived a runner restart. Its next step crossed from investigation into action: the third request asks for `restart_service`.

The model actually requests that tool. Runkite denies it and records `policy_tool_denied`; the SDK receives a tool error and completes the conversational turn.

The run finishes successfully because the agent handled the denied tool request. That status is not evidence that the service restarted. The authorization audit and service counter answer the execution question:

```json
{
  "fixture_counts": {
    "read_incident": 2,
    "restart_service": 0
  },
  "governance": {
    "durable_denials": 1,
    "reason_codes": ["policy_tool_denied"],
    "upstream_restart_calls": 0
  }
}
```

These fields come from the [sanitized qualification report](https://gist.github.com/sharanharsoor/84f4dbd4913106d06f8b8156fa8f47a9). The model made the request; the service never received it. Changing the model route had not changed the connector's permission.

## A successful completion request was not enough

Getting to those three completed turns required checking more than endpoint connectivity. During the first live integration, a normal completion through Semantic Router 0.4.0 and Ollama 0.40.3 worked. The streamed tool loop failed.

Ollama added a top-level `timings` object to the final streaming usage chunk. The stable router's strict response decoder rejected it with `invalid_upstream_json`, after the stream had already emitted a tool-call chunk. Receiving HTTP 200 and seeing a tool name were insufficient evidence of a completed agent turn.

The working example pins **Ollama 0.34.3**, the version identified in Semantic Router's [recorded Ollama stream fixture](https://github.com/vllm-project/semantic-router/blob/6573123715e214a9eaa3ad23372ee5157411e519/src/semantic-router/pkg/extproc/processor_res_ollama_stream_test.go). Both backend bindings identify the `ollama` provider dialect so the router can also adapt output limits correctly.

That failure shaped the verifier: it checks complete native stream events and matching durable output. I also wanted to see what happened when a turn could not finish normally. An unreachable inference endpoint produces a durable error without tool execution. Cancellation of an active model call reaches the runner over gRPC and the native SDK signal and leaves the run interrupted. The endpoint test changes its own runner's target; it does not shut down the shared router or test failover.

Cancellation remains cooperative. It does not prove that backend compute stopped, undo a tool that already executed or guarantee exactly-once external effects. Routing headers and token observations also do not establish dollar savings: Runkite's existing AI SDK run usage is an aggregate tagged with its final observed model.

## Run the same experiment

The same investigation can run locally, including the model calls. The [public example](https://gist.github.com/sharanharsoor/84f4dbd4913106d06f8b8156fa8f47a9) contains the source patch, reproduction README, engineering note and sanitized results. It uses the existing managed AI SDK runner and is distributed as a patch against the tested Runkite revision. Starting from that revision keeps the setup reproducible as both projects evolve.

You need Docker Compose, Node.js 22.13 or later, Python 3.9 or later, and a Go toolchain satisfying Runkite's `go.mod`. Allow several GB for the images and model weights. No provider account or real incident system is required.

First, clone the tested source:

```bash
git clone https://github.com/getrunkite/runkite.git
cd runkite
git checkout edf36c133955aeea3d9c28b353e64b0a64a3a3b4
```

Download the recorded patch revision from the Gist, then apply it:

```bash
curl -fL 'https://gist.githubusercontent.com/sharanharsoor/84f4dbd4913106d06f8b8156fa8f47a9/raw/6735e6fc9436a3ef893b17ea445d1c7cf6db8676/runkite-semantic-router.patch' \
  -o ../runkite-semantic-router.patch
git apply --check ../runkite-semantic-router.patch
git apply ../runkite-semantic-router.patch
```

From the repository root:

```bash
CGO_ENABLED=0 go build -o ./runkite ./cmd
npm --prefix typescript/runkite-runner ci
npm --prefix examples/semantic_router_ai_sdk ci
docker compose -p runkite-sr-lab -f examples/semantic_router_ai_sdk/compose.yaml up -d
docker compose -p runkite-sr-lab -f examples/semantic_router_ai_sdk/compose.yaml exec -T ollama ollama pull qwen3:0.6b
docker compose -p runkite-sr-lab -f examples/semantic_router_ai_sdk/compose.yaml exec -T ollama ollama pull qwen3:1.7b
RUNKITE_BINARY="$PWD/runkite" python3 examples/semantic_router_ai_sdk/verify.py --extended
```

The verifier prints a report path and a top-level `status: passed` when the scenario completes. Open that report and check the individual results: three visible main Runs, two incident reads, zero upstream restarts, and passed `router_outage` and `cancellation` checks. A cancellation that loses the race to completion is inconclusive, not a passed cancellation test.

The tested stack uses Runkite runner **0.6.1**, AI SDK **7.0.155**, `@ai-sdk/openai` **4.0.104**, Semantic Router **0.4.0**, Envoy **1.35.3** and Ollama **0.34.3**. Compose pins the container digests. Stable Semantic Router uses the separate Envoy inference frontend; current development documentation may describe a different deployment mode.

The harness creates its own authenticated Runkite plane, SQLite database, runner and loopback MCP fixture, then stops those processes. It leaves the Compose stack running for inspection. Inference is published at `127.0.0.1:18899`; the router management API and Ollama remain on the private container network. The Go dev listeners bind all interfaces and use ephemeral client and runner credentials.

Read the generated report before the process logs. It records selected models, native response identities, durable completion, history, fixture counters and the two extended checks. Raw runtime files contain ephemeral credentials and conversation data; publish the sanitized report rather than the entire artifact directory.

When finished, stop this example's stack:

```bash
docker compose -p runkite-sr-lab -f examples/semantic_router_ai_sdk/compose.yaml down
```

## Bring your own agent next

The incident began as a read, became a diagnostic follow-up, and ended with an action the agent was not allowed to take. Across those turns, the model route changed, the runner restarted, and the conversation and connector permissions still had a clear owner.

That gives you a practical starting point for your own agent: choose information it may read and an operation it must not perform. Replace the example model bindings with your compatible model pool, choose a routing policy, and check the actual route, the durable conversation and the service that would receive the action. Check complete streaming tool calls when changing backends, just as the local test does.

This integration extends the operating model I wanted when I built Runkite: keep the native SDK's agent logic, give model selection to Semantic Router, and keep agent execution and connector authority inspectable on infrastructure you own.

**[Run the example](https://gist.github.com/sharanharsoor/84f4dbd4913106d06f8b8156fa8f47a9)**, explore **[Runkite on GitHub](https://github.com/getrunkite/runkite)**, or visit **[getrunkite.com](https://getrunkite.com/)** for the project and documentation. If you try it with another model backend or hit a protocol edge, bring the reproduction and observed result to the project. That is how the next integration becomes easier to operate.
