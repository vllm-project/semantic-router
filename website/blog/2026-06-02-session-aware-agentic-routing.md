---
slug: "session-aware-agentic-routing"
title: "Session-Aware Agentic Routing: Continuity-Aware Model Selection for Long-Horizon LLM Agents"
description: "How Session-Aware Agentic Routing in vLLM Semantic Router preserves long-horizon agent continuity with session memory, safe model-switch boundaries, prefix-cache-aware switch pricing, and replayable traces."
authors:
  - name: "Xunzhuo Liu, Bowei He, Huamin Chen, Haichen Zhang (AMD), Andy Luo (AMD), and the vLLM Semantic Router Team"
    url: "https://github.com/vllm-project/semantic-router"
tags: ["ecosystem","performance","agentic-routing"]
image: "/img/blog/vllm/2026-06-02-session-aware-agentic-routing/hero-v2.png"
source_url: "https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-06-02-session-aware-agentic-routing.md"
---
Long-horizon LLM agents create a routing problem that single-turn prompt routers were not designed to solve. A router still needs to know which model is best for the current request, but it also needs to know when switching models would break the session.

This post introduces **Session-Aware Agentic Routing (SAAR)**, a session-aware model selection policy in vLLM Semantic Router. SAAR keeps semantic routing, but adds router-owned session memory, hard locks around tool loops and non-portable provider state, safe reset boundaries, prefix-cache-aware switch pricing, and replayable traces.

Across **21,600** deterministic turns, SAAR cuts model switches by **79.29%**, eliminates **3,836** unsafe switches, and reduces estimated physical-model cost by **78.71%**. Across **2,896** live AMD ROCm requests, it preserves session continuity with **0** observed violations.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/hero-v2.png" alt="Session-aware agentic routing overview" width="94%" />
</picture>
<br />
<em>Figure 1: Long-horizon agents need routing decisions that understand the session trajectory, not only the latest prompt.</em>
</p>

## From Prompt Routing To Session Routing

vLLM Semantic Router started from a simple systems observation: not every request should take the same path through an inference stack. A short factual question, a security-sensitive prompt, a multimodal request, a hard reasoning task, and a domain-specific query may all deserve different treatment.

The first generation of that idea was prompt routing. The router extracted signals from the current request, matched a routing decision, and selected an appropriate path. Iris made those signals composable. Athena made the router more strategic by expanding model selection, memory, replay, long-context signals, multimodal primitives, and AMD ROCm deployment paths.

Agents change the unit of routing again.

A coding or research agent is not one prompt. It is a session. It plans, calls tools, receives tool outputs, edits files, runs tests, recovers from errors, pauses, resumes, and often sends very short follow-up messages such as "continue", "fix it", "run that again", or "use the previous result." Those turns are meaningful only because of the trajectory that came before them.

That is why this milestone matters for Semantic Router. The router is no longer answering only:

> Which model should handle this request?

For agent traffic, the router also has to answer:

> Is it safe to switch models inside this session right now?

That second question is what SAAR is designed to handle.

## Why Single-Turn Routing Breaks Down For Agents

Single-turn routing can be locally correct and still be wrong for the session.

Consider a typical tool-using agent loop:

| Turn | What the client sends | What a prompt router sees | What a session router must remember |
|---|---|---|---|
| 1 | "Refactor this module and run the tests." | A coding task | The session has started on a physical model |
| 2 | The model emits a tool call | A model response | The next tool result belongs to the same model |
| 3 | The client sends the tool result | A terse observation | The model that asked for the tool should receive the result |
| 4 | The user says "fix the failing case" | A short follow-up | The instruction depends on prior code, test output, and routing state |
| 5 | The session idles and resumes later | A new short message | The router can reconsider whether the old model is still worth holding |

The latest message alone does not contain enough information. A prompt router may decide that the tool result looks cheap and send it to a smaller model. It may see a generic "continue" and re-run the normal selector. It may miss that provider-managed continuation state belongs to one physical backend. It may discard a warm prefix cache for a frontier model because the current message is short.

Each of those mistakes has a different failure mode:

- A tool result can go to a model that did not make the tool call.
- A non-portable continuation id can be sent to the wrong physical backend.
- A long, warm session can lose prefix locality and become unnecessarily expensive.
- A logical model such as `auto` can become hard to debug because users no longer know which physical model actually served the turn.

The important point is not that agents should never switch models. They should. A good router should still move from a cheap model to a stronger model when the task becomes harder, and it should move back when the session reaches a safe boundary. The problem is that the router needs session context to know which moments are safe.

## The SAAR Design

SAAR keeps the existing Semantic Router decision pipeline. Signals are still extracted from the request, decisions are still matched, and model-selection algorithms still rank candidate models inside a matched decision.

SAAR adds a session-control layer around that result.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/policy-flow.png" alt="Session-aware agentic routing policy flow" width="92%" />
</picture>
<br />
<em>Figure 2: SAAR combines router memory, hard locks, reset boundaries, switch economics, and replayable traces before selecting a physical model.</em>
</p>

There are five pieces:

| Piece | What it stores or decides | Why it matters |
|---|---|---|
| Router memory | Last physical model, matched decision, phase, switch count, idle time, cache evidence, and replay metadata | Gives the router session context without becoming application memory |
| Hard locks | Prevent switching during active tool loops or non-portable provider-managed state | Preserves correctness before optimizing cost or quality |
| Reset boundaries | Allow reselection after idle timeout or decision drift | Prevents session-aware routing from degrading into sticky sessions |
| Switch economics | Prices handoff cost, switch history, remaining-turn priors, and prefix-cache checkout | Makes switching asymmetric across model tiers and session lengths |
| Replay traces | Records why the router stayed, switched, or refused to switch | Makes a logical model such as `auto` inspectable |

This is a model-selection policy, not an endpoint load balancer. Semantic Router can choose a model or cluster through the gateway contract. Endpoint membership, health checks, and load balancing inside a cluster remain infrastructure responsibilities.

## The Most Important Rule: Sometimes The Router Must Not Switch

The safest model switch is not always the one with the best score on the latest prompt. For agent traffic, some turns are continuity-constrained.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/switch-boundaries.png" alt="Safe and unsafe model-switch boundaries" width="92%" />
</picture>
<br />
<em>Figure 3: Tool loops and provider-managed continuation state are hard continuity constraints; idle and decision-drift boundaries permit safe reselection.</em>
</p>

SAAR treats two cases as hard locks:

- **Tool-loop continuity.** If a physical model asked for a tool call, the tool result should return to that same physical model. The follow-up observation is not a fresh prompt; it is part of a local execution loop.
- **Provider-managed state.** If the request carries non-portable continuation state, such as a response identifier that belongs to one backend, SAAR holds the previous physical model instead of silently moving the state elsewhere.

These rules are intentionally stronger than cost rules. If a switch is unsafe, the router should not "buy" its way out with a cheaper model.

SAAR also defines the opposite boundary: when the router may switch again. Idle timeout and decision drift reopen the selection. If an agent pauses long enough, the value of continuity decays. If the matched decision changes because the user moved from code editing to synthesis or from retrieval to debugging, the old model choice should not stick forever.

This distinction is the heart of session-aware agentic routing:

| Situation | SAAR behavior | Reason |
|---|---|---|
| Tool call is waiting for a tool result | Hold the previous physical model | The tool result belongs to that model's local reasoning loop |
| Request carries non-portable provider state | Hold the previous physical model | The state may not be valid on another backend |
| Session has idled past the configured boundary | Allow reselection | Continuity pressure has decayed |
| Matched routing decision changes | Allow reselection | The task shape changed |
| Session is long and warm on an expensive model | Raise the switch threshold | Prefix locality is valuable |
| Cheap short retry on a small model | Lower the switch threshold | Checkout cost is small |

## Router Memory Is Not User Memory

The phrase "router memory" can be misleading, so the boundary is important.

SAAR memory is not conversation memory, retrieval memory, or user profile memory. It does not summarize the conversation and it does not try to remember facts for the model. Its job is narrower: keep enough routing state to make the next model-selection decision safe and explainable.

For each session, the router tracks facts such as:

- the last physical model selected behind the logical model;
- the last matched routing decision;
- whether the session is in a normal, tool-loop, provider-state, idle-reset, or drift-reset phase;
- how many recent switches happened;
- the latest context length and cache evidence;
- a replay id that links the response back to the router's decision trace.

That scope keeps the system operationally useful without turning the router into a second agent memory layer. Application memory should remain in the application. Retrieval memory should remain in the retrieval stack. SAAR memory exists only to make routing across turns coherent.

## Prefix Cache Makes Model Switching Asymmetric

For long agent sessions, model switching is not just a quality decision. It is also an input-side systems decision.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/cache-checkout-discipline.png" alt="Prefix-cache checkout discipline" width="92%" />
</picture>
<br />
<em>Figure 4: The same switch has a different cost depending on model tier, session length, and physical prefix reuse.</em>
</p>

A short retry on a cheap model and a 40-turn warm session on a frontier model should not be treated the same way. The latter has accumulated a valuable prefix. Switching away from it may require the next physical model to pay a much larger input cost even if the visible user message is short.

SAAR therefore prices a cached-input checkout delta: the gap between normal prompt input price and cached-input price for the physical model under consideration. The longer and more expensive the session, the stricter the policy becomes about discarding prefix locality.

This also clarifies cached-token accounting for a routed logical model. If the user calls `auto`, the router may map that logical name to different physical models over time. A cache hit reported by one backend is physical evidence for that backend. It is not automatically transferable to another backend. SAAR keeps backend-reported cached tokens separate from router-estimated reuse, and it does not rewrite upstream usage fields.

That separation is useful operationally. Operators can still inspect physical cache behavior while the router uses its own memory to decide whether switching is worth the checkout cost.

## How A Request Moves Through SAAR

The serving path stays familiar. Clients send requests to the OpenAI-compatible gateway, usually with a logical model name such as `auto`. To enable session-aware routing, they also send a stable session identifier such as `x-session-id`.

SAAR then handles each turn in this order:

1. Read the current request, session id, tool-call context, provider-state markers, and candidate model set.
2. Run the normal Semantic Router signal and decision pipeline.
3. Produce a base model-selection result from the configured method, such as hybrid scoring.
4. Load the previous session routing state from router memory.
5. Apply hard locks for tool loops and provider-managed state.
6. Check idle timeout and decision drift boundaries.
7. Adjust switch scores using prefix-cache checkout cost and switch history.
8. Select the physical model and emit diagnostics.
9. Update router memory and write a replay trace.

The configuration lives inside a routing decision's model-selection algorithm:

```yaml
routing:
  decisions:
    - name: agentic_routing
      modelRefs:
        - model: qwen3-8b
        - model: qwen3-32b
      algorithm:
        type: session_aware
        session_aware:
          base_method: hybrid
          idle_timeout_seconds: 300
          tool_loop_hard_lock: true
          context_portability_hard_lock: true
          decision_drift_reset: true
          prefix_cache_weight: 0.20
          switch_history_weight: 0.04
```

The values are intentionally policy knobs, not one-size-fits-all constants. A customer-service assistant with short sessions may use a more permissive idle boundary. A coding agent with long tool loops and expensive context may use stricter continuity and prefix-cache settings.

## Observability Is Part Of The Feature

Model selection behind `auto` is only useful if operators can explain it.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/observability-trace.png" alt="Session-routing observability trace" width="92%" />
</picture>
<br />
<em>Figure 5: SAAR turns hidden physical routing choices behind a logical model into inspectable traces and response headers.</em>
</p>

SAAR emits diagnostics such as selected model, selected decision, replay id, session phase, selected confidence, and context-token count. The replay id joins a served response back to the router trace that explains the decision.

A useful trace answers questions like:

- What model would the base selector have chosen?
- Did the router hold the previous model because of a tool-loop lock?
- Did provider-managed state make switching unsafe?
- Did the session cross an idle or drift boundary?
- How did prefix-cache evidence change the adjusted candidate scores?
- Was the final decision a stay, a switch, or a locked stay?

This makes session-aware routing operable. Without replay, a router behind a logical model becomes hard to debug. With replay, an operator can audit why the router preserved continuity or decided that a switch was safe.

## How We Evaluate It

The evaluation is designed around one question: does the policy make routing more agent-friendly without hiding correctness problems?

We use three layers of evidence.

First, a deterministic policy matrix tests the control logic across many synthetic sessions. This isolates the routing policy from serving noise and lets us stress tool loops, provider state, idle boundaries, drift boundaries, model tiers, and switch history.

Second, live OpenAI-compatible serving runs exercise the same invariants through the router and backend serving path on AMD ROCm. This checks that headers, session ids, diagnostics, and failure handling survive real request flow.

Third, deterministic agent-task traces add task structure. Instead of only counting switches, these traces include simulated tool observations and exact final-answer scoring.

The goal is not to make every plot say "fewer switches." Sticky sessions can do that. The goal is to show that SAAR removes unsafe switches, keeps useful movement, respects expensive prefix locality, and remains observable in live serving.

## Result 1: SAAR Moves The Unit Of Control From Turn To Session

The deterministic policy matrix covers balanced, tool-heavy, frontier-heavy, idle-heavy, provider-state-heavy, and drift-heavy sessions. Each workload runs five seeds, 40 sessions per seed, and 18 turns per session, for **21,600** total turns.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/synthetic-headline.png" alt="Deterministic session-routing benchmark results" width="92%" />
</picture>
<br />
<em>Figure 6: Headline policy result across 21,600 deterministic turns.</em>
</p>

The headline result is that SAAR reduces model churn while preserving the ability to move:

| Policy | Switches | Unsafe switches | Estimated cost reduction | Quality delta |
|---|---:|---:|---:|---:|
| Single-turn | 9,709 | 3,836 | 0.00% | +0.0000 |
| Sticky session | 340 | 0 | 98.65% | -0.1433 |
| Initial SAAR | 1,810 | 200 | 70.92% | -0.0122 |
| Full SAAR | 2,011 | 0 | 78.71% | -0.0453 |

Single-turn routing switches often and creates unsafe movement. Sticky sessions nearly eliminate movement, but they also give up too much quality because they refuse to reselect after the task changes. Full SAAR sits in the middle for the right reason: it removes unsafe movement while still letting idle and drift boundaries reopen the decision.

This is the shift from turn-level control to session-level control. The router is no longer treating every message as a fresh independent event.

## Result 2: Hard Locks Remove The Correctness Failures

The second result isolates the most important invariant: when switching is unsafe, SAAR should not switch.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/safety-effect.png" alt="Hard-lock safety effect" width="92%" />
</picture>
<br />
<em>Figure 7: Hard locks remove unsafe switching during tool loops and non-portable provider state.</em>
</p>

Tool-loop switch violations fall from **3,404 to 0**. Provider-state switch violations fall from **432 to 0**.

These are not minor tuning wins. They are correctness boundaries. A tool result is not an ordinary prompt. A non-portable continuation id is not an ordinary text field. If a router ignores those facts, it can break the interaction while still appearing to make a reasonable semantic choice on the latest message.

SAAR fixes that by making continuity constraints explicit in the policy.

## Result 3: SAAR Is Not Sticky Sessions With A New Name

The obvious baseline is sticky routing: pick the first model for a session and hold it.

Sticky routing is attractive because it is easy to reason about. It also solves many unsafe switch cases by avoiding switching entirely. But that simplicity becomes a product problem for agents. Long sessions drift. Users change tasks. A cheap initial model may no longer be appropriate. A strong model may no longer be necessary.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/ablation-effect.png" alt="Session-aware routing ablation against sticky routing" width="92%" />
</picture>
<br />
<em>Figure 8: SAAR is not just sticky sessions; it balances continuity with movement.</em>
</p>

The ablation shows why SAAR needs multiple mechanisms:

| Variant | Switch reduction | Unsafe switches | Cost reduction | Interpretation |
|---|---:|---:|---:|---|
| No tool lock | 74.96% | 760 | 60.05% | Reintroduces tool-loop violations |
| No provider-state lock | 77.98% | 200 | 69.82% | Reintroduces non-portable-state violations |
| No drift reset | 83.14% | 0 | 81.31% | Over-sticks after task drift |
| No idle boundary | 83.98% | 0 | 80.14% | Over-sticks after natural pauses |
| No frontier cost | 73.96% | 0 | 54.75% | Switches away from expensive warm sessions too easily |
| Full SAAR | 79.29% | 0 | 78.71% | Preserves locks while retaining safe reselection |

Locks provide correctness. Reset boundaries provide liveness. Prefix-cache checkout pricing provides economic discipline. Removing any one of them changes the behavior in a way that is visible in the metrics.

## Result 4: The Invariants Hold In Live AMD ROCm Serving

Policy simulation is useful, but a router has to work through real request flow. The live serving runs use OpenAI-compatible traffic through the router and AMD ROCm backend paths, with matched schedules for routed and direct-backend runs.

<p align="center">
<picture>
<img src="/img/blog/vllm/2026-06-02-session-aware-agentic-routing/live-rocm-effect.png" alt="Live ROCm session-routing results" width="92%" />
</picture>
<br />
<em>Figure 9: Live ROCm runs preserve continuity under long sessions and injected backend failures.</em>
</p>

Across long-session runs, the router completes **2,896** live requests with **0** observed continuity violations.

| Workload | Requests | Success rate | p95 overhead | Continuity violations |
|---|---:|---:|---:|---:|
| balanced-32x64 | 2,048 | 100.00% | 6.181 ms | 0 |
| stateful-16x48 | 768 | 100.00% | 26.805 ms | 0 |
| idle-16x5-75s | 80 | 100.00% | 283.463 ms | 0 |

The idle workload includes real wall-clock sleeps, so its p95 overhead should be interpreted separately from hot-path routing overhead. The important result is continuity: the live path preserves the hard-lock and reset-boundary behavior tested in the deterministic matrix.

## Result 5: Sessions Recover After Backend Faults

Long-horizon agents need session-level recovery, not just per-request success. A backend can return HTTP 503 for one request, but the session should continue later without losing the routing invariants that keep the interaction coherent.

| Fault phase | Requests | Injected 503s | Affected sessions | Recovery | Continuity violations |
|---|---:|---:|---:|---:|---:|
| provider state | 360 | 48 | 8 | 100.00% | 0 |
| tool loop | 360 | 72 | 8 | 100.00% | 0 |
| topic drift | 432 | 48 | 8 | 100.00% | 0 |

Across the one-shot disruption matrix, **32/32** affected sessions recovered later. Across the repeated-failure matrix, **24/24** affected sessions recovered after **168** injected HTTP 503 responses.

This matters because agent sessions are longer than ordinary chat turns. A transient backend fault should not make the router forget that a tool loop is active, that provider state is non-portable, or that the session has a replayable history.

## Result 6: Task Traces Exercise The Agent Loop

Continuity counters are necessary, but they are not enough. We also run deterministic multi-turn task traces with simulated tool observations and exact final-answer scoring. There is no judge model in the loop: the final answer either contains the required labels or it does not.

In the AMD serving task run, **18/18** exact-scored task instances complete, replay headers are present on **96/96** routed turns, and no continuity violation is observed.

This is still smaller than a broad real coding-agent benchmark, but it is a stronger signal than policy counters alone because it exercises a task loop with tool observations and final-answer checks.

## What This Changes For vLLM Users

Session-aware routing makes Semantic Router more useful for agent-serving stacks where a logical model name hides a model portfolio.

For users, the experience can remain simple: call a model such as `auto`, send a stable session id, and let the router pick the physical model. For operators, the behavior becomes more controllable: configure when continuity is required, when idle sessions can reset, how much prefix locality matters, and how routing decisions are traced.

This is especially useful when:

- candidate models have different cost, latency, and capability profiles;
- agents use tools across multiple turns;
- clients depend on provider-managed continuation state;
- long sessions build valuable prefix-cache locality;
- operators need to inspect which physical model served each turn behind a logical route.

It also creates a clean infrastructure boundary. Semantic Router owns policy-level model selection. Envoy, Kubernetes, and serving backends still own endpoint membership, health checks, and load balancing. That separation keeps SAAR focused on what it can safely decide: model continuity, model switching, and traceability at the session level.

## The Larger Direction

This milestone continues the same arc that started with Signal-Decision routing.

Iris made routing decisions composable. Athena moved Semantic Router toward a strategic system brain for mixture-of-models and agentic deployments. Multimodal hardening broadened the evidence surface from text prompts to request-level signals. Session-aware agentic routing broadens the time horizon: the router now reasons not only about a request, but about where that request sits inside a long-running interaction.

That direction matters for modern serving stacks. Agent systems increasingly want one logical model interface over many physical options. They want cheaper models for easy steps, stronger models for hard steps, continuity through tool loops, cache-aware handling of long contexts, and enough observability to trust the system in production.

SAAR is a step toward that operating model. It does not make the router an agent. It makes the router aware of the minimum session facts required to serve agents well.

The core idea is simple: a router behind `auto` should know when a model switch is allowed, when it is forbidden, and what the switch costs for a warm long-running session.

## Join Us

**Looking for collaborations!** SAAR is the next step in making Semantic Router useful for long-horizon agents, and there is a lot of open work ahead.

We are looking for contributors who want to help with:

- session-aware routing policies for real agent traffic;
- multi-turn and tool-loop evaluation suites;
- AMD ROCm serving validation and performance experiments;
- router observability, replay traces, and production debugging workflows;
- Envoy, Kubernetes, and gateway integrations that keep routing policy separate from endpoint load balancing.

If you are building agent systems, running model portfolios on AMD GPUs, or researching continuity-aware model selection, we would like to collaborate.

Resources:

- GitHub: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- Documentation: [vllm-semantic-router.com](https://vllm-semantic-router.com)
- Community: join the **#semantic-router** channel on [vLLM Slack](https://vllm-dev.slack.com/archives/C09CTGF8KCN)
