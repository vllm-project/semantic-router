---
slug: multiturn-model-switching
title: "Beyond Prompt Routing: Model Selection, Conversation State, and KV Cache"
description: "How to compose semantic routing with an inference router, preserve agent continuity, and measure the quality and cache costs of switching models."
authors: [anupsharma]
tags: [evaluation, agentic, mixture-of-models, routing, vllm, architecture]
image: /img/vllm-sr-logo.social.png
---

An agent has completed three turns on model A. It has read a document, stored
some facts, and started working toward an answer. The fourth turn needs stronger
reasoning, so the router sends it to model B.

The request succeeds. But did the conversation improve?

Model B may receive the entire transcript and still inherit an incorrect answer
from an earlier turn. Its serving replica may have to compute the growing prompt
again. If a tool call is in flight, changing the model can also change how the
result is interpreted. Meanwhile, an inference router might prefer a different
replica because it has a shorter queue or a useful cached prefix.

These questions motivated a series of live benchmarks through vLLM Semantic
Router, Envoy, and GPU-backed vLLM servers. In the final 960-call evaluation,
one reactive switching policy reduced strong-model use by **31.8 percentage
points** and mean session latency by **550 ms**. It also failed three final checks
that the control passed. A separate native-tool experiment showed why moving
to a capable model *before* a known difficult boundary can help.

The numbers expose a design problem worth solving: how do we choose the right
model, preserve the conversation, and still benefit from efficient inference?

<!-- truncate -->

## A conversation carries more than a prompt

A single-turn router can optimize a request in isolation. An agent accumulates
dependencies: the next step may rely on an earlier answer, a tool call, or state
held by a provider. “Keep the session together” therefore needs a more precise
meaning.

| State | What it contains | What a switch must account for |
|---|---|---|
| Conversation history | Messages, instructions, answers, and tool results | The destination needs the complete, correctly ordered context—and earlier mistakes remain in it. |
| Tool ownership | The model that issued a call and the pending result | A compatible continuation must consume the result without breaking the call/result sequence. |
| Provider-managed state | Response lineage or other server-held context | Resolve or reconstruct that state before moving to a backend that cannot access it. |
| KV cache | Computed attention keys and values for a token prefix | Reuse requires compatible computation and available cache blocks; a session ID alone supplies neither. |

This distinction matters even when the API looks stateful. If a router stores
history and expands a continuation into a complete stateless request, the
transcript can become portable. A provider-held identifier that cannot be
resolved elsewhere has different constraints. In Semantic Router, router-owned
Responses history can become portable after expansion, while active tool loops
and nonportable context retain hard ownership constraints. See the
[continuity protection contract](/docs/tutorials/learning/protection).

Portability also does not establish correctness. A stronger model can read the
same history perfectly and still carry an earlier error forward.

## Two routing decisions, two kinds of evidence

There are two useful questions in a fleet of models:

1. **Which logical model is eligible and suitable for this work?** Semantic
   Router evaluates request signals, policy, model capabilities, and configured
   conversation protection.
2. **Which serving endpoint should execute it?** An inference router selects a
   replica within the admitted pool, using the scheduling signals its deployment
   supports, such as health, queue pressure, and cache availability.

The selected pool is the contract between these layers. Suppose semantic policy
requires model B. A warm replica of model A is outside the eligible pool; its
cache advantage does not make it a valid destination. Within model B's pool,
the inference router can balance a warm prefix against queueing delay.

One inference router used in this project is **llm-d**. Its integration uses
Gateway API routes and InferencePools to express this boundary. The
[Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
describes the gateway passing a request to the selected pool's endpoint picker.
Other inference routers can fit the same design if they preserve the eligible
model set and expose their serving decisions.

<figure style={{width: '100%', maxWidth: '42rem', marginInline: 'auto'}}>
  <a href="/img/blog/multiturn-routing/routing-architecture.svg" target="_blank" rel="noopener noreferrer" aria-label="Open the routing architecture diagram at full size">
    <img
      src="/img/blog/multiturn-routing/routing-architecture.svg"
      alt="An agent enters an AI gateway. Semantic Router admits model B using request and conversation context. The inference router selects replica B1 inside that pool; each replica holds its own KV cache. Model A remains outside the selected pool."
      width={1000}
      height={800}
      loading="lazy"
      decoding="async"
      style={{maxHeight: 'none'}}
    />
  </a>
  <figcaption style={{fontSize: '0.875rem', lineHeight: 1.65}}>
    Model selection establishes the eligible pool. Endpoint selection schedules
    within it. Click to open the full diagram.
  </figcaption>
</figure>

This is where Semantic Router adds value to an existing serving stack: it can
make capability, policy, and conversation context part of the decision before
the request reaches the replica scheduler. The inference router retains the
serving knowledge needed to execute that decision efficiently.

## KV cache makes switching a systems decision

In ordinary transformer serving, KV cache contains intermediate computations
for a particular model and token sequence. Sending the same transcript to a
different model does not transfer those computations. The destination needs
its own compatible cached prefix or must prefill the prompt.

Even staying on the same model does not guarantee a hit: another replica may
not have the prefix, cache blocks may have been evicted, or the rendered prompt
may have changed. Cache sharing or transfer between compatible engines must be
an explicit serving capability; cross-model reuse cannot be assumed.

[vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
reuses computation for matching prefixes. Its benefit is in prefill, rather
than generating new output tokens. That makes long shared prefixes attractive
for reuse, but cache-hit ratio alone cannot explain total latency.

A practical switching decision weighs several terms:

```text
expected task-quality gain
  versus
extra prefill + queueing + handoff/repair + model execution cost
```

Hard eligibility and ownership constraints come first. Among legal choices,
the tradeoff depends on how much work remains. Paying to rebuild a long prefix
may be worthwhile before ten difficult steps; it may be wasteful for a final
acknowledgement. These are quantities to calibrate on the application's tasks.

Semantic Router's protection machinery has switch margins, continuity costs,
and recent-outcome gates for this purpose. A gate checks a proposed switch; it
does not independently discover the best model. Its quality evidence also needs
an owner: producing tokens is not proof that a turn was correct.

## An architecture for an existing gateway and model fleet

Start by assigning responsibilities, even if several run in the same gateway
process. Semantic Router is an Envoy external processor; the diagram's boxes
represent decision boundaries rather than a requirement for separate proxies.

| Layer | Responsibility | Integration contract |
|---|---|---|
| Client or agent | Own the task, history, and tool execution | Send usable conversation identity and valid message/tool sequences. |
| AI gateway | Authenticate requests and apply tenant limits and request policy | Preserve context needed by routing and carry the selected destination to the serving layer. |
| Semantic Router | Choose an eligible logical model and apply continuity protection | Record the proposal, final selection, and reason for retaining or switching models. |
| Inference router | Select a healthy endpoint within the admitted model pool | Respect pool membership and expose endpoint choice and available scheduling evidence. |
| Model server | Execute inference and manage its cache | Report actual token usage, latency, and available cache measurements. |

There are four integration details to make explicit.

**Preserve identity across turns.** Establish how the gateway supplies session
and conversation identity, how it is scoped to a tenant, and where history is
stored. Test missing identity: Semantic Router's protection records diagnostics
and fails open when configured identity headers are absent, so silently dropping
them can remove expected continuity behavior.

**Carry model selection through to the pool.** If an existing gateway already
has model routing, define which decision takes precedence. Map the admitted
logical model to its backend pool and assert that the endpoint picker selects
inside that pool. Multiple models or adapters in one serving pool require an
additional capability filter; pool membership alone is then insufficient.

**Define failure behavior at protected boundaries.** If the owner of an active
tool loop becomes unavailable or excluded by policy, specify what the agent
does next. Hard ownership and eligibility can conflict; Semantic Router's apply
path can return a selection error instead of transferring that state. Recovery
may require an explicit restart or repair. Retrying a side-effecting tool also
requires the application's idempotency contract.

**Join decisions with outcomes.** Keep a request trace connecting the selected
model, serving endpoint, routing reason, usage, and task verdict. Semantic
Router's Replay records provide routing evidence; backend metrics provide
serving evidence. Feed quality outcomes from an evaluator or task owner, rather
than inferring success from HTTP 200 or output length.

This lets an application inspect a slow or failed turn and distinguish a
capability mismatch, an ownership decision, a cold prefix, and a busy replica.

## When should the conversation change models?

Consider a tool boundary. If the next step requires a capability the current
model lacks, selecting a capable owner *before* it issues the tool call gives
that model both sides of the interaction. Once the call is active, preserving
ownership avoids changing its continuation semantics halfway through.

In a controlled native-tool benchmark, we compared this prospective choice
with escalation after observed failure. Across eight paired seeds, selecting
the stronger model at the known boundary improved tool-call success from
62.5% to 100%, tool-continuation success from 0% to 75%, and final success from
37.5% to 100%. The cost was 52.5 percentage points more strong-model turns,
11.35 points lower measured prefix-cache hit ratio, and 1,275 ms more per session.

The boundary signal was supplied explicitly by the test fixture. This establishes
why timing and ownership matter; learning to predict such boundaries remains
application-specific work. Exercising the live path also revealed an ownership
preservation defect, which was corrected before the later evaluation.

A recent-outcome gate faces another problem. It may wait for a failure before
allowing escalation. By then, the agent has already appended the incorrect
answer to its history.

<figure style={{width: '100%', maxWidth: '42rem', marginInline: 'auto'}}>
  <a href="/img/blog/multiturn-routing/conversation-handoff.svg" target="_blank" rel="noopener noreferrer" aria-label="Open the conversation handoff diagram at full size">
    <img
      src="/img/blog/multiturn-routing/conversation-handoff.svg"
      alt="Six turns illustrate the observed late escalation: the small model fails turn three; the strong model takes turns four through six with the full history, including the earlier error. A proposed recovery path verifies or repairs the failed step before continuing."
      width={1000}
      height={640}
      loading="lazy"
      decoding="async"
      style={{maxHeight: 'none'}}
    />
  </a>
  <figcaption style={{fontSize: '0.875rem', lineHeight: 1.65}}>
    Escalation changes the executor, but leaves earlier answers in the transcript.
    Repairing the failed step is a proposed recovery strategy, not a measured result
    of this evaluation.
  </figcaption>
</figure>

Recovery needs an explicit design: verify the suspicious answer, reconstruct
task state from authoritative inputs, or replay the failed step when safe. That
is different from sending the next ordinary turn to a stronger model.

## What the live measurements showed

The final holdout ran **160 sessions and 960 calls** through Semantic Router,
Envoy, and two vLLM servers hosting Qwen3-0.6B and Qwen3-8B on one NVIDIA L40S.
Each policy received the same 16 held-out seeds across two six-turn workloads:
ordered state retention and a one-off retrieval challenge followed by easy
acknowledgements. The candidate thresholds were frozen before inspecting outcomes.

| Policy | Final-turn success | Turn accuracy | Strong-model turns | Prefix-cache hit ratio | Mean latency/session |
|---|---:|---:|---:|---:|---:|
| Strong model throughout | 100% | 66.7% | 100% | 80.8% | 1,793 ms |
| Small model throughout | 50.0% | 53.6% | 0% | 80.8% | 456 ms |
| Continuity protection, gate disabled | 100% | 66.1% | 83.3% | 66.1% | 1,771 ms |
| Candidate gate, observe | 100% | 66.7% | 83.3% | 65.9% | 1,694 ms |
| Candidate gate, enforce | 90.6% | 69.3% | 51.6% | 64.8% | 1,221 ms |

“Final-turn success” is the task-defined last-turn check, not success on every
step. In the state-retention workload it checks the ordered final synthesis;
in the one-off challenge workload it checks the closing acknowledgement. The
latter can pass despite a missed retrieval challenge, which is why turn accuracy
must be reported alongside it. Each arm contains 32 sessions.

Compared with the gate-disabled control, enforcement reduced strong-model use
by 31.8 percentage points and mean latency by 550 ms. Final-turn success fell
from **32/32 to 29/32**. The paired bootstrap 95% interval for that difference
was [-21.9, 0.0] percentage points; the turn-accuracy difference was also
inconclusive. The candidate failed its quality requirement and remains rejected
for enforcement.

All three final failures were in state-retention sessions with the schedule
`small → small → small → strong → strong → strong`. After the small model failed
turn three, the strong model continued with that answer in its context. Those
sessions produced incorrectly ordered values at final synthesis. The traces
are consistent with late escalation carrying damaged state forward; a repair
experiment is needed to isolate that mechanism causally.

The holdout used controlled keyword signals and static proposals to isolate
continuity and gate behavior. It did **not** measure semantic-classifier accuracy.
Calls were sequential, with short outputs and deterministic task checks; the
latency figures are not production throughput or concurrency estimates.
Prefix-cache ratios came from backend counter deltas, averaged per session.
Strong-model share is a usage proxy, rather than a measured dollar saving.

For endpoint composition, we added a separate simulator-backed integration
contract using llm-d as the inference router. It checks selected model, pool,
and endpoint membership with at least two ready replicas per pool. That test is
under review; it supplies no real-GPU prefix-affinity or scheduling-performance
measurement.

## Benchmarks to run on your own system

Build the evaluation around complete agent tasks before tuning switch thresholds.
Include long shared prefixes, changing instructions, retrieval, native tools,
and tasks where an early error affects later steps. Use realistic task verifiers
such as repository tests, validated tool arguments, or checked structured state.

Compare a capable model held for the session, a small model held for the session,
per-turn routing, and routing with continuity protection. Add cache-aware
endpoint selection as a separate serving comparison. Run proposed gates in
observe mode before enforcement so you can distinguish recorded decisions from
applied changes.

| Test | Question it should answer | Evidence to collect |
|---|---|---|
| Complete-task evaluation | Does routing finish the work correctly? | Task success, intermediate errors, and paired uncertainty. |
| Tool-loop continuation | Does the call/result sequence survive routing? | Tool schema validity, owner changes, continuation and final success. |
| History portability | Can the selected backend actually resume? | Full-history and stored-history paths, unknown IDs, and backend ownership. |
| Pool-to-endpoint composition | Is the request served by an eligible replica? | Logical model, pool, endpoint identity, and capability checks. |
| Warm/cold prefix runs | Is cache-aware scheduling helping? | Backend cached tokens or blocks, TTFT, queue time, and end-to-end latency. |
| Load and failure runs | What happens under contention or owner loss? | Latency percentiles, errors, retries, failover decisions, and tool side effects. |

Use paired tasks and seeds, freeze development-fitted thresholds, and retain an
untouched holdout. Vary policy order to limit warm-up effects and control shared
prefixes across runs. Under concurrent load, process-wide cache counters cannot
attribute hits to a single request; use request-level accounting where available
or report the measurement as aggregate.

Set acceptance requirements before looking at results. A lower latency average
cannot compensate for broken tool ownership or lost task correctness. Likewise,
100% completion on a small controlled fixture does not establish production
reliability. Promote only the policies whose tradeoffs meet the application's
requirements, with a rollback path and continuing outcome measurement.

## Build for the next turn and the rest of the task

Semantic Router can bring request meaning, model capability, and conversation
protection into an inference architecture. The inference router can then make
the admitted choice efficient on the serving fleet. Together, they let a system
reason about both *what should do the work* and *where it should run*.

Our measurements show why both layers need task-level evaluation. A prospective
tool-boundary choice improved continuity at a cost. A reactive gate saved
strong-model turns and time, but failed to preserve final outcomes. The next
design needs better boundary signals and explicit recovery of failed state.

For an existing model fleet, begin with eligible pools, reliable conversation
identity, and observable continuity decisions. Then measure model switching and
endpoint scheduling independently, and evaluate their combined effect on the
agent's completed work.

### Implementation and measurement references

- [Semantic Router continuity protection](/docs/tutorials/learning/protection)
  and [memory and Replay](/docs/tutorials/learning/memory-and-replay).
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
  and [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/).
- [Evaluation record and holdout results](https://github.com/vllm-project/semantic-router/issues/4080#issuecomment-5938865660).
  Router revision: `0ee9955fc6ae0d79104e33bc933ae0f75ed52acf`;
  Qwen3-0.6B: `c1899de289a04d12100db370d81485cdf75e47ca`;
  Qwen3-8B: `b968826d9c46dd6066d109eabc6255188de91218`.
