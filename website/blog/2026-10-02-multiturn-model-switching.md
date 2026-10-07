---
slug: multiturn-model-switching
title: "Beyond Prompt Routing: Model Selection, Conversation State, and KV Cache"
description: "960 live calls reveal what model switching saves and what it can break. An architecture for combining semantic routing, conversation protection, and cache-aware inference."
authors: [anupsharma, 1fanwang]
tags: [evaluation, agentic, mixture-of-models, routing, vllm, architecture]
image: /img/vllm-sr-logo.social.png
---

An agent has completed three turns on model A. It has read a document, stored
some facts, and started working toward an answer. The fourth turn needs stronger
reasoning, so the router sends it to model B.

The request succeeds. But did the conversation improve?

Model B may receive the entire transcript and still inherit an incorrect answer.
Its replica may have to compute the growing prompt again. If a tool call is in
flight, changing models can also change how the result is interpreted.

We benchmarked these tradeoffs through vLLM Semantic Router, Envoy, and GPU-backed
vLLM servers. In the [960-call holdout](https://github.com/vllm-project/semantic-router/issues/4080#issuecomment-5938865660), one reactive switching policy reduced
strong-model use by **31.8 percentage points** and mean session latency by
**550 ms** compared with a gate-disabled control, but failed three final checks
that the control passed. A separate
native-tool experiment showed the benefit of choosing a capable model *before*
a difficult boundary.

This is the problem beyond prompt classification: choosing the right model for
the next turn without compromising the rest of the task. It also determines
where semantic routing belongs in an existing inference stack.

<!-- truncate -->

## A conversation carries more than a prompt

Four kinds of state matter when a conversation changes models:

| State | What a switch must preserve or rebuild |
|---|---|
| History | Complete, correctly ordered messages and tool results. Earlier mistakes travel too. |
| Tool ownership | A compatible continuation of the model's pending call/result sequence. |
| Provider-managed state | Access to stored context, or reconstruction before moving to another backend. |
| KV cache | Compatible prefix computations available at the destination, not just a session ID. |

Router-owned Responses history can become portable after expansion into a full
request. That does not make all state portable: active tool loops and nonportable
context retain ownership constraints under the
[continuity protection contract](/docs/tutorials/learning/protection).
Nor does portable history guarantee correct history.

## Two routing decisions, two kinds of evidence

**Semantic Router chooses the logical model.** Request signals, capabilities,
policy, and configured continuity protection determine which model may do the work.

**An inference router chooses the serving endpoint.** It schedules within that
model's admitted pool, using supported signals such as health, queue pressure,
and cache availability.

These choices need not conflict. If semantic policy requires model B, a warm
replica of model A is not eligible. The endpoint picker can prefer a warm replica
*within B's pool*, but must not override the model decision merely for a cache hit.

One inference router used in this project is **llm-d**. Its integration uses
Gateway API routes and InferencePools to express this boundary. Other inference
routers can follow the same contract: preserve the eligible model set, then
schedule within it.

## KV cache makes switching a systems decision

The transcript can travel; model A's KV cache does not become model B's cache.
Those attention computations belong to a particular model and token sequence.
The destination needs its own compatible cached prefix or must prefill the prompt.

Even staying on one model does not guarantee reuse: another replica may lack the
prefix, blocks may have been evicted, or the rendered prompt may have changed.
Cache sharing requires explicit serving support.
[vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
reduces matching-prefix prefill work, not the work of generating new output tokens.

Among choices that satisfy eligibility and ownership constraints, switching weighs:

```text
expected task-quality gain
  versus
extra prefill + queueing + handoff/repair + model execution cost
```

Rebuilding a long prefix may be worthwhile before ten difficult steps, but not
for a final acknowledgement. Semantic Router's protection machinery exposes
switch margins, continuity costs, and recent-outcome gates to check proposed
switches. These gates still need meaningful quality feedback: HTTP 200 is not
evidence that the task step was correct.

## An architecture for an existing gateway and model fleet

For a fleet with many models and thousands of replicas, keep **model choice**
separate from **replica scheduling**. The request flow below illustrates one
request selecting model B, then replica B1, rather than searching every pod indiscriminately.

<figure style={{width: '92%', maxWidth: '100%', marginInline: 'auto', textAlign: 'center'}}>
  <a href="/img/blog/multiturn-routing/routing-architecture.svg" target="_blank" rel="noopener noreferrer" aria-label="Open the routing architecture diagram at full size">
    <img
      src="/img/blog/multiturn-routing/routing-architecture.svg"
      alt="An agent sends history and tools through an AI gateway. Semantic Router uses request and conversation context to admit model B. The inference router selects B1 within that pool; model A is excluded. Each replica has its own KV cache."
      width={1000}
      height={800}
      loading="lazy"
      decoding="async"
      style={{width: '100%', height: 'auto', maxHeight: 'none'}}
    />
  </a>
  <figcaption style={{fontSize: '0.875rem', lineHeight: 1.65, marginTop: '0.75rem'}}>
    Figure 1: Choose an eligible model, then schedule inside its pool.
    These are logical boundaries; they may share a gateway process.
  </figcaption>
</figure>

The agent supplies history and tool state. The gateway authenticates the request
and preserves tenant-scoped conversation identity. Semantic Router, an Envoy
external processor, selects an eligible model and applies continuity protection.
The inference router picks a replica; the model server executes the request.
The response returns to the agent, which owns tool execution and the next turn.

Three integration choices make this composition reliable:

1. **Choose who owns context.** Preserve conversation identity across turns and
   decide where stored history is resolved. Test missing identity headers:
   protection can fail open when its configured headers are absent.
2. **Make the model-to-pool contract explicit.** Carry the final model selection
   through the gateway. Assert endpoint membership and, for shared pools,
   model/adapter compatibility. A protected owner becoming unavailable needs
   explicit recovery, not an arbitrary fallback halfway through a tool loop.
3. **Trace the whole request.** Join routing reason, final model, endpoint,
   token usage, cache evidence, latency, and task verdict. This separates a
   capability mismatch from a cold prefix or a busy replica.

## What the live benchmarks showed

The holdout for the frozen candidate, `agentic-context/multiturn-switch@2026-10-dev1`, ran
**160 sessions and 960 calls** using Qwen3-0.6B and Qwen3-8B
on two vLLM servers sharing one NVIDIA L40S, behind Envoy and Semantic Router.
Each of five policies received the same 16 held-out seeds across two six-turn
workloads: ordered state retention, and a one-off retrieval challenge followed
by easy acknowledgements. Thresholds were frozen before inspecting the holdout.

A recent-outcome gate can delay a proposed switch until observed outcomes justify
it. **Observe** records its verdict without applying it; **enforce** applies it.

| Policy | Final-turn pass rate | Turn accuracy | Strong-model turns | Prefix-cache hit ratio | Mean latency/session |
|---|---:|---:|---:|---:|---:|
| Strong model throughout | 100% | 66.7% | 100% | 80.8% | 1,793 ms |
| Small model throughout | 50.0% | 53.6% | 0% | 80.8% | 456 ms |
| Protection, gate disabled | 100% | 66.1% | 83.3% | 66.1% | 1,771 ms |
| Candidate gate, observe | 100% | 66.7% | 83.3% | 65.9% | 1,694 ms |
| Candidate gate, enforce | 90.6% | 69.3% | 51.6% | 64.8% | 1,221 ms |

Each arm contains **32 sessions and 192 calls**. **Final-turn pass rate is not whole-task accuracy:**
it checks ordered final synthesis in state retention, but only the closing
acknowledgement in the retrieval workload. An acknowledgement can pass after a
failed retrieval, which is why intermediate turn accuracy matters too.

Compared with the gate-disabled control, enforcement used fewer strong-model
turns and finished faster, but final checks fell from **32/32 to 29/32**. The
paired difference was -9.4 percentage points, with a bootstrap 95% interval of
[-21.9, 0.0]; the apparent turn-accuracy improvement was also inconclusive. The
candidate did not meet our quality requirement and was rejected for enforcement.

These are controlled switching measurements, **not semantic-classifier accuracy
or production-scale throughput**. Signals and proposals were controlled, calls
were sequential, and outputs were short. Cache ratios were backend counter
deltas averaged per session; strong-model share is a usage proxy, not measured
dollar savings. A separate [llm-d simulator test](https://github.com/vllm-project/semantic-router/pull/4443)
checks model/pool/endpoint composition, not real-GPU scheduling performance.

## When should the conversation change models?

All three failed final checks came from state-retention sessions with the same
schedule: three small-model turns, then three strong-model turns. The small
model failed turn three; the stronger model received that answer in its history
and continued. The final synthesis contained incorrectly ordered values.

<figure style={{width: '92%', maxWidth: '100%', marginInline: 'auto', textAlign: 'center'}}>
  <a href="/img/blog/multiturn-routing/conversation-handoff.svg" target="_blank" rel="noopener noreferrer" aria-label="Open the conversation handoff diagram at full size">
    <img
      src="/img/blog/multiturn-routing/conversation-handoff.svg"
      alt="The small model fails turn three. The strong model continues turns four through six with the earlier error in its history. A proposed recovery path verifies or repairs that step before continuing."
      width={1000}
      height={640}
      loading="lazy"
      decoding="async"
      style={{width: '100%', height: 'auto', maxHeight: 'none'}}
    />
  </a>
  <figcaption style={{fontSize: '0.875rem', lineHeight: 1.65, marginTop: '0.75rem'}}>
    Figure 2: A switch changes the executor, not earlier answers.
    Repair is a proposed strategy, not a measured result.
  </figcaption>
</figure>

The traces are consistent with late escalation carrying damaged state forward;
they do not isolate that mechanism causally. Testing verification, state repair,
or safe replay is the next step. Side-effecting tools need idempotency safeguards.

Timing also mattered in a separate native-tool benchmark. Across eight paired
seeds, choosing the stronger model **before a known tool boundary** improved
final success from **37.5% to 100%**. The tradeoff was 52.5 percentage points more
strong-model turns, 11.35 points lower prefix-cache hit ratio, and 1,275 ms more
per session. The fixture supplied the boundary explicitly; predicting it in
real applications remains work to do.

Together, these experiments suggest two distinct policies: select an appropriate
owner before difficult work begins, and recover failed state rather than merely
changing the model on the following turn.

## What to measure in your own stack

Compare fixed strong and small models, per-turn routing, and continuity-protected
routing on paired tasks with an untouched holdout. Then test endpoint scheduling
separately, under realistic load. At minimum, check:

- **Task correctness:** final outcomes and intermediate errors, not just successful requests.
- **Continuity:** tool call/result sequences and stored-history portability.
- **Serving behavior:** eligible endpoint selection, warm/cold prefixes, time to
  first token, queue time, and latency percentiles.
- **Failure recovery:** owner loss, retries, and protection against duplicate tool side effects.

Run new gates in observe mode first. Set quality requirements before tuning for
speed or model usage. The architectural payoff is a division of responsibility:
Semantic Router decides **what should do the work**, the inference router decides
**where it should run**, and task-level evaluation checks whether the combination
actually finishes the job.

### Start with a reproducible replay

From the repository root, with Python 3.10+ and jq, inspect the bundled
[coding-agent session](https://github.com/vllm-project/semantic-router/blob/62bb0b94d2ff567ff49c2a236b7bd5270f37e897/bench/agent_session_replay.py)
without sending a request:

```bash
python3 bench/agent_session_replay.py --dry-run | \
  jq '{fixture, dry_run, tools, summary}'
```

It plans three requests with ten tools. To replay them through your own gateway:

```bash
REPLAY_ID="blog-replay-$(date +%s)"
python3 bench/agent_session_replay.py \
  --base-url http://127.0.0.1:8899/v1 \
  --session-id "$REPLAY_ID" \
  --extra-header "x-conversation-id=$REPLAY_ID"
```

Match the identity headers to your protection configuration, use a fresh ID for
each comparison, and add `--api-key-env` when authentication is required.

<details>
<summary>What this replay measures, and what it does not</summary>

The fixture supplies the history: generated answers are not fed into later
requests, and tools are not executed. It measures request shape and routing
diagnostics, **not completed-agent-task quality**.

The current CLI counts HTTP 2xx as success, even for a malformed response body,
and can exit zero while reporting a tool-loop switch violation. Check the
report, including model identity on every turn. Dry-run zero switch counts
are not continuity evidence. Missing cached-token detail is recorded as zero;
use backend telemetry to distinguish missing data from no cache reuse.

</details>

For tasks where generated answers drive subsequent actions, use
[sr-bench's frozen-plan workflow](/docs/benchmarking/sr-bench/plan-and-run) and
[task reports](/docs/benchmarking/sr-bench/results). Default agent runs do not
supply session identity; [per-task identity is tracked separately](https://github.com/vllm-project/semantic-router/issues/4256).
Do not treat those defaults as tests of session-scoped protection.

### References and reproducibility

- [Continuity protection](/docs/tutorials/learning/protection),
  [memory and Replay](/docs/tutorials/learning/memory-and-replay), and
  [recent-outcome gates](https://github.com/vllm-project/semantic-router/pull/3436).
- [Tool-loop ownership fix](https://github.com/vllm-project/semantic-router/pull/4297),
  [Chat/Responses benchmark runner](https://github.com/vllm-project/semantic-router/pull/4082),
  [coding-agent replay](https://github.com/vllm-project/semantic-router/pull/4154), and
  [per-task continuity reporting](https://github.com/vllm-project/semantic-router/pull/4239).
- [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
  and [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/).
- [Evaluation record and holdout results](https://github.com/vllm-project/semantic-router/issues/4080#issuecomment-5938865660).
  Router revision: [`0ee9955fc6ae0d79104e33bc933ae0f75ed52acf`](https://github.com/vllm-project/semantic-router/commit/0ee9955fc6ae0d79104e33bc933ae0f75ed52acf);
  Qwen3-0.6B: [`c1899de289a04d12100db370d81485cdf75e47ca`](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca);
  Qwen3-8B: [`b968826d9c46dd6066d109eabc6255188de91218`](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218).
