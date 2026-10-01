---
slug: multiturn-model-switching
title: "When Not to Switch Models Mid-Conversation"
description: "What more than 1,000 routed calls taught us about continuity, reactive model switching, tool loops, KV cache, and llm-d endpoint selection."
authors: [anupsharma]
tags: [evaluation, agentic, mixture-of-models, routing, vllm, llm-d]
image: /img/vllm-sr-logo.social.png
---

Semantic routing sounds simple when every request is independent: classify the
prompt, choose the best model, and send the request. A real agent conversation
is not independent. It accumulates assistant answers, tool calls, provider
state, and KV cache. Changing models halfway through a session can improve
quality, but it can also discard useful cache state, change tool behavior, or
hand a stronger model a conversation that has already gone wrong.

That led us to a more useful question than “can Semantic Router switch models?”

> When is switching valuable enough to justify its continuity cost—and when is
> the switch already too late?

We built a reproducible evaluation around the session-aware routing and
progress-gate machinery in vLLM Semantic Router. The work included deterministic
contract coverage, live Router + Envoy + vLLM runs, native tool calls, an
untouched holdout, and a separate llm-d integration contract. The most important
result was negative: our first fitted reactive gate saved strong-model traffic
and latency, but it was not safe to enforce.

<!-- truncate -->

## First, separate model routing from endpoint routing

One question came up repeatedly: what happens when Semantic Router chooses one
model but an llm-d endpoint picker prefers another pod?

There should be no conflict because these components own different decisions:

```text
request
  -> Semantic Router chooses the logical model pool
  -> HTTPRoute selects that InferencePool
  -> llm-d chooses a ready replica inside the pool
```

Semantic Router should not choose a pod. llm-d should not widen the set of
models allowed by semantic or business policy. We proposed an executable E2E
contract in [PR #4443](https://github.com/vllm-project/semantic-router/pull/4443)
that sends traffic through both layers and verifies that the reported logical
model and simulator-selected pod belong to the same two-replica pool.

That contract proves the control-plane boundary. It does not pretend that a
simulator proves real-GPU cache savings, prefix affinity, or load-balancing
quality.

## The first live experiment found a real bug

Our native-tool workload compared two policies:

- **Reactive:** start cheaply and escalate after observing a failure.
- **Prospective:** move to the stronger model at the known tool boundary, then
  preserve that model through the tool loop.

Across eight paired seeds, prospective ownership improved native tool-call
success from 62.5% to 100%, tool-continuation success from 0% to 75%, and final
success from 37.5% to 100%.

The gain was not free. Prospective routing used 52.5 percentage points more
strong-model turns, reduced the measured cache-hit ratio by 11.35 points, and
added 1,275 ms per session.

The experiment also exposed a product defect: a strong model selected on a
bypass path was not preserved as the tool-loop owner. The fix landed in
[PR #4297](https://github.com/vllm-project/semantic-router/pull/4297). That is
exactly why evaluations should exercise the real request path instead of only
simulating policy decisions.

## Fitting a reactive gate

Next, we tested recent-outcome gates that suppress a proposed escalation until
there is enough evidence that the current model is regressing. We compared:

- the strongest model for the entire session;
- the cheapest model for the entire session;
- continuity protection without a progress gate; and
- several progress-gate thresholds.

The development fit selected an aggressive candidate that needed one observed
regression before allowing escalation. It preserved 100% final success on the
development aggregate while using fewer strong-model turns than ungated
routing.

But the development data also contained a warning: on a one-off challenge, the
candidate escalated after a single failure and kept paying for the strong model
after the task returned to easy turns. We froze the candidate anyway—not as a
recommendation, but as a hypothesis to test on untouched data.

## The consolidated holdout

The final holdout used:

- Semantic Router at immutable commit `0ee9955`;
- Envoy in the request path;
- Qwen3-0.6B and Qwen3-8B served by vLLM;
- one NVIDIA L40S;
- 16 untouched seeds;
- two six-turn workloads; and
- five predeclared policies.

In total, the run executed 160 sessions and 960 routed calls. No threshold was
changed after accessing holdout outcomes. The GPU portion took 1,167 seconds
and cost approximately $0.63 on the selected pay-as-you-go runner.

| Policy | Final success | Turn accuracy | Strong turns | Cache hit | Latency/session |
|---|---:|---:|---:|---:|---:|
| Strongest sticky | 100% | 66.7% | 100% | 80.8% | 1,793 ms |
| Cheapest sticky | 50.0% | 53.6% | 0% | 80.8% | 456 ms |
| Protection, no gate | 100% | 66.1% | 83.3% | 66.1% | 1,771 ms |
| Candidate, observe | 100% | 66.7% | 83.3% | 65.9% | 1,694 ms |
| Candidate, enforce | 90.6% | 69.3% | 51.6% | 64.8% | 1,221 ms |

Enforcement did what it was designed to do economically. Against protection
without a gate, it used 31.8 percentage points fewer strong-model turns and was
550 ms faster per session.

It also lost three final tasks.

The final-success delta was -9.4 points, with a paired bootstrap 95% interval
from -21.9 to 0.0 points. The turn-accuracy change was inconclusive. That is not
enough evidence to authorize enforcement.

## Why “switch after failure” can fail

All three losses followed the same schedule:

```text
small -> small -> small -> strong -> strong -> strong
```

The gate suppressed the first two escalation proposals because it had not yet
observed enough negative evidence. The small model then failed the third turn.
Only after that failure did the switch occur.

The strong model received the full conversation, including the incorrect prior
answer. In three of sixteen sustained-regression sessions, it returned the
stored values in the wrong order at final synthesis. The strongest-sticky,
observe, and no-gate policies all completed every final task.

The stronger model was not incapable. The timing was wrong.

This is the key lesson: a reactive gate observes evidence only after the
conversation has paid for the failure. Switching models does not automatically
repair the transcript or restore the state that should have existed before the
failure.

## What we would deploy today

We would keep the progress gate in observe mode and preserve hard continuity
rules for tool loops and provider-bound state. We would use prospective signals
at boundaries where risk is knowable before dispatch—for example, entering a
native tool loop—rather than treating every past failure as a generic reason to
switch.

The next candidate needs at least one of these:

1. **Prospective boundary risk:** predict that the next turn requires a
   capability the current model is unlikely to satisfy.
2. **Repair or replay:** when escalation follows a failure, replay or repair the
   failed step before allowing the new model to continue.
3. **Task-state-aware outcomes:** distinguish a harmless formatting miss from a
   failure that corrupts future conversation state.

It also needs broader agent workloads before any universal production claim.
Our exact-answer tasks are useful controlled probes, not a substitute for a
representative application evaluation.

## The honest conclusion

Semantic Router already has machinery to preserve session ownership, protect
tool loops, record Replay evidence, and compose with an endpoint picker. The
difficult part is not making a switch. It is knowing whether a switch will
improve the rest of the task after accounting for everything the conversation
has already accumulated.

Our first candidate did not pass that bar, so we did not enable it. That is a
successful evaluation outcome: we fixed one real continuity bug, proposed
missing contracts, quantified the quality/cost tradeoff, and learned when not
to switch.

The ongoing work and reproducible benchmark coverage are tracked in
[issue #4080](https://github.com/vllm-project/semantic-router/issues/4080) and
[PR #4130](https://github.com/vllm-project/semantic-router/pull/4130).
