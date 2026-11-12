---
title: Performance and cost
sidebar_label: Performance and cost
sidebar_position: 2
description: What System One costs a request, and the batching, caps, budgets and placement choices that bound it.
---

# Performance and cost

A System One question costs a forward pass of a small model, not a generation
by a chat model: there is no output to decode and no text to parse. What a
request costs is the number of calls it makes, the size of what each reads, and
where the model runs.

:::tip
Most requests make **one** call: the request stage's. A rule that nothing reads
is not asked, and a stage with no question makes no call.
:::

## Calls per request

| Stage | Calls | Asked when |
| --- | --- | --- |
| Request signals | One per deployment, with every question about every state in it | A decision, projection or request plugin reads a rule |
| `systemone` selector | One | The matched decision uses the selector |
| Judges (`same_request`, `relevance`, `tool_need` + `tool_rank`) | One per feature, with one entry per judged candidate | The feature runs and has candidates |
| `masking` | One, for the text the request stage did not already answer | The decision masks |
| Response signals | One per deployment | A `response_guard` rule reads them |
| `escalate` checks | One per judged answer | `accept.systemone` or `accept.refusal` |

## Batching

- **Questions share a call.** Every question a stage asks one deployment
  travels together, whatever signal or rule it comes from. The model runtime
  batches calls from concurrent requests too
  ([engine arguments](../../model-runtime/deploy/engine-args.md#batching)).
- **Questions share an answer.** Rules of one preset on one deployment about
  one state share one question.
- **State-once models** (d3-turbo, d3-ultra) read a state once and answer
  every question from that reading, so the tenth question about a request costs
  little more than the first. Decision 2.0 computes the shared context of a
  multi-question request once (`enable_shared_context`).
- **Answers are reused.** The Router keeps the last 4,096 answers per local
  model (`VLLM_SRUN_RESULT_CACHE` changes the number; 0 turns it off), and a
  remote API's answers for up to 10 minutes, until the configuration changes.

## Caps

- **Judges** read at most `judge.max_candidates` candidates: 64 by default and
  at most 4096 for a per-candidate judge, at most 255 (the default) for a
  choice judge. Overflow
  is truncated by recall rank, or in listed order without recall, never failed;
  `vsr_match_judge_truncated_total` counts it.
- **Recall** bounds what a judge sees: a cache judge confirms its top 3
  recalled hits, a RAG judge its top 20 documents.
- **Tools** above 255 after the policy are truncated, or narrowed first by
  `recall: {method: bm25}`.

## Budgets and deadlines

| Setting | Bounds |
| --- | --- |
| A rule's or judge's `timeout_ms` | Its answer, up to 60,000 ms; else the deployment's, else 1,000 ms |
| A deployment's `timeout_ms` | One attempt of a call to an attached or remote deployment |
| `global.model_catalog.signal_timeout_ms` | All of a request's signals; 0 (the default) derives it from the request's own deadline |
| An executor's `budget` | `escalate`'s deadline and calls, its `answer_check` calls included in the deadline |

:::note
A late answer leaves its questions unknown, and the failure policy of each rule
or feature decides. A deadline ends the Router's wait, but does not stop a
forward pass already running.
:::

## When the decision model is overloaded

A decision model refuses a call it has no room for: its request queue is full,
and it answers `overloaded`. A rule that fails closed then matches. A
request-stage `jailbreak` or `pii` rule does so unless its `on_unknown` is
`no_match`. A decision with a `fast_response` then refuses the request as if it
were unsafe. Under load this refuses most requests, and coding agents take the
refusal as final.

`global.model_catalog.on_signal_overload` makes such a request get an error the
client retries. Without it, nothing changes:

```yaml
global:
  model_catalog:
    on_signal_overload:
      action: retry          # refuse (the default) or retry
      wait_ms: 2000          # 0 to 60,000; 0 does not wait
      retry_after_seconds: 0 # up to 60; 0 derives it from wait_ms
```

With `action: retry`:

- **Wait for capacity.** Every request-stage call that comes back
  `overloaded` is asked again while the request waits for capacity, whatever
  its rules' failure policy. That covers a rule that fails closed, one that
  fails open such as `safety`, and a routing question. If the model answers,
  the request is read as it is without the overload.
  - The wait lasts up to `wait_ms` from the request's first overloaded answer,
    within the signals' deadline. A rule's own `timeout_ms` bounds each attempt
    of its call, not the wait.
  - A request's calls to one model ask again together, in rounds, so the
    questions one call carried travel in one call again.
- **Bound the load.** A request's calls to a model ask again in at most 3
  rounds: at a quarter of its wait, at half and at its end.
  - While the model refuses calls as overloaded, one waiting request's round
    asks it per step, however many requests wait. The step is jittered, from
    100 ms doubling to 1 s.
  - In such a round, a request with more than 8 calls waiting for the model,
    such as a long input's windows, sends only one of them, on its own, as a
    probe. The probe does not count as one of the 3 rounds.
  - The first answer to any call opens the model again, and every request
    waiting for it has its round then.
- **Learn no pace from the wait.** A guard scan that waited learns no pace for
  its model, and does not judge from its time that it fell behind: its time
  says nothing of how fast the model reads.
- **Answer a retryable error.** If the model still does not answer, the Router
  answers an error the client retries in two cases:
  - Only rules that failed closed on the overloaded model refuse the request.
    The error replaces the refusal.
  - A rule left unread because the model was overloaded would refuse the
    request had it matched. The error replaces the forward. A rule that fails
    open, such as `safety`, no longer lets the request through when the model
    was too busy to read it.

  An Anthropic Messages client gets `529 overloaded_error`. An OpenAI Chat
  Completions or Responses client gets `429` with the code `overloaded`. Its
  SDK and agents retry a 429 after its `Retry-After`, where some treat a 5xx as
  a failure. Both errors reach the client before any byte of a stream, and
  carry:
  - `Retry-After`: `retry_after_seconds`, else `wait_ms` rounded up to whole
    seconds, at least 1. It is at most 60: some clients ignore a longer value,
    and others sleep it in full.
  - `x-vsr-warnings: guard_overloaded`.
- **Never forward unchecked.** The request is never forwarded while a rule
  that could refuse it is unread for overload.
  - A rule that matches still refuses it, and so does a decision whose
    condition matches on error.
  - Other failures, such as a timeout or a model that is not ready, keep
    their outcome.
  - A prompt beyond every context window still gets
    `400 context_length_exceeded`.
  - A routing question left unread lets the request go on: it can choose a
    route, but not refuse the request.

These are counted apart from refusals and from the `signal_failed_closed` log:

- `vsr_signal_overload_waits_total{deployment, outcome}` counts the calls asked
  again, by how the last one ended: `ok` when capacity freed, `overloaded`
  when the wait ran out, or the reason of another failure.
- `vsr_signal_overload_retryable_total{format}` counts the requests answered
  the retryable error, by the client's wire format.

## Long inputs

A long user turn, such as a pasted document, a log or a long tool output, is
read in one of two ways, by the Router itself, so a local and a remote decision
model behave the same.

**Routing questions read a bounded view.** These are the rules that do not
read their whole input: `domains`, `complexity`, `modality`, `fact_check`,
`turn_relation`, `user_feedbacks` and `systemone` rules with
`coverage: truncate`.

- A turn longer than the deployment's `routing_view` keeps its head and its
  tail around a marker, `[... 41000 tokens omitted ...]`.
  The head takes a quarter of the view and the tail the rest, because the
  request's instruction usually closes the turn.
- A turn within the view is sent byte for byte as it came.

**Guard rules read the whole input, in windows.** These are `jailbreak`,
`safety` and `pii`, at the request and at the response stage.

- An input longer than one `scan_window` is split into windows that overlap by
  `scan_window.overlap`, so text across a cut is read whole by one of them.
- The windows are read tail first, then the head, then the middle by halves:
  the window at its middle, then those at its quarters, its eighths and so on,
  from a starting point that changes with every request. The windows read by
  any moment are spread over the whole input, so no part of it is predictably
  left unread. Each window is a call of its own, at most
  `scan_window.max_parallel` at a time per deployment. The `jailbreak`,
  `safety` and `pii` rules that read the same input share each window's call,
  and a window's answers count as soon as it is read.
- Any window that finds an attack, a harm or personal data flags the input.
  `pii` spans keep their places in the input, so masking replaces every one.
- A window that makes a rule match decides it: no unread window can undo a
  match. When that match alone refuses the request, because a decision with a
  `fast_response` plugin matches on the rule by itself and no other decision
  ranks above it, the window ends the request's guard scans: the windows not
  read yet are not sent, and the request is refused at once instead of at the
  deadline.
- One input reads at most 256 windows. An input that needs more of them is
  read in larger windows between its head and its tail, up to
  `scan_window.max_tokens` (by default the largest the model takes: 31,744
  bytes for a 32,768-token validated length), so 256 cover it whole. Only an
  input longer than 256 of the largest windows, about 7.5 MB, leaves windows
  unread, and they count as unread.
- A larger window dilutes what it holds: the same injection scores lower in a
  window of 8,000 tokens than in one of 2,000. The head and the tail, the first
  and the last `scan_window.tokens` of the input, are always read at that
  size. A larger window whose answer leaves a rule short of its threshold, by
  at most `scan_window.reread_margin` in log-odds, is read again in windows of
  `scan_window.tokens` before the rule decides. Dilution lowers a score's
  log-odds by about the same amount at any threshold, so the default of 3
  re-reads from about 0.91 for a rule at 0.995, 0.27 at 0.88 and 0.05 at 0.5,
  and never once the window matched. A window within half the margin, where a
  diluted match lands, is read again at once, ahead of every window not yet
  sent, and can still end the scan early; one further below waits until the
  windows still unread are read, closest to its threshold first, so a scan
  that fell behind reads its runs before such re-reads. A part that the
  input's windows already read and answered is not read again.
- An input whose windows are all read by the signals' deadline is read at the
  configured size, as fast as its windows are read. A scan that falls behind,
  whose windows left would not all be read in time at the pace it read the
  others, reads them in runs of windows read as one when the deployment's
  measured pace says those would be read in time, the shortest runs that are;
  otherwise it keeps its windows and reads what it can. A decision model reads
  each token at about the same cost in a window of any size, so larger windows
  save little time and seldom fit.

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        routing_view: {max_tokens: 8192, head_share: 0.25}
        scan_window: {tokens: 8192, overlap: 512, max_parallel: 4}
      hosted-decision:
        provider: systemone
        endpoint: https://decisions.example.com/v1/systemone
        model: example-decision-model
        timeout_ms: 6000
        routing_view: {max_tokens: 4096}
        scan_window: {tokens: 4096, overlap: 256}
```

| Setting | Default | Bounds |
| --- | --- | --- |
| `routing_view.max_tokens` | 8,192; 4,096 for `provider: systemone` | 256 to 1,048,576 |
| `routing_view.head_share` | 0.25 | 0 to 1 |
| `scan_window.tokens` | 8,192; 4,096 for `provider: systemone` | 256 to 1,048,576 |
| `scan_window.overlap` | 512; 256 for `provider: systemone` | 0 to less than half a window |
| `scan_window.max_parallel` | 4 | 1 to 64 |
| `scan_window.max_tokens` | the model's validated length less 1,024 | `scan_window.tokens` to 1,048,576 |
| `scan_window.reread_margin` | 3 (log-odds below the rule's threshold) | 0 to 20 (0 reads no window again) |

**Views and windows are sized in bytes.** A token of `max_tokens` or
`scan_window.tokens` counts as four bytes, about what a token of prose takes,
so the 8,192 default is a 32 KB view. The model's validated length (its card's
`validated_input_tokens`, else `max_input_tokens`) less 1,024 tokens for the
question caps both at one byte a token, because a byte-level tokenizer gives no
byte more than one token. Dense text such as JSON or base64 takes far fewer
bytes a token than prose, so a view of it holds up to about three times as many
tokens, but no setting makes a view or a window overrun the model: with a
32,768-token validated length, neither ever holds more than 31,744 bytes.

**Size them from the deadline.** A call should end well before its deadline,
which is the rule's or the deployment's `timeout_ms` for a remote API, and the
signals' deadline for a local model (45 s for a served request by default,
`global.model_catalog.signal_timeout_ms`). If an API reads R tokens a second and
takes T seconds to answer the smallest call, a view of up to
`(timeout - T) × R` tokens answers in time: an API that reads 2,000 tokens a
second and takes 2 s for the smallest call reads about 8,000 tokens in a 6 s
timeout, a `max_tokens` of 8,000 for prose, or about 2,700 if turns are often
JSON or base64. A scan's windows share the deadline: the deployment reads
`scan_window.max_parallel` of them at a time, the tail and the head first, so
the windows read when the deadline comes are the ones a partial scan decides
on. A window not read in time counts as unread, and with `best_effort` the rule
decides on the windows read. A scan that falls behind reads the rest in larger
windows only when the deployment's measured pace says they would be read in
time; the cost a call adds besides its text is all they save, so on one
decision model an input of about a million tokens may be read only in part,
and `scan_partial` says so. Lower `scan_window.tokens` for a slow API, and
raise the deadline for inputs you expect to scan in full; more windows in
flight help only while the model has room to batch them.

**Coverage and failures.** When the deadline ends a scan before every window is
read, and no window read decides the rule, `on_unscanned` decides:
`best_effort` (the default) decides on the windows read, `match` matches
(strict), `no_match` does not, and `unknown` leaves the rule to `on_unknown`. A
rule's `min_coverage`, a share from 0 to 1 that is off by default, sets how
much `best_effort` must have read: below it the rule reads as `unknown`, so a
`jailbreak` or `pii` rule matches by default and the guard refuses what it read
too little of. When no window is read at all (the model is down, or no window
answered in time), a `jailbreak` or `pii` rule follows `on_unknown`, which
matches by default (fail closed), and a `safety` rule is unknown, which its
decision's `rules.on_unknown` decides.

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_attack
          min_coverage: 0.5
```

Coverage is reported for every guard rule that read its input in windows:

- `x-vsr-warnings` carries `scan_partial`, for every caller, on a request that
  was routed while a request-stage guard rule had read only part of its input;
- `vsr_signal_scan_coverage_total{signal, result}` counts each guard rule's read
  as `full`, `partial` or `none`, `vsr_signal_scan_coverage_ratio{signal}` gives
  the share of its windows it read, and `vsr_signal_scan_decided_total{signal}`
  counts the rules a window decided before the rest was read;
- the routing preview (`scan_coverage`) and the Router Replay record
  (`scan_coverage`) give `full`, `partial n/m` or `none` per rule;
- `x-vsr-trace` carries it as `scan_coverage` for a caller allowed the debug
  header.

| Model | Validated length per question | Maximum |
| --- | --- | --- |
| vela3-fast | 32,768 tokens; its own 8,192-token windows for router and span questions | 32,768 |
| vela3-lite, -mini, -flash | 262,144 tokens; 32,768 per span target | 262,144 |
| vela3-turbo, d3-turbo | 32,768 tokens | 262,144 with long context |
| d3-ultra | 131,072 tokens | 262,144 with long context |

:::warning
A whole-input scan of a long request on a CPU with few cores can take longer
than the signal deadline. Measure your input lengths and concurrency, then give
the deployment dedicated CPUs or a GPU.
:::

## Placement

- **vela3-fast on a CPU** is the default. The loaded deployment holds about
  7 GB; give the Router 12 GiB.
- **A GPU** runs the larger Vela 3.0 and Decision sizes, and serves many more
  requests per second at the same latency.
- **A remote API** adds a network round trip to each call, and its models have
  no calibrated thresholds.
- **One deployment for everything** loads one model. Spreading types over
  several deployments makes one call per deployment per stage.

[Choose a model, size and hardware](../../model-runtime/get-started/choose-a-model.md)
has the measured latency per size, and the
[model runtime records](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)
the full measurements.
