---
title: How the Router uses System One
sidebar_label: How the Router uses it
description: The System One call behind each routing feature, how the Router merges calls, how judges cap their candidates, and when recall needs an embedding model.
---

# How the Router uses System One

Every judgment the Router makes is a System One question: the signals that
route a request, the selector that picks its model, the guard that reads its
answer, and the judges that decide what is similar. They all go through one
client, in one of the two forms of [the API](./api):

- a **standard request**, one state and its questions;
- a **batch**, named standard requests in one call, when the questions are
  about different states.

The examples below show the call each feature makes; `{...}` elides question
details. On the wire a question carries the ID its model was trained on when
it has one (vela3-lite's `domain`), and otherwise a Router-owned ID such as
`preset:complexity` or `systemone:needs_tools`. The Router names batch
entries `"1"`, `"2"`, … in a fixed order; the entry names in the examples are
illustrative.

## Routing signals

Before routing, the request stage asks every rule that decisions,
projections or request plugins read. Questions about the same state go to
their deployment as one standard request, however many there are:

```text
{"state": {"request": "Prove that the square root of 2 is irrational."},
 "questions": {"domain": {...}, "jailbreak": {...}, "complexity": {...}, "needs_tools": {...}}}
```

```yaml
recipes:
  - name: default
    signals:
      domains:
        - name: math
          labels: [math]
      jailbreak:
        - name: attack
      complexity:
        - name: difficulty
      systemone:
        - name: needs_tools
          question:
            type: noul
            instructions: Does answering this request need a tool call?
          predicate: {gte: 0.7}
    decisions:
      - name: hard-math
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: domain, name: math}
            - {type: complexity, name: difficulty, label: hard}
        modelRefs:
          - model: reasoner
```

- A rule that reads more than the current turn (`input.history`,
  `input.previous_turns`, `input.tool_results`, images) reads another state.
  When the rules read different states, the call is a batch with an entry
  per state, and no state is sent at the top level.
- Rules of one preset on one deployment about one state share one question;
  each applies its own `labels` and `predicate` to the answer.
- A rule that no decision, projection or plugin reads is not asked.
- A request on the Images API (`POST /v1/images/generations`) is asked about
  its prompt like any text request. It must reach a decision whose
  [output](../tutorials/decision/output) is an image, or it is refused with
  `image_route_required` before any model is called.
- A late or failed answer leaves the rule unknown. A decision's
  `rules.on_unknown`, or a condition's `on_error`, says what that means; a
  `jailbreak` or `pii` rule with `on_unknown: match` matches when its model
  gives no answer.

## The `systemone` selector

After a decision matches, its `systemone` selector asks which of its
`modelRefs` should answer: one `choice` whose options are the candidates and
their descriptions, in its own call. Each candidate's probability is its
selection score:

```text
{"state": {"request": "Refactor this module to remove the global cache."},
 "questions": {"selector": {"type": "choice",
   "instructions": "Which model should answer this request?",
   "choices": [{"key": "qwen3-8b", "description": "Fast general model for routine code"},
               {"key": "qwen3-32b", "description": "Strong reasoning model for hard code"}]}}}
```

When the model is not ready, answers late or answers invalidly, the first
`modelRef` answers. See [System One selection](../tutorials/algorithm/selection/systemone).

## `response_guard` and `masking`

Response-stage rules ask about the model's answer once it exists. The
[`response_guard`](../tutorials/plugin/response-guard) plugin asks only the
rules its guard rules read, and an `AND` stops at its first false condition:

```text
{"state": {"request": "How do I pick a lock?", "answer": "First, insert a tension wrench ..."},
 "questions": {"preset:jailbreak": {...}, "preset:harm": {...}}}
```

A `hallucination` rule reads the answer and its grounding context, the
retrieved documents and tool results, and needs a span-capable deployment:

```text
{"state": {"request": "When does the library open?", "answer": "It opens at 8:00 on Sundays.",
           "context": "The library is open Monday through Friday, 09:00 to 17:00."},
 "questions": {"preset:hallucination": {"type": "span", ...}}}
```

[`masking`](../tutorials/plugin/masking) asks the `pii` rule it names about
every text position of the final provider-bound request: system
instructions, messages, tool-call arguments and tool results, including what
the context plugin injected. Text the request stage already answered is
reused by its hash, and a span it cannot get fails the request closed.

## Judges: `same_request`, `relevance`, `tool_need` and `tool_rank`

A [`match`](./reference#the-match-block) block has two optional stages.
**Recall** narrows a candidate set by embedding or BM25 similarity. **Judge**
asks a System One question about the candidates. A per-candidate judge asks
about each (request, candidate) pair, one batch entry per candidate; a
choice judge offers the candidates as the options of one question.

**`same_request`** confirms each cached request the response cache recalled,
so a near-miss paraphrase with a different meaning is not served:

```text
{"states": {
  "hit_1": {"state": {"request": "How do I reset my password?",
                      "candidate": "I forgot my password. How do I change it?"},
            "questions": {"same_request": {"type": "noul", "instructions":
              "Does the candidate request ask for the same thing as the request, so that one answer serves both?"}}},
  "hit_2": {"state": {"request": "How do I reset my password?", "candidate": "How do I reset my router?"},
            "questions": {"same_request": {...}}}}}
```

**`relevance`** judges each document a RAG source recalled, and each segment
context compression considers, with the latest user message as the request:

```text
{"states": {
  "doc_1": {"state": {"request": "How long is the refund window?",
                      "candidate": "Refunds are accepted within 30 days of purchase."},
            "questions": {"relevance": {"type": "noul", "instructions":
              "Does the candidate passage contain information that helps answer the request?"}}},
  "doc_2": {"state": {"request": "How long is the refund window?", "candidate": "Our offices close on public holidays."},
            "questions": {"relevance": {...}}}}}
```

**`tool_need` and `tool_rank`** select tools in one standard request:
`tool_need` is a yes/no gate, and `tool_rank` is a choice whose options are
the candidate tools the tools policy allows:

```text
{"state": {"request": "Open a ticket: the export button fails."},
 "questions": {"tool_need": {"type": "noul", ...},
               "tool_rank": {"type": "choice", "choices": ["docs.search", "tickets.lookup", "tickets.create"]}}}
```

| Feature | Default `match` |
| --- | --- |
| `response_cache.semantic.match` | `recall: {method: embedding, top_k: 3, min_similarity: 0.8}` + `judge: {preset: same_request, min_score: 0.8}` |
| `context.sources[rag].match` | `recall: {method: embedding, top_k: 20, min_similarity: 0.3}` + `judge: {preset: relevance, top_k: 6, min_score: 0.5}` |
| `context.sources[memory].match` | `recall: {method: embedding, top_k: 5, min_similarity: 0.7}` |
| `context.compress.match` | `recall: {method: bm25}` + `judge: {preset: relevance}`, per segment |
| `tools.select.match` | `judge: {preset: tool_rank, top_k: 3, min_ratio: 0.1}`, gated by `tool_need` |

A feature that writes its own `match` block replaces the default as a whole;
the [plugin reference](../tutorials/plugin/overview#match-block) lists which
recall methods each feature allows.

## `escalate`

On Chat, [`escalate`](../tutorials/algorithm/looper/escalate) calls its
candidates in order and stops at the first accepted answer. With
`accept.systemone`, the decision model's `answer_check` question judges each
answer, with the request and the answer as the state:

```text
{"state": {"request": "What is 17 × 23?", "answer": "391"},
 "questions": {"answer_check": {"type": "noul", ...}}}
```

`accept.refusal: false` asks the `refusal` question about the answer instead.
On a System One entrypoint, `escalate.stages` asks the original typed request
of each decision model in turn, and native acceptance rules read the typed
answers; nothing is converted to Chat text.

## How calls are merged

The Router merges the calls of each stage:

- **Per deployment.** Every question a stage asks one deployment goes in one
  call, to local and remote deployments alike. The Router never splits a call
  to fit a server's per-call limits.
- **Per state.** Questions about the same state share one entry; questions
  about different states are separate entries. One distinct state is sent as a
  standard request, two or more as a batch.
- **Per stage.** The request stage, the selector, the response stage and the
  judges of each feature run when their inputs exist, so a request with all of
  them makes several calls. A deployment shared by all of them still loads
  one model.

A local runtime answers a failed question in place. A remote System One API
fails the whole call and names the question and, in a batch, its entry
(`state`); the Router then asks once more, within the same deadline, only the
states that still have other questions, in the same two forms, so only the
failed question's rule about that entry is unknown. An API that names no
entry leaves the question unknown about every state that asked it.

A per-candidate judge sends at most `judge.max_candidates` entries. Answers to
the same questions about the same state are reused: from a local runtime's
result cache, and for up to 10 minutes from a remote API, until the
configuration changes.

## Caps and truncation

Every judge has a hard cap on the candidates it reads, `judge.max_candidates`:

- a choice judge (`tool_rank`) offers at most 255 candidates, the protocol's
  option limit, which is also its default;
- a per-candidate judge (`relevance`, `same_request`) judges 64 by default.

A value below 1, or above 255 on a choice judge, fails to load. Overflow never
fails a request: the judge reads the first `max_candidates` candidates by
recall rank or, without a recall stage, in the order the feature lists them,
most recent first. Each truncation adds a `match.judge.truncated` event to the
request's trace span and counts in `vsr_match_judge_truncated_total`. A tool
catalog larger than 255 after the policy is truncated in the order it is
listed, or narrowed first by `recall: {method: bm25}`.

## Recall or judge, and when you need an embedding model

| A `match` with | Gives | Suits |
| --- | --- | --- |
| `recall` only | Classic embedding or BM25 similarity | Large stores where similarity is enough: memory, broad RAG |
| `judge` only | A System One judgment of every candidate | Small, naturally bounded sets: a few tools, a small knowledge base, a user's handful of memories |
| `recall` + `judge` | Recall narrows the set; the judge decides | Precision matters: cache hits, RAG answers |

Embeddings serve recall only, and no decision path reads them. The
pre-declared `vela-embedding` deployment starts only when a semantic cache,
memory or a vector-store RAG source needs it.
`global.model_catalog.embeddings.default: {deployment: ...}` names another
embedding deployment, the way `system.decision_model` names the decision
model, and a store indexed with another model names its own in an
`embedding: {deployment, dimension}` block. `recall: {method: bm25}` needs no
model at all. A remote `openai_embeddings` deployment declares the
`dimensions` and `modalities` of its vectors in `capabilities`. See
[Embeddings](../model-runtime/guides/embeddings).

## What a response shows

Every routed response carries the public header set: the selected recipe,
decision and model, `x-vsr-response-path`, and `x-vsr-warnings` with any code
a `response_guard` or the hallucination check raised. What System One
answered rides in one debug header, `x-vsr-trace`, and only for a caller
`global.router.headers.debug` allows:

- `signals` lists the matched rules as `type:name`, such as `jailbreak:attack`
  or `systemone:needs_tools`, and `type:rule:option` for a choice rule without
  labels;
- `confidence`, `algorithm` and `ranking` show how the decision and its model
  were chosen;
- `modality` lists the selected decision's output kinds, such as
  `(text image)`.

A judge's truncation is the `match.judge.truncated` span event, not a trace
member. `POST /api/v1/routing/preview` evaluates the same rules for a request
without calling a model. [VSR routing headers](../troubleshooting/vsr-headers)
lists every header and trace member.
