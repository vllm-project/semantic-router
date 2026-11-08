# Plugins

## Overview

Plugins add route-local behavior after a decision matches. They can rewrite a
request, retrieve context, short-circuit generation, inspect a response, or
control what operational data is retained.

Shared services and stores belong under `global:`; the decision plugin only
enables and tunes that behavior for one route.

## Key Advantages

- Keeps behavior attached to the route that needs it.
- Reuses shared stores and services without duplicating their configuration.
- Makes request mutation, retrieval, and response inspection auditable.

## What Problem Does It Solve?

Routes often need different behavior even when they share the same Router.
Plugins keep those differences next to the decision instead of hiding them in
application middleware or enabling them globally.

## When to Use

Use a plugin when behavior should apply only after a specific route matches.
Use `global:` instead when every route shares the same service or backing
store. Plugin entries live under `recipes[].decisions[].plugins`.

## Configuration

```yaml
recipes:
  - name: default
    decisions:
      - name: cached-support
        description: Reuse cached responses for support requests.
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: support-model
        plugins:
          - type: response_cache
            configuration:
              enabled: true
              ttl_seconds: 3600
```

## Plugin Inventory

| Type | Goal | Shared dependency | Guide |
|---|---|---|---|
| `fast_response` | Return a configured response without calling a model | None | [Fast Response](./fast-response) |
| `system_prompt` | Insert or replace route-specific instructions | None | [System Prompt](./system-prompt) |
| `header_mutation` | Add, update, or delete provider-bound headers the Router does not own | None | [Header Mutation](./header-mutation) |
| `request_params` | Enforce request parameter limits | None | [Request Parameters](./request-params) |
| `tools` | Allow, block or remove tools and tool history; let the decision model filter or add the tools a request needs | Decision model for `select`; a tool catalog for `add` | [Tools](./tools) |
| `response_cache` | Reuse compatible prior responses | `global.stores.response_cache` | [Response Cache](./response-cache) |
| `prompt_cache` | Insert Anthropic prompt-cache breakpoints on instructions and tools | None | [Prompt Cache](./prompt-cache) |
| `context` | Retrieve documents and memories, compress the request within a budget, and store memories after the response | A RAG backend, `global.stores.memory`, the decision model for the relevance judge | [Context](./context) |
| `router_replay` | Override replay capture for one route, and shadow a sampled copy to a candidate model | `global.services.router_replay` | [Router Replay](./router-replay) |
| `response_guard` | Block or annotate an answer on the signals it is checked with | The response-stage signals its rules read | [Response Guard](./response-guard) |

[Content Safety](./content-safety) bundles three supported
plugins, not an additional plugin type.

## Capability Gating

A plugin that acts on request content runs only when the request needs that
content and the selected model can take it. The `context` plugin acts on
text, so it is skipped when the request has no text in its instructions or
messages. Text is assumed for every model, so a model card never gates them.
The model card gates only a plugin that acts on another modality, such as
images: it is skipped when the card declares capabilities and omits that
modality. A model with no recognized declaration is not gated. Other plugin
types are not gated and still run on configuration alone.

A skip is typed and carries no request content:

| Reason | Meaning |
|---|---|
| `request_does_not_require` | The request has nothing for the plugin to act on. |
| `backend_lacks_capability` | The selected model declares capabilities and omits a modality the plugin acts on. |

A skip is counted in `llm_plugin_execution_total` with status
`skipped_<reason>`. For a memory source, the same reason appears as the memory outcome
with status `skipped`. Non-text content such as images is left unchanged
whether a plugin runs or not.

## Similarity: the `match` Block {#match-block}

Every feature that finds similar items uses the same `match` block:

```yaml
match:
  recall: {method: embedding, top_k: 20, min_similarity: 0.3}                # narrows the candidates: embedding | bm25
  judge:  {preset: relevance, top_k: 6, min_score: 0.5, max_candidates: 64}  # decides with a System One question
```

- **Stages.** A block runs the stages it writes, at least one; an empty block
  fails to load with `match_empty`. A feature that writes no block runs its
  default (below); a block it writes replaces the default as a whole.
  - With `recall` only, the result is classic embedding or BM25 similarity.
  - With `judge` only, a System One question judges every candidate; this
    suits small, naturally bounded candidate sets.
  - With both, `recall` narrows the set and `judge` decides.
- **`recall`** narrows the candidates by index similarity:
  - `method: embedding` (the default) compares embeddings in the store's
    embedding space (`global.model_catalog.embeddings.default` unless the
    store names its own), and `min_similarity` is a cosine in [-1, 1];
    `method: bm25` uses the lexical index, needs no model, and
    `min_similarity` is a BM25 score;
  - `top_k` bounds the candidates; it is required when they come from an index.
- **`judge`** asks one System One question about the request and the
  candidates, in one call, on the decision model or the named `deployment`:
  - `preset: relevance | same_request | tool_rank`, or a custom `question`
    (`choice`, `noul` or `score`);
  - a choice question (`tool_rank`) chooses among the candidates, which are its
    options, in one standard request, and scores each with its probability; a
    noul or score question (`relevance`, `same_request`) is asked about each
    candidate, one batch entry per candidate, and scores it with P(yes) or the
    expected level;
  - `top_k`, `min_score` (absolute) and `min_ratio` (a fraction of the best
    score, in (0, 1]) keep candidates, best first;
  - `on_error: skip` (the default) keeps the recall result, or the candidates
    as they came without recall, when the judge fails or is late;
    `on_error: fail` fails the feature's match; `timeout_ms` bounds the call
    (else the deployment's `timeout_ms`, else 1000 ms).
- **`judge.max_candidates`** is a hard cap: at most 255 for a choice judge (the
  options of one question, its default), and 64 by default for a judge asked
  about each candidate. More candidates never fail: the judge reads the first
  `max_candidates` by recall rank, or as the feature lists them (most recent
  first) without recall. Each truncation adds a `match.judge.truncated` event to
  the request's trace span and counts in `vsr_match_judge_truncated_total`.

| Feature | Default `match` | Notes |
|---|---|---|
| `response_cache.semantic.match` | `recall: {method: embedding, top_k: 3, min_similarity: 0.8}`, `judge: {preset: same_request, min_score: 0.8}` | The judge confirms each recalled hit; a rejected one is a miss. Exact mode stays |
| `context.sources[rag].match` | `recall: {method: embedding, top_k: 20, min_similarity: 0.3}`, `judge: {preset: relevance, top_k: 6, min_score: 0.5}` | `recall: {method: bm25}` and a judge without recall read a local knowledge base: a `vectorstore` source over the in-memory vector store. A source over an external vector database recalls by `embedding` |
| `context.sources[memory].match` | `recall: {method: embedding, top_k: 5, min_similarity: 0.7}` | The memory store's `default_retrieval_limit` and `default_similarity_threshold` replace the recall defaults. A judge without recall reads the user's newest memories |
| `context.compress.match` | `recall: {method: bm25}`, `judge: {preset: relevance}` | Each segment of the request is a candidate about the latest user message; BM25 prefilters for the judge's 64; the budget bounds what is kept |
| `tools.select.match` | `judge: {preset: tool_rank, top_k: 3, min_ratio: 0.1}` | At most 255 tools; more are truncated, or narrowed by `recall: {method: bm25}` |

Every stage is counted in `vsr_match_recall_total` (feature, method, outcome)
or `vsr_match_judge_total` (feature, preset, outcome), and timed in
`vsr_match_stage_duration_seconds`.

## Operational Boundaries

- Plugins can interact when more than one mutates the provider-bound request or
  response. The Router pipeline fixes their execution order; reordering entries
  in YAML does not change it.
- Retrieval, memory, cache, and replay may persist request-derived content.
  Configure retention, tenant/user scope, authentication, and encryption for
  the selected backend.
- Header and prompt mutation can cross trust boundaries. Do not copy untrusted
  caller metadata into privileged headers or system instructions.
- Validate the complete recipe before deployment so unsupported plugin names or
  incompatible settings fail before traffic reaches the Router.
