# Keyword Signal

## Overview

`keyword` matches explicit words and phrases in the request. Define keyword
rules under `routing.signals.keywords`.

This family is heuristic: it routes from configured words, phrases, or lightweight retrieval methods instead of learned intent classifiers.

## Key Advantages

- Gives deterministic routing for obvious lexical cases.
- Supports simple regex-style matching and stronger BM25 or n-gram variants.
- Is easy to audit because the trigger phrases are explicit.
- Often provides the fastest path to a useful routing graph.

## What Problem Does It Solve?

Some routes do not need a full classifier. They just need to recognize stable words such as billing, password reset, or urgent support.

`keyword` solves that by turning lexical matches into reusable named signals instead of scattering string checks across decisions.

## When to Use

Use `keyword` when:

- lexical cues are stable and high-signal
- you want deterministic routing for support or policy keywords
- you need a low-latency first pass before learned signals
- prompt wording matters more than semantic paraphrase coverage

## Configuration

```yaml
routing:
  signals:
    keywords:
      - name: code_keywords
        operator: OR
        method: bm25
        keywords: ["code", "function", "debug", "algorithm", "refactor"]
        bm25_threshold: 0.1
        case_sensitive: false
      - name: urgent_keywords
        operator: OR
        method: ngram
        keywords: ["urgent", "immediate", "asap", "emergency"]
        ngram_threshold: 0.4
        ngram_arity: 3
        case_sensitive: false
```

Use plain keyword lists for simple matching, then add `method: bm25` or `method: ngram` when exact text matching becomes too brittle.

### Matching earlier user messages

By default a keyword rule evaluates only the current (latest) user message.
Set `include_history: true` on a rule when its keywords state a constraint
that should keep applying for the rest of the conversation, such as a
privacy or local-only marker:

```yaml
routing:
  signals:
    keywords:
      - name: local_only_markers
        operator: OR
        keywords: ["local model only", "do not upload to cloud"]
        include_history: true
```

With `include_history: true`, the rule also matches the most recent 20 prior
user messages. Assistant and system messages are never included. Messages are
joined with newlines, so an exact-match phrase cannot match across two
messages; BM25 and n-gram rules score words rather than phrases, so they do not
keep that boundary.

Leave `include_history` off (the default) for keywords that describe what the
current message asks for, such as reasoning or architecture markers;
otherwise a phrase from an earlier turn keeps triggering the rule on every
later request.

## Dependencies and Limitations

Keyword rules require no model, but they are sensitive to wording and can be
triggered intentionally. Use them for routing hints, not security boundaries,
and test false positives across supported languages. See complete examples for
[`nlp.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/keyword/nlp.yaml)
and
[`regex.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/keyword/regex.yaml).
