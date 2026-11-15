# Keyword Signal

## Overview

`keyword` matches explicit words and phrases in the request. Define keyword
rules under `recipes[].signals.keywords`.

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
recipes:
  - name: default
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
recipes:
  - name: default
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

### Counting matches

Set `count` or `density` on an `OR` rule to match on how often its keywords
occur instead of whether one occurs. `count` compares the number of matches;
`density` compares matches per 1,000 characters of the text the rule reads.
Both take the predicate fields `gt`, `gte`, `lt`, and `lte`:

```yaml
recipes:
  - name: default
    signals:
      keywords:
        - name: many_questions
          description: Prompts that ask four or more explicit questions.
          operator: OR
          keywords: ["?", "？"]
          count: {gte: 4}
        - name: numbered_steps
          operator: OR
          method: regex
          keywords: ['(?m)^\s*\d+[.)]\s+']
          count: {gte: 2}
        - name: constraint_dense
          operator: OR
          keywords: ["must", "must not", "at least", "at most", "exactly"]
          density: {gt: 16}
```

A literal keyword counts each occurrence that stands as a word of its own;
CJK keywords, and keywords with no letters or digits such as `?`, count every
occurrence. With `method: regex`, each pattern counts its non-overlapping
matches. BM25, n-gram, and fuzzy matching score similarity rather than locate
occurrences, so a rule that counts uses literal keywords or `method: regex`.

The rule publishes the number as its value whether or not its predicate holds:
a decision condition can apply its own `predicate` to `keyword:<rule>`, and a
projection input can read it with `value_source: raw`.

## Masking What a Rule Finds

An `OR` rule of literal keywords or `method: regex`, without `fuzzy_match`,
`count` or `density`, says where each match is, so the
[masking plugin](../../plugin/masking#masking-your-own-rules) can hide what
it finds before the request reaches the provider. Name it in the plugin's
`rules` as `{type: keyword, name: <rule>}`.

## Dependencies and Limitations

Keyword rules require no model, but they are sensitive to wording and can be
triggered intentionally. Use them for routing hints, not security boundaries,
and test false positives across supported languages. See complete examples for
[`nlp.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/keyword/nlp.yaml)
and
[`regex.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/keyword/regex.yaml).
