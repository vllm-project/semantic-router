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

### Matching tool results

Set `input.tool_results: true` on a rule to match the textual tool results in
the request, and `input.request: false` to leave the current user message
out. Coding-agent harnesses place the output of every tool call, such as a
file read or a shell command, in a tool result and resend the whole
conversation on each turn, so this is where a secret a tool happened to read
appears. A rule that leaves the request out never sees the prompt, and the
earlier user messages stay behind `include_history`, so a rule with
`input.request: false` names at least one of the two. The pii signal reads
the same parts under the same names. Only literal and regex keywords may
read tool results; the scored methods, `fuzzy_match` and `count` or
`density` are built for user prose, not for recognizing a secret's shape.

```yaml
recipes:
  - name: default
    signals:
      keywords:
        - name: credential_cloud_keys
          operator: OR
          method: regex
          keywords: ['\b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b']
          case_sensitive: true
          input:
            tool_results: true
            request: false
          on_unscanned: match
```

Each tool result is matched on its own: an `AND` rule needs all of its
keywords inside the same tool result, and a `NOR` rule matches only when
every text it reads is free of its keywords. A request without tool results
never matches a rule that reads only them. Only text blocks are inspected. A
keyword found in a tool result the Router read counts whatever `on_unscanned`
says; when none was found and a tool result could not be read as text, such
as an image, `on_unscanned` decides: `best_effort` (the default) decides on
the text read, `match` counts the request as a match even when no tool result
holds text (strict for an `OR` or `AND` rule), and `no_match` does not match
the rule (strict for a `NOR` rule, which otherwise reads the unread tool
result as clean).

The matched keywords reported in the `x-vsr-matched-keywords` debug header,
in Router Replay and in logs are the configured patterns, never the matched
text, so a pattern written for a secret does not copy the secret into
diagnostics.

[`credentials.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/keyword/credentials.yaml)
carries three ready-made rules, `credential_cloud_keys`,
`credential_access_tokens` and `credential_private_keys`, whose patterns are
taken from [gitleaks](https://github.com/gitleaks/gitleaks) under its MIT
license. They cover provider-issued keys with a recognizable prefix and PEM
private-key blocks; generic passwords and high-entropy strings without a
prefix are out of their reach. The AWS rule recognizes the access key id
(`AKIA…`), not the secret access key beside it, which has no prefix and
which gitleaks finds only through its entropy rules.

### Measuring accuracy of the credential rules

[`keyword_credentials_eval_test.go`](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/classification/keyword_credentials_eval_test.go)
scores the three `credentials.yaml` rules through the real keyword matcher on
Samsung's [CredData](https://github.com/Samsung/CredData) (Apache-2.0), 11,030
source files with 67,896 labeled lines of which 15,753 are true credentials,
each file scanned as one tool result, and on a negative corpus such as this
repository's own tree. The test is skipped unless the dataset path is set:

```bash
git clone https://github.com/Samsung/CredData
(cd CredData && pip install -r requirements.txt && python download_data.py --jobs 8)
cd src/semantic-router
SR_CREDDATA_DIR=/path/to/CredData SR_CREDENTIAL_NEGATIVE_DIR=/path/to/semantic-router \
  go test ./pkg/classification/ -run TestCredentialKeywordRulesAccuracy -v
```

Precision counts flagged lines that carry a true label; recall counts the true
lines in each rule's reach, that is PEM blocks, AWS and Google keys, and Slack,
Stripe, GitHub and Anthropic tokens. CredData at commit `86a9182`:

| Rule | Flagged lines | Precision | In-scope true lines | Recall |
| --- | --- | --- | --- | --- |
| `credential_private_keys` | 1,253 | 95.2% | 1,193 | 100% |
| `credential_cloud_keys` | 128 | 86.7% | 172 | 64.5% |
| `credential_access_tokens` | 9 | 100% | 19 | 52.6% |

CredData replaces every labeled value with a random string of the same shape,
and 56 of the 61 cloud-key misses are AWS keys whose replacement left the
key's base32 alphabet. The token misses are two-segment legacy Slack tokens, a
Stripe publishable key and a truncated Anthropic key, none of which the
gitleaks patterns accept. The flagged lines without a true label are PEM
headers in documentation and test fixtures, encrypted test keys, and AWS
documentation examples such as `AKIAIOSFODNN7EXAMPLE`; gitleaks removes those
with allowlists the keyword signal does not carry, so a tool result that prints
a placeholder key matches too. Across all 15,753 true lines the rules flag
8.9%: generic passwords, UUIDs and high-entropy strings without a prefix are
not in their reach. On this repository's 11,962 text files the rules flag 11:
test fixtures and comments that quote the AWS documentation example, the
rule set itself and this page.

## Dependencies and Limitations

Keyword rules require no model, but they are sensitive to wording and can be
triggered intentionally. Use them for routing hints, not security boundaries,
and test false positives across supported languages. See complete examples for
[`nlp.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/keyword/nlp.yaml),
[`regex.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/keyword/regex.yaml)
and
[`credentials.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/keyword/credentials.yaml).
