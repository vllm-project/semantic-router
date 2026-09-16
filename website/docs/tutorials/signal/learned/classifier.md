# Classifier Signal

## Overview

`classifier` exposes reusable label scores from a local native sequence classifier,
a remote sequence classifier, or a configured external LLM. Decisions test a
declared label using a numeric predicate or an explicitly bound operating point.

Specialized domain, PII, jailbreak, fact-check, KB, and preference signals
remain the preferred interfaces for their respective domains.

## Key Advantages

- integrates arbitrary sequence-classification heads without adding domain logic
- constrains LLM classifiers to declared labels and deterministic JSON output
- computes one label map that multiple decisions can gate at different scores

## What Problem Does It Solve?

Some trained classifiers do not belong to the built-in signal taxonomies. The
classifier signal exposes those labels and scores to decisions without mixing
classification with route outcomes.

## When to Use

Use this signal for a genuine reusable classification head or a prompted LLM
labeler. Prefer embedding/KB signals for reference-phrase similarity and
preference signals for response-style routing.

## Configuration

```yaml
routing:
  signals:
    classifiers:
      - name: phishing
        type: local
        model_path: models/phishing-email
        labels: [BENIGN, PHISHING]
        use_cpu: true

  decisions:
    - name: phishing-local
      description: Keep suspected phishing requests on the local model.
      priority: 200
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: classifier
            name: phishing
            label: PHISHING
            predicate:
              gte: 0.5
      modelRefs:
        - model: local-small
          use_reasoning: false
```

LLM classifiers reference a named `global.model_catalog.external` entry and
add `instructions`. The runtime fixes temperature, output schema,
exact-label validation, and a 1 MiB default response limit. Set
`max_response_bytes` on the external model entry to override that limit.
Because the runtime owns the output schema, `parser_type` on that entry
must be `json` or unset; other values are rejected at config load.
The model must report a score for every declared
label; each score must be between `0` and `1`, and the complete distribution
must sum to approximately `1.0`. These are model-reported confidence scores,
not calibrated classifier probabilities. Classifier leaves are the only
decision predicates that accept `on_error`; failures expose the bounded
`classifier_evaluation_failed` code in eval/replay diagnostics.

On failure, the decision tree evaluates this leaf as `Unknown` until the full
AND/OR/NOT expression is known. Root-level `rules.on_unknown` then chooses
`no_match`, `match`, or `fail_request`. `no_match` and `match` resolve only
their own decision; `fail_request` is global fail-closed: it rejects the whole
request with a 503 even when another decision matches cleanly, regardless of
priority. When `rules.on_unknown` is omitted, condition-level `on_error`
(`no_match` or `match`) preserves the previous generic-classifier result.
Setting `rules.on_unknown` disables every condition-level `on_error` in that
tree, so the Router rejects a configuration that sets both.
`prompt_guard.on_error` (`allow` or `block`) remains the compatibility
default for jailbreak rules. Diagnostics include both the signal error and any
terminal policy that was applied. See
[Safety models](../../../installation/runtime/safety).

`sequence_classifier` classifiers also reference a named external model, but
use the shared `http_classify` contract and preserve its full label distribution.
The response must contain exactly the declared labels, with scores that sum to
approximately `1.0`; sigmoid multi-label outputs and label subsets are rejected.
They require at least two labels and do not accept `instructions`, `model_path`,
or `use_cpu`.

Local classifiers use `model_path` and support two or more declared labels.
The native Candle backend reads `model_type` from the checkpoint's `config.json`
and supports classic BERT and ModernBERT, including mmBERT. Set `use_cpu: true`
for CPU execution. The declared labels must match the checkpoint's numeric
`id2label` order.
Each rule owns a prepared model handle, so a recipe can declare multiple local
classifiers. Local decision predicates retain `gte: 0.5` or higher. Model or
label changes prepare a candidate generation before activation; a failed
candidate leaves the current generation available.

A recipe can select execution explicitly with a `classifier.<rule name>` entry
in `model_bindings`; this replaces the rule's `model` or `model_path` selector.
Local and sequence rules support local sequence deployments or HTTP
`http_classify`, while LLM rules retain their scored extraction instructions
and require HTTP `http_chat`. These use `label_distribution.v1`, with the rule's
ordered `labels` as the mapping. See [In-process models](../../../installation/runtime/in-process).

## Independent labels with a frozen operating point

A multi-label classifier, such as Hazard, can run independently of the Safety
signal. Bind `label_scores.v1` and an explicit version-2 operating-point file.
The binding has only its path and SHA256; model, tokenizer, execution, label
order, window geometry and thresholds are bound inside that file. There is no
implicit file discovery. Relative paths resolve inside the deployment artifact.

```yaml
routing:
  model_bindings:
    classifier.content-risk:
      deployment: content-risk-cpu
      adapter: modernbert
      contract: label_scores.v1
      operating_point:
        path: operating_point.json
        sha256: <SHA256 of the exact version-2 sidecar>
  signals:
    classifiers:
      - name: content-risk
        type: local
        labels: [violence, criminal_activity, sexual_content, child_exploitation,
                 hate, harassment_abuse, regulated_substances, weapons, self_harm,
                 privacy, specialized_advice, misinformation]
  decisions:
    - name: weapon-risk
      priority: 100
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: classifier
            name: content-risk
            label: weapons
      modelRefs:
        - model: answer-model
global:
  model_catalog:
    deployments:
      content-risk-cpu:
        provider: candle
        artifact: models/content-risk
        device: cpu
        precision: fp32
        input:
          max_tokens: 32768
          overflow: reject
```

Replace the SHA256 placeholder and use the artifact's complete ordered labels.
The document budget must equal the sidecar's budget. Omitting `predicate` uses
that label's frozen threshold (`score >= threshold`); several labels or none
may match. An explicit predicate queries the raw independent score instead.
Scores do not sum to one. Categorical and unbound classifiers still require
their existing predicates.

The policy supports Candle float32 or an explicitly qualified ORT native graph.
The sidecar binds the graph and every external tensor file by SHA256, the
execution provider, and the physical window capacity. A different graph,
precision conversion or unlisted provider is rejected. The document budget
remains separate from each window’s execution budget.

It tokenizes once, covers the original content token IDs with the
declared overlapping windows, restores special tokens, resets positions, and
takes the maximum sigmoid score for each label. It rejects document overflow,
incomplete scans, changed artifacts and unsupported execution. Version 1 lacks
the required identities and is rejected. The existing Safety/Hazard combination is
unchanged.

Eval's `metrics.classifier.rules` records policy SHA256, actual provider, device,
precision, token usage, content-window offsets, thresholds and elapsed time.
The current owned executor runs one window at a time; its latency includes the
complete scan. A reference batch size in the sidecar describes calibration
provenance, not the runtime batch size. Execution errors remain `Unknown`,
including under `NOT`, and follow the existing `on_unknown` policy.

To bind an already selected runtime-only score policy to final native files,
run the packaging tool from `src/semantic-router`:

```bash
go run ./cmd/classifier-operating-point \
  --model /path/to/native-model \
  --policy /path/to/selected-score-policy.json \
  --output /path/to/new-operating-point.json
```

It prints the sidecar SHA256, preserves score/window fields and verifies the
existing weight identity. It adds final config/tokenizer hashes and the
Candle execution identity for version 1; version-2 execution declarations are
preserved and their files verified. It neither selects thresholds nor qualifies a
model, and refuses to overwrite an existing file. Publish this sidecar with the
exact native files; do not copy thresholds between checkpoints.

The local path processes request text inside the Router. Both `llm` and
`sequence_classifier` send that text to their configured external model, so
choose the provider and retention policy accordingly. Labels and thresholds
must be evaluated as one versioned contract. See complete examples for
[`llm`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/label-score.yaml)
and
[`sequence_classifier`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/sequence-label-score.yaml).
