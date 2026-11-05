# Classifier Signal

## Overview

`classifier` exposes reusable label scores from a local native sequence classifier,
a remote sequence classifier, or a configured external LLM. Decisions test a
declared label using a numeric predicate or the operating point its deployment
pins.

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

Each classifier rule asks its own question, so it names the deployment that
answers it with `deployment`. A rule without `deployment` is asked of the
decision model. [Choose a model for a signal](../../../installation/choose-a-model-for-a-signal.md)
explains how rules resolve.

```yaml
global:
  model_catalog:
    deployments:
      phishing-model:
        provider: model_runtime
        artifact: /models/phishing-email
        device: cpu

routing:
  signals:
    classifiers:
      - name: phishing
        deployment: phishing-model
        labels: [BENIGN, PHISHING]

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

The deployment's `provider` decides how the rule is asked:

| `provider` | How the rule is answered |
| --- | --- |
| `model_runtime` | A sequence classifier in the model runtime scores the declared labels |
| `classify` | The Router posts the text to `<endpoint>/classify` and reads the full label distribution |
| `openai_chat` | A chat model scores the declared labels, following the rule's `instructions` |
| (no `deployment`) | The decision model answers a label question |

### Chat models as classifiers

A rule answered by an `openai_chat` deployment adds `instructions`. The Router
fixes the output schema and exact-label validation, and reads responses up to
1 MiB unless the deployment's `limits.max_response_bytes` sets another limit.
It asks at temperature 0; the deployment's `generation` sets another
temperature or completion length.

```yaml
global:
  model_catalog:
    deployments:
      risk-judge:
        provider: openai_chat
        endpoint: https://llm.example.com/v1
        model: qwen/qwen3.8-27b
        auth: {api_key_env: RISK_JUDGE_KEY}
        timeout_ms: 5000
        generation: {max_tokens: 256}

routing:
  signals:
    classifiers:
      - name: tool-risk
        deployment: risk-judge
        labels: [SAFE, RISKY]
        instructions: Choose RISKY only for destructive or irreversible tool operations.
```

Set `disable_rationale: true` on such a rule to request only
`{"scores": {...}}`. The response may omit `rationale` or include an empty
string; if present, it must still be a string. Other response fields are
rejected. When the flag is omitted or `false`, the prompt requests both
`scores` and `rationale`, and the response must include a nonempty rationale.
The flag applies only to rules answered by an `openai_chat` deployment.
The model must report a score for every declared
label; each score must be between `0` and `1`, and the complete distribution
must sum to approximately `1.0`. These are model-reported confidence scores,
not calibrated classifier probabilities. Classifier and System One leaves
accept condition-level `on_error`; classifier failures expose the bounded
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
[Safety models](../../../model-runtime/guides/safety.md).

### Remote classification services

A rule answered by a `classify` deployment preserves the service's full label
distribution. The response must contain exactly the declared labels, with
scores that sum to approximately `1.0`; sigmoid multi-label outputs and label
subsets are rejected. Such a rule needs at least two labels and takes no
`instructions`.

### Local classifiers

A rule answered by a `model_runtime` deployment runs in the
[model runtime](../../../model-runtime/guides/classify.md#use-your-own-classifier),
which loads Hugging Face ModernBERT and mmBERT sequence classifiers; other
architectures need a [family plugin](../../../model-runtime/plugins.md). The
deployment's `device` decides where it runs. The rule declares two or more
labels, which must match the checkpoint's numeric `id2label` order.
Each rule owns a prepared model handle, so a recipe can declare multiple local
classifiers. Local decision predicates retain `gte: 0.5` or higher. Model or
label changes prepare a candidate generation before activation; a failed
candidate leaves the current generation available.

The rules in a recipe's own `routing.signals` name their deployments the same
way, so each recipe chooses the models of its rules. The rule's ordered
`labels` are its label mapping. See [Run it with the router](../../../model-runtime/deploy.md).

## Independent labels with a frozen operating point

A multi-label classifier, such as Hazard, can run independently of the Safety
signal. Answer the rule with a model runtime deployment whose package ships a
version-2 `operating_point.json`. Model, tokenizer, execution, label order,
window geometry and thresholds are bound inside that file, and the runtime
verifies it before serving. The deployment's `operating_point` pins the file:
`path` names it inside the deployment artifact and `sha256` its bytes. The rule
then reads independent label scores. Preparation rejects the deployment when
the runtime serves a policy with a different digest.

```yaml
routing:
  signals:
    classifiers:
      - name: content-risk
        deployment: content-risk-cpu
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
        provider: model_runtime
        artifact: /models/content-risk
        device: cpu
        input:
          max_tokens: 32768
          overflow: reject
        operating_point:
          path: operating_point.json
          sha256: <SHA256 of the exact version-2 sidecar>
```

Replace the SHA256 placeholder and use the artifact's complete ordered labels.
The document budget must equal the sidecar's budget. Omitting `predicate` uses
that label's frozen threshold (`score >= threshold`); several labels or none
may match. An explicit predicate queries the raw independent score instead.
Scores do not sum to one. Rules answered without an operating point still
require a predicate.

The model runtime applies the policy at the model's published FP32 precision
(`dtype: auto`). The sidecar
binds the weights, tokenizer and window geometry by SHA256; a changed artifact
is rejected. The document budget remains separate from each window's execution
budget.

It tokenizes once, covers the original content token IDs with the
declared overlapping windows, restores special tokens, resets positions, and
takes the maximum sigmoid score for each label. It rejects document overflow,
incomplete scans, changed artifacts and unsupported execution. Version 1 lacks
the required identities and is rejected. The existing Safety/Hazard combination is
unchanged.

Eval's `metrics.classifier.rules` records policy SHA256, actual provider, device,
precision, token usage, content-window offsets, thresholds and elapsed time.
Its latency includes the complete scan. A reference batch size in the sidecar
describes calibration provenance. Execution errors remain `Unknown`,
including under `NOT`, and follow the existing `on_unknown` policy.

Version 3 policies that bound the retired ONNX Runtime CK Flash Attention path
are no longer accepted; re-qualify the policy on the model runtime at
`dtype: auto`.

To bind an already selected runtime-only score policy to final native files,
run the packaging tool from `src/semantic-router`:

```bash
go run ../../tools/models/classifier-operating-point/main.go \
  --model /path/to/native-model \
  --policy /path/to/selected-score-policy.json \
  --output /path/to/new-operating-point.json
```

It prints the sidecar SHA256, preserves score/window fields and verifies the
existing weight identity. It adds final config/tokenizer hashes;
version-2 execution declarations are preserved and their files verified.
The tool neither selects thresholds nor qualifies a
model, and refuses to overwrite an existing file. Publish this sidecar with the
exact native files; do not copy thresholds between checkpoints.

A `model_runtime` deployment processes the request text locally. `openai_chat`
and `classify` deployments send that text to a remote service, so choose the
provider and retention policy accordingly. Labels and thresholds must be
evaluated as one versioned contract. See complete examples for
[a chat model](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/label-score.yaml)
and
[a classification service](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/sequence-label-score.yaml).
