# Safety

## Overview

The `safety` signal predicts content risks. It complements `jailbreak`, which
predicts prompt attacks, and `pii`, which finds entity spans. Each safety rule
has an overall unsafe threshold and may select specific hazard categories.

## What Problem Does It Solve?

Content risk and prompt attacks require different labels. A benign discussion
of a harmful topic can be safe, while a harmful request need not contain any
attempt to override system instructions. Safety extracts content-risk scores;
PromptGuard supplies the separate jailbreak signal.

## When to Use

Use a binary rule for a broad content policy, or attach a Hazard condition when
only selected categories should select a decision. Validate thresholds on your
application's languages and input lengths before enabling refusal policies.

## Configuration

```yaml
routing:
  signals:
    safety:
      - name: unsafe-content
        threshold: 0.5
      - name: unsafe-privacy
        threshold: 0.5
        hazard:
          labels: [violence, criminal_activity, sexual_content, child_exploitation,
                   hate, harassment_abuse, regulated_substances, weapons,
                   self_harm, privacy, specialized_advice, misinformation]
          categories: [privacy]
          threshold: 0.6
```

Omitted `model` uses the built-in local Safety or Hazard module. An explicit
`model` resolves a classification endpoint in `global.model_catalog.external`.
Both forms expose the same score semantics. For local artifacts, `labels` must
exactly match `config.json`'s `id2label` order; the loader checks the head's
`problem_type` as well. For external endpoints, labels are aligned by name.

- `labels` and `unsafe_labels` default to `[safe, unsafe]` and `[unsafe]`.
- A binary rule compares the sum of its selected unsafe label scores with
  `threshold`. Equality matches.
- A rule with `hazard` first requires the binary condition, then compares the
  maximum score among `categories` with `hazard.threshold`. Hazard uses
  independent probabilities; scores are never added or renormalized.
- Multiple rules with an identical model, label and activation contract share
  one prediction per request. Their thresholds and matches remain separate.
- Hazard is skipped when the overall unsafe threshold is not reached. This
  saves inference, but category recall also depends on the binary gate.

The categories listed above describe the Vela taxonomy. It does not provide
specialized labels for every policy concern, such as copyright, high-risk
governance or manipulation. A topic mention is not automatically harmful; use
the model's evaluation report and your own validation set to choose thresholds.

Reference a rule in a decision and define the action there:

```yaml
routing:
  decisions:
    - name: refuse-unsafe
      priority: 300
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: safety
            name: unsafe-content
      modelRefs:
        - model: default-model
      plugins:
        - type: fast_response
          configuration:
            message: I cannot help with this unsafe request.
```

Replace `default-model` with an alias in `providers.models`. Safety signals
extract scores; they do not refuse requests without a consuming decision.
Keep prompt-attack decisions alongside content-risk decisions and choose their
relative priorities explicitly.

A model error yields an unknown signal. `rules.on_unknown: fail_request`
returns HTTP 503 when the decision remains unknown. A scored unsafe request
selects the refusal policy. Diagnostics expose matched rule names in
`x-vsr-matched-safety`, the classification result, dashboard and replay record.

See [shared model configuration](../../global/safety-models-and-policy.md)
for native context budgets, external endpoints and failure policies, and the
[complete HTTP example](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/safety/content-safety.yaml)
for a category-specific policy.

## Long-input scanning

Native heads use whole-input inference by default. A separately calibrated
window policy can scan local risks throughout a long request:

```yaml
global:
  model_catalog:
    modules:
      safety:
        safety:
          model_id: models/my-safety-model
          max_sequence_length: 32768
          window:
            size: 512
            overlap: 255
```

`max_sequence_length` limits the complete tokenized request, including special
tokens. Longer inputs fail; the router never silently keeps only the first
window. `window.size` includes special tokens, while `overlap` counts content
tokens. For a tokenizer adding two special tokens, this example advances by
255 content tokens. Each window restores the original special tokens and
starts positions at zero. Empty content and invalid window budgets fail.

The scan uses original token IDs, covers every content token, and leaves the
last window short. It sums selected unsafe probabilities within each window,
then takes the largest window score and applies the rule's threshold once.
Hazard similarly takes the maximum selected category probability across
windows. Set a separate `window` under the `hazard` head if that artifact has
been evaluated with scanning. External classifiers retain their own input
processing contract.

Choose thresholds evaluated with the exact model, window size, overlap and
precision you deploy. Window scanning can recover local risks that a whole-input
classifier misses, but it cannot interpret a distant refusal or protective
purpose outside the same window. Validate quoted material and other long-range
context in your application. Omit `window` when whole-input semantics are needed.
