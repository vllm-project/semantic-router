# Safety

## Overview

The `safety` signal predicts content risks. It complements `jailbreak`, which
predicts prompt attacks, and `pii`, which finds entity spans. Each safety rule
has an overall unsafe threshold and may select specific hazard categories.

## What Problem Does It Solve?

Content risk and prompt attacks require different labels. A benign discussion
of a harmful topic can be safe, while a harmful request need not contain any
attempt to override system instructions. Safety extracts content-risk scores;
Guard supplies the separate jailbreak signal.

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
    - name: handle-content-risk
      priority: 300
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: safety
            name: unsafe-content
      modelRefs:
        - model: safety-capable-model
```

Replace `safety-capable-model` with an alias in `providers.models` whose backend
handles content risks appropriately. A positive signal can include a person
seeking help in a crisis; it does not imply malicious intent or justify a blanket
refusal. Use category-specific decisions when a refusal is appropriate. Safety
signals extract scores; the consuming decision chooses the response policy.
Keep prompt-attack decisions alongside content-risk decisions and choose their
relative priorities explicitly.

A model error yields an unknown signal. `rules.on_unknown: fail_request`
returns HTTP 503 when the decision remains unknown. A scored unsafe request
selects the configured handling route. Diagnostics expose matched rule names in
`x-vsr-matched-safety`, the classification result, dashboard and replay record.

See [shared model configuration](/docs/installation/runtime/safety)
for native context budgets, external endpoints and failure policies, and the
[complete HTTP example](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/safety/content-safety.yaml)
for a category-specific policy.

## Select Vela Shield

The built-in Safety module uses
[Vela Safety](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Safety)
by default.
[Vela Shield](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Shield)
is a separately trained alternative with the same `safe`/`unsafe` labels, so
existing rules and thresholds apply without other changes. Validate thresholds
again after switching models.

To use Shield for every safety rule, set the module's model:

```yaml
global:
  model_catalog:
    modules:
      safety:
        safety:
          model_id: models/Vela-1.0-Encoder-307M-Shield
```

To use Shield for one rule in one recipe, declare a deployment and bind the
rule to it. Other recipes keep the module's model:

```yaml
global:
  model_catalog:
    deployments:
      shield:
        artifact: models/Vela-1.0-Encoder-307M-Shield
        revision: a981a99eeb05a2859b88b5cee9af4352897ec4ec
        provider: candle
recipes:
  - name: care
    routing:
      model_bindings:
        safety.unsafe-content:
          deployment: shield
          adapter: modernbert
          contract: label_distribution.v1
      signals:
        safety:
          - name: unsafe-content
            threshold: 0.5
```

The module form uses the pinned revision from the built-in registry; a
deployment uses the `revision` it declares. Either way, the download contains
only the root classifier. The Shield repository also publishes auxiliary heads
under `heads/` and a label-conditioned encoder under `lc/`; the router does not
load them and does not download them.

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

For a named local binding, deployment `input.max_tokens` replaces the module's
complete-document budget. A document budget of 65,536 and a `window.size` of
32,768 are valid when the loaded checkpoint and graph support 32K forwards.
Only the window must fit the single-forward capacity; the document may require
multiple windows. Counts come from the classifier tokenizer, not the downstream
generative model's tokenizer. A failed window fails the complete scan instead of
returning a successful score for only the inspected prefix.

The scan uses original token IDs, covers every content token, and leaves the
last window short. It sums selected unsafe probabilities within each window,
then takes the largest window score and applies the rule's threshold once.
Hazard similarly takes the maximum selected category probability across
windows. Set a separate `window` under the `hazard` head if that artifact has
been evaluated with scanning. External classifiers retain their own input
processing contract. The published Vela Hazard operating point retains its
supplied 2,048-token windows and 32K document policy; configuring a larger
document budget does not qualify that operating point for a different scan.

Choose thresholds evaluated with the exact model, window size, overlap and
precision you deploy. Window scanning can recover local risks that a whole-input
classifier misses, but it cannot interpret a distant refusal or protective
purpose outside the same window. Validate quoted material and other long-range
context in your application. Omit `window` when whole-input semantics are needed.
