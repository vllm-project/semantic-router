# Embedding Signal

## Overview

`embedding` matches requests by semantic similarity to representative examples.
Define embedding rules under `routing.signals.embeddings`.

It depends on the embedding model configured in
`global.model_catalog.embeddings`.

Those assets can run locally or through an external OpenAI-compatible text embedding endpoint. See [Runtime embeddings](../../../installation/runtime/embeddings) for the shared provider configuration; signal candidates, thresholds, and decision conditions remain unchanged.

## Key Advantages

- Handles paraphrases better than plain keyword rules.
- Lets teams tune routing with example phrases instead of retraining a classifier.
- Works well for support intents, product flows, and semantic FAQ routing.
- Provides a smooth step up from purely lexical signals.

## What Problem Does It Solve?

Keyword routing misses semantically similar prompts that use different wording. Full domain classification can also be too coarse when the route depends on a narrow intent.

`embedding` solves that by matching new prompts against example candidates in embedding space.

## When to Use

Use `embedding` when:

- phrasing varies but intent stays stable
- you want semantic routing without introducing a full custom classifier
- examples are easier to maintain than domain labels
- support or workflow intents need better recall than keywords can provide

## Configuration

```yaml
routing:
  signals:
    embeddings:
      - name: technical_support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - how to configure the system
          - installation guide
          - troubleshooting steps
          - error message explanation
          - setup instructions
      - name: account_management
        threshold: 0.72
        aggregation_method: max
        candidates:
          - password reset
          - account settings
          - profile update
          - subscription management
          - billing information
```

Tune the threshold and candidate list together; that matters more than adding many low-quality examples.

By default, a rule matches only when its similarity score reaches its
`threshold`. Unmatched scores remain available to numeric predicates and
projections.

For ranked intent selection, you can explicitly enable soft matching below.
When no rule meets its threshold, this permits matches above
`min_score_threshold` instead. Leave it disabled when a rule's threshold must
be a strict boundary, such as a risk or privacy condition.

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          enable_soft_matching: true
          top_k: 1
          min_score_threshold: 0.5
          prototype_scoring:
            enabled: true
            cluster_similarity_threshold: 0.9
            max_prototypes: 8
            best_weight: 0.75
            top_m: 2
            margin_threshold: 0.05
```

The family-level `prototype_scoring` settings compress each rule's candidate
bank and control its scoring. A rule can declare its own `prototype_scoring`
object beside `candidates`. Omitting it inherits the family settings; declaring
it replaces the complete object, with omitted fields using built-in defaults.
An empty object therefore uses built-in defaults rather than family overrides.

To retain every distinct authored candidate, including multilingual examples,
set this on the rule:

```yaml
prototype_scoring:
  enabled: false
  best_weight: 0.75
  top_m: 2
```

`enabled: false` disables clustering and the prototype cap, not aggregation.
`max` still combines the best similarity and top-M support; `mean` still averages
the retained bank. With compression enabled, `max_prototypes: 0` uses the default
cap of 8. Retaining more candidates adds local scoring work; candidate embeddings
are already computed before compression and request embedding calls are unchanged.
Rule settings travel with an exported or initialized recipe and apply to both
text and image queries.

The Router scores every embedding rule. By default, `top_k: 0` retains every
rule that meets its threshold, so independent predicates remain available to
projections and decision priority. Set a positive `top_k`, such as `1` in the
ranked example above, only when lower-ranked matches should be discarded.
This limits emitted evidence; it does not reduce embedding inference work.
Use recipe-local [partitions](../../projection/partitions) when a specific
group of competing signals should have one winner.

## Design and validate candidate sets

Treat a rule's `candidates` as a small semantic classifier, not as a keyword
list:

- Describe the kind of input you want to recognize. For text, use varied
  examples of the intent rather than several versions of one sentence. For an
  image rule, describe visible structure such as `photograph of a passport
  page`; do not rely on literal words that would require OCR.
- Cover the category from several distinct angles. Near-duplicate candidates
  add little recall and can make a rule look better calibrated than it is.
- Include routine, benign examples in your evaluation set. When winner-style
  emission is appropriate, a competing benign rule can also give ordinary
  inputs a better semantic match. Test it with your actual `top_k` and
  threshold settings; a benign rule is not a security blocklist.
- Use `aggregation_method: max` to favor the strongest example while considering
  support from other prototypes. The rule threshold applies to the combined
  score, so the best individual similarity can exceed it without a match. Use
  `mean` only when broad agreement across the candidate set is the behavior
  you want.
- Calibrate `threshold` against labeled positive and negative traffic for the
  deployed model. Thresholds do not transfer reliably between models,
  dimensions, modalities, or traffic distributions.

Re-run the labeled evaluation whenever you change a candidate, model, or
threshold. Record those three inputs together so a configuration update does
not silently reuse an incompatible threshold.

## Multimodal queries (`query_modality`)

Each embedding rule accepts an optional `query_modality` field that declares which modality of incoming request payload the rule's query is computed from. The candidates remain text in every case; the rule cosine-matches the text-anchor set against a query embedding from the declared modality, all in the same shared multimodal space.

Accepted values:

- `"text"` (default, backward-compatible): query embedded from request text. Existing rules with no `query_modality` field behave exactly as before.
- `"image"`: query embedded from an allowlisted inline
  `data:image/...;base64,...` attachment in an OpenAI-style chat message.
- Audio query embeddings are not currently supported; use `text` or `image`.

`"image"` requires
`global.model_catalog.embeddings.semantic.embedding_config.model_type: multimodal`
so candidates and queries share one embedding space. The Router rejects an
image-modality rule paired with a text-only embedding model.

### Worked example: route sensitive imagery on-prem

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        multimodal_model_path: models/multi-modal-embed-small
        embedding_config:
          model_type: multimodal

routing:
  signals:
    embeddings:
      # query_modality defaults to text.
      - name: technical_support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - how to configure the system
          - installation guide
          - troubleshooting steps

      # Match an inline image attachment against text anchors in the shared
      # multimodal embedding space.
      - name: medical_imagery_phi
        query_modality: image
        threshold: 0.55
        aggregation_method: max
        candidates:
          - chest X-ray with patient identifier strip
          - dermatology lesion close-up photograph
          - electronic health record application screenshot showing patient demographics
          - ultrasound scan with patient name overlay
```

A decision can then route on the new signal the same way it routes on any other:

```yaml
routing:
  decisions:
    - name: route_medical_imagery_on_prem
      description: Keep medical imagery on the in-cluster vision model.
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: medical_imagery_phi
      modelRefs:
        - model: in-cluster-vlm
```

### Image-routing example

The optional
[`image-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml)
example includes `identifier_document_imagery`,
`code_or_terminal_imagery`, and a benign `ambient_office_imagery` rule. Replace
its candidates with examples from your deployment and recalibrate the
threshold. The example values are not portable defaults.

The shipped thresholds were obtained with
[`cmd/image-routing-calibration`](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/cmd/image-routing-calibration)
against `llm-semantic-router/multi-modal-embed-small` (snapshot
`fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4`, 384 dimensions, default
`prototype_scoring`) and a hand-reviewed manifest of repository images in which
every fixture is labelled positive for a named candidate, negative, or excluded
as ambiguous. Treat them as a starting point for that model only: calibrate
against your own labelled images before relying on the rules, and recalibrate
whenever you change the model, the candidates, or the scoring configuration.

### Distinction from the `modality` signal type

`query_modality` (this section) declares **input modality** for an embedding rule — which modality of payload the query is computed from. The separate [`modality`](modality) signal type declares **output modality** (`AR`, `DIFFUSION`, `BOTH`) for routing image-generation requests. The two concepts share a name but solve different problems and live on different config surfaces.

## Dependencies and Limitations

- Text or image content is processed by the configured embedding runtime. A
  remote provider currently supports text only and receives the text being
  embedded. Audio query embeddings are not supported.
- Similarity scores and thresholds are not portable across embedding models,
  dimensions, or modalities. Recalibrate whenever those change.
- Image matching is semantic rather than OCR or PII extraction; use a dedicated
  detector when literal text or regulated entities matter.
- Complete examples:
  [`support.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/support.yaml)
  and
  [`image-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml).
