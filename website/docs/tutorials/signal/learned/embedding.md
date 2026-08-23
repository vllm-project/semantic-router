# Embedding Signal

## Overview

`embedding` matches requests by semantic similarity to representative examples.
Define embedding rules under `routing.signals.embeddings`.

It depends on the embedding model configured in
`global.model_catalog.embeddings`.

Those assets can run locally or through an external OpenAI-compatible text embedding endpoint. See [Remote Embedding Providers](../../global/remote-embeddings) for the shared provider configuration; signal candidates, thresholds, and decision conditions remain unchanged.

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
document:
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

Configure ranked fallback behavior with the embedding model settings:

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

`prototype_scoring` compresses each embedding rule's candidate bank into a smaller set of representative prototypes, then scores the rule from those prototypes instead of relying on one flat candidate list forever.

The Router scores every embedding rule and then applies `top_k` as the
emission limit. The default is `1`, so only the strongest embedding signal is
returned. Set `top_k: 0` to return every rule that meets its threshold.

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
- Use `aggregation_method: max` when any strong example should match. Use
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

document:
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
document:
  decisions:
    - name: route_medical_imagery_on_prem
      description: Keep medical imagery on the in-cluster vision model.
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: medical_imagery_phi
```

### Image-routing example

The optional
[`image-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml)
example includes `identifier_document_imagery`,
`code_or_terminal_imagery`, and a benign `ambient_office_imagery` rule. Replace
its candidates with examples from your deployment and recalibrate the
threshold. The example value is not a portable default.

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
