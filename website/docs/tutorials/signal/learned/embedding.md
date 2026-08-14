# Embedding Signal

## Overview

`embedding` matches requests by semantic similarity to representative examples. It maps to `config/fragments/signal/embedding/` and is declared under `routing.signals.embeddings`.

This family is learned: it depends on the semantic embedding assets in `global.model_catalog.embeddings`.

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

Source fragment family: `config/fragments/signal/embedding/`

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

Ranked fallback behavior is tuned separately under the router-owned embedding catalog:

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

### Default opt-in pack

The repo ships an opt-in image-modality embedding pack at
`config/fragments/signal/embedding/image-routing.yaml`. It contains three
illustrative rules: `identifier_document_imagery`,
`code_or_terminal_imagery`, and the negative-space anchor
`ambient_office_imagery`. The checked-in threshold is a starting point for the
bundled multimodal embedding model, not a portable default. See
[Embedding Anchor Design Principles](./embedding-design-principles) for its
calibration context, then replace the example anchors and recalibrate against
your own labeled corpus.

### Authoring tips for image anchors

- Anchors describe **visual signatures**, not text content of the image. "electronic health record screenshot showing patient demographics" works because clinical-record UIs have a recognizable visual signature; an anchor like "the words John Doe SSN 123-45-6789" would not, because the model embeds visual structure, not OCR.
- The default `aggregation_method: max` is usually appropriate for distinct sensitive-imagery categories: any anchor matching strongly is enough to fire the signal.
- Use several semantically distinct anchors per category, and add anchors for
  relevant negative space. Near-duplicate anchors add little coverage and can
  make a rule look better calibrated than it is.
- Do not gate on a single anchor; cosine similarity is noisy enough that one anchor and one image will produce false positives at scale. The anchor pack as a whole is the signal, not any individual phrase.

These tips generalize to text-modality packs as well. See [Embedding Anchor Design Principles](./embedding-design-principles) for the consolidated guidance.

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
- Maintained examples:
  [`support.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/support.yaml)
  and
  [`image-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml).
