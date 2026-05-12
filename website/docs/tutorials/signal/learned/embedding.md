# Embedding Signal

## Overview

`embedding` matches requests by semantic similarity to representative examples. It maps to `config/signal/embedding/` and is declared under `routing.signals.embeddings`.

This family is learned: it depends on the semantic embedding assets in `global.model_catalog.embeddings`.

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

Source fragment family: `config/signal/embedding/`

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

The router now scores every embedding rule first and only then applies `top_k` as an emission limit. The default is `1`, so only the strongest embedding signal is returned unless you explicitly raise the limit. Set `top_k: 0` if you need the legacy "return every matched embedding rule" behavior.

## Multimodal queries (`query_modality`)

Each embedding rule accepts an optional `query_modality` field that declares which modality of incoming request payload the rule's query is computed from. The candidates remain text in every case; the rule cosine-matches the text-anchor set against a query embedding from the declared modality, all in the same shared multimodal space.

Accepted values:

- `"text"` (default, backward-compatible): query embedded from request text. Existing rules with no `query_modality` field behave exactly as before.
- `"image"`: query embedded from an image attachment (base64, data-URI, or local file path) on the incoming request.
- `"audio"`: query embedded from an audio attachment. Schema-accepted but rejected at config-load until the candle-binding `MultiModalEncodeAudioFromBase64` FFI is exposed; remove the rule or use `text`/`image` until that lands.

`"image"` and `"audio"` require `global.model_catalog.embeddings.semantic.embedding_config.model_type: multimodal` so the candidates and queries are embedded into the same shared space. The router fails configuration validation at load time if an image- or audio-modality rule is paired with a text-only embedding model.

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
      # Existing text-modality rule - no migration needed; query_modality
      # defaults to "text".
      - name: technical_support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - how to configure the system
          - installation guide
          - troubleshooting steps

      # New image-modality rule - declares that this rule's query is computed
      # from an image attachment. The classifier API accepts image payloads
      # via ClassifyDetailedMultimodal and cosine-matches them against the
      # text anchors below in the shared multimodal embedding space. The
      # request-path extractor that pulls images out of OpenAI-shaped chat
      # completion content arrays and feeds them into this rule lands in a
      # follow-up PR; on this PR alone the API surface is in place but the
      # running router does not yet read attachments off chat completions.
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
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: medical_imagery_phi
      modelRefs:
        - model: in-cluster-vlm
```

### Authoring tips for image anchors

- Anchors describe **visual signatures**, not text content of the image. "electronic health record screenshot showing patient demographics" works because clinical-record UIs have a recognizable visual signature; an anchor like "the words John Doe SSN 123-45-6789" would not, because the model embeds visual structure, not OCR.
- The default `aggregation_method: max` is usually appropriate for distinct sensitive-imagery categories: any anchor matching strongly is enough to fire the signal.
- Authoring 8-15 anchors per category gives a robust signal without overfitting. Add anchors for the negative space too (a separate "ambient_office_imagery" rule with anchors for whiteboards, conference rooms, generic infographics) so a low-confidence image stays low-confidence rather than landing closer to a sensitive anchor by accident.
- Do not gate on a single anchor; cosine similarity is noisy enough that one anchor and one image will produce false positives at scale. The anchor pack as a whole is the signal, not any individual phrase.

### Distinction from the `modality` signal type

`query_modality` (this section) declares **input modality** for an embedding rule — which modality of payload the query is computed from. The separate [`modality`](modality) signal type declares **output modality** (`AR`, `DIFFUSION`, `BOTH`) for routing image-generation requests. The two concepts share a name but solve different problems and live on different config surfaces.
