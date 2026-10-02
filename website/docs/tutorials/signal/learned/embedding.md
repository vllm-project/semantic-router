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

The family-level `prototype_scoring` settings compress each text rule's candidate
bank and control its scoring. A rule can declare its own `prototype_scoring`
object beside `candidates`. Text rules inherit the family settings when it is
omitted. Image and audio rules instead retain every candidate and use the raw
maximum similarity by default: nearby text anchors can match different media.
Declaring it replaces the complete object, with omitted fields using built-in defaults.
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
In this explicit override, `max` combines the best similarity and top-M support;
`mean` still averages
the retained bank. With compression enabled, `max_prototypes: 0` uses the default
cap of 8. Retaining more candidates adds local scoring work; candidate embeddings
are already computed before compression and request embedding calls are unchanged.
Rule settings travel with an exported or initialized recipe and apply to both
text, image, and audio queries.

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
- Use `aggregation_method: max` to favor the strongest example. Text rules
  also consider support from other prototypes by default; image and audio
  rules use the strongest candidate similarity without compression. An explicit
  prototype blend applies the threshold to its combined score. Use
  `mean` only when broad agreement across the candidate set is the behavior
  you want.
- Calibrate `threshold` against labeled positive and negative traffic for the
  deployed model. Thresholds do not transfer reliably between models,
  dimensions, modalities, or traffic distributions.

Re-run the labeled evaluation whenever you change a candidate, model, or
threshold. Record those three inputs together so a configuration update does
not silently reuse an incompatible threshold.

## Multimodal queries (`query_modality`)

Each embedding rule accepts an optional `query_modality` field that declares which modality of incoming request payload the rule's query is computed from. Text and image candidates are encoded by the same prepared model in its shared embedding space. Candidate modality is independent of query modality.

Accepted values:

- `"text"` (default, backward-compatible): query embedded from request text. Existing rules with no `query_modality` field behave exactly as before.
- `"image"`: query embedded from an allowlisted inline
  `data:image/...;base64,...` attachment in an OpenAI-style chat message.
- `"audio"`: query embedded from bounded inline PCM or float WAV data. Remote audio URLs, MP3, and other compressed formats are not accepted.

Image/audio queries and image candidates require
`global.model_catalog.embeddings.semantic.embedding_config.model_type: multimodal`
or an explicit recipe embedding binding. The Router checks the prepared model's
actual modality capabilities at startup; a name or catalog entry does not prove
that an image or audio encoder is available.

### Positive and negative candidate banks

| Field | Encoded as | Role |
| --- | --- | --- |
| `candidates` | Text | Positive examples |
| `image_candidates` | Images | Positive examples |
| `negative_candidates` | Text | Contrasting examples |
| `negative_image_candidates` | Images | Contrasting examples |

At least one positive candidate is required. Image references are absolute or
`./` local paths, inline base64, or image data URIs; candidate loading never
fetches remote URLs. Every bank uses the rule's aggregation policy. With `max`,
the score is `max(cosine(query, positive))`; when negatives are present, it is
`max(cosine(query, positive)) - max(cosine(query, negative))`. Text and image
examples in one bank compete in the same maximum. A rule matches when its raw
score is greater than or equal to `threshold`. Thresholds use cosine units
`[-1, 1]`, or margin units `[-2, 2]` when a negative bank is present.

Image/audio queries and rules with image candidates retain every authored
example by default. Explicit `prototype_scoring` can enable clustering. Positive
and negative banks are built independently, and the same aggregation is applied
to both. `mean` means the difference of the two bank means, not the mean of a
combined positive/negative list.

`SignalValues["embedding:<name>"]` retains the raw score; `:positive` and
`:negative` expose its components. Contrastive signal confidence is
`clamp((score + 2) / 4, 0, 1)` for consumers requiring a bounded value. It describes
position in the margin range, **not a probability**. Thresholds and raw-value
projections continue to use the unnormalized margin. Optional soft matching also
compares `min_score_threshold` to the raw score; choose it in the same units.

### Worked example: route sensitive imagery on-prem

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        multimodal_model_path: models/vela-1.0-omni-nano
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

The maintained pack uses `llm-semantic-router/Vela-1.0-Omni-Nano`, snapshot
`2ff2d66385dbdd661a560ec3e8bcb45a0527d92e`, with the complete 384-dimensional
output. The code rule uses seven positive image prototypes and 178 negative
image prototypes. All prototypes come from the frozen development split of the
229-image reviewed corpus. Their original bytes and SHA256 identities are listed
in `config/assets/image-routing/manifest.json`. Runtime images package these
files under `/app/share/image-routing`; no runtime download is required.

The threshold `0.022578716` is selected on development data while excluding every
prototype from each query's source group, including the query itself. The six
validation positives and 30 validation negatives never enter the banks or
threshold selection. The fixed results are:

| Model and bank | Development TP / FP / FN | Validation TP / FP / FN |
| --- | --- | --- |
| Nano, all development prototypes | 7 / 2 / 0 | 6 / 0 / 0 |
| Mini, all development prototypes, its own threshold `0.028753608` | 5 / 2 / 2 | 6 / 1 / 0 |

Eight development files sharing bytes with seven validation files are excluded
from both candidate banks and threshold fitting. All 36 validation items remain
in the report. Identical-content label groups were consistent; no labels were
changed. Development scoring excludes both source-group and content matches.
The pack retains the complete remaining negative bank. Mini's remaining false positive is
an Ollama setup-wizard screenshot; it is retained in the report. The corpus's
original text-anchor baseline had already been examined before this split was
frozen, so these are grouped diagnostic results, not an unseen external benchmark.

The identifier rule retains its text anchors and threshold `0.29`. The office
rule subtracts a contrasting text bank at threshold `0.054949798`. Each has only
one labeled positive in this corpus; both separate the reviewed images, but
neither has an independent positive holdout. These examples require local data
and calibration before deployment as a document or office detector.

Run `tools/calibration/image-routing` with an explicit prepared Omni artifact to
reproduce the production scorer, source-group exclusions, held-out results, and
artifact checksums. `testdata/prototype-protocol.json` records the frozen split.
Changing any model, prototype, image bytes, dimension, or scoring policy requires
new calibration; Nano thresholds must not be reused for Mini.

### Distinction from the `modality` signal type

`query_modality` (this section) declares **input modality** for an embedding rule — which modality of payload the query is computed from. The separate [`modality`](modality) signal type declares **output modality** (`AR`, `DIFFUSION`, `BOTH`) for routing image-generation requests. The two concepts share a name but solve different problems and live on different config surfaces.

## Dependencies and Limitations

- Text, image, or audio content is processed by the configured embedding runtime. A
  remote provider currently supports text only and receives the text being
  embedded. Local Vela Omni deployments support all three input modalities.
- Similarity scores and thresholds are not portable across embedding models,
  dimensions, or modalities. Recalibrate whenever those change.
- Image matching is semantic rather than OCR or PII extraction; use a dedicated
  detector when literal text or regulated entities matter.
- Complete examples:
  [`support.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/support.yaml)
  and
  [`image-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml).
