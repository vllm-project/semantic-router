# Release note: the built-in signals default to Vela 2.0 0.3B

Vela 2.0 is public on Hugging Face:
[`vllm-sr/Vela-2.0-0.3B`, `-0.8B`, `-4B` and `-9B`](https://huggingface.co/collections/vllm-sr/vela-20),
Apache-2.0, with no token needed. The model runtime keeps the revisions it
pins, so nothing it loads changes.

This note records the initial Vela 2.0 default rollout and its measured A/B.
The measurements and original threshold decisions below are historical. Since
then, [decision-model selection](./decision-model.md) and the
[model runtime](./built-in-model-runtime.md) added shared task bindings, complete
safety scans and stage-aware bundles. Follow [Choose a model](../model-runtime/choose-a-model)
for current configuration. In particular, supplying a model does not imply
Engine mode, one deployment does not imply one forward, and the operator now
lets unbound tasks follow the selected default model.

## Initial rollout defaults

With no model configured, the domain, prompt guard, safety, fact check, user
feedback, modality, PII and hallucination signals run on one deployment of
`vllm-sr/Vela-2.0-0.3B`, `@Vela-2.0-0.3B`, in one call per request. They no
longer run on eight Vela 1.0 specialists.

- **Questions:** each signal asks the question the model was trained on for it,
  with the labels of its Vela 1.0 model, so rules and policies read the answer
  as before.
- **CPU profile:** on a CPU the deployment runs `max_speed`, which gives the
  same answers to within about 0.00001.
- **Input:** the model reads up to 8,192 tokens of a request and truncates the
  rest. The Vela 1.0 Guard and PII specialists scanned up to 32K in windows.
- **Modality:** a `modality_detector` with `method: classifier` and no
  `classifier.model_path` now runs the 0.3B. It used to be a configuration
  error.
- **What stays:** Hazard, embeddings, multimodal embeddings and the reranker
  keep their Vela 1.0 models.
- **Operator:** the operator's `prompt_guard.model_id` defaults to the 0.3B,
  and its `threshold` to 0.75. A `SemanticRouter` created earlier keeps the
  values it stored.

## The maintainers chose this default against two of its goals

[#4639](https://github.com/vllm-project/semantic-router/issues/4639) asked for
accuracy level or better on every signal and router latency level or better,
both measured on a CPU. The maintainers chose to switch although both fail,
accepting the modality and user feedback regressions. Through the Router, on the
[router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
([A/B record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-router-signals.md)):

- **Ahead:** prompt guard (held-out AUC +0.026) and safety (+0.052).
- **Level:** PII and hallucination on held-out and fresh files.
- **Behind, most:** modality (held-out AUC −0.180; the 0.3B misses most
  requests for a new image) and user feedback (accuracy −0.038 held-out,
  −0.178 fresh).
- **Behind:** domain (accuracy −0.037 held-out) and fact check (held-out AUC
  −0.101).
- **CPU latency:** on 12 cores a request takes 79 ms at the median against
  16 ms on Vela 1.0, about 4.9 times as long, and the Router serves 11.9
  against 38.9 requests per second (12.8 against 51.8 at concurrency 16).
  Every request carries the questions, their options and the 17 PII labels (at
  least 560 tokens) through one forward.
  [#4668](https://github.com/vllm-project/semantic-router/issues/4668) works on
  the CPU latency. On a GPU the 0.3B answers in about 7 ms.

## Thresholds

The module defaults are recalibrated to the 0.3B's scores so that each keeps
the Vela 1.0 specialist's operating point on the suite's dev split: prompt guard
0.5 → 0.75, domain 0.5 → 0.28, PII 0.9 → 0.01, fact check 0.95 → 0.93, user
feedback 0.7 → 0.37.

- **PII:** the model's span head applies its own per-label thresholds before it
  returns a span, so 0.01 accepts every span it returns.
- **Other models:** a module that runs any other model and sets no threshold
  keeps its earlier default.
- **Maintained configurations:** the recipes and E2E profiles that run the
  defaults carry the mapped values of their rule thresholds.
- **Your own rule thresholds** were likely chosen for Vela 1.0: map them with
  the record's table (prompt guard 0.3–0.9 → 0.74–0.77, PII → 0.01, safety
  0.5 → 0.46, modality `confidence_threshold` 0.7 → 0.51), or restore Vela 1.0.

## Restore Vela 1.0

One block brings the specialists back, with their default thresholds:

```yaml
global:
  model_catalog:
    system:
      safety: models/Vela-1.0-Encoder-307M-Safety
      prompt_guard: models/Vela-1.0-Encoder-307M-Guard
      domain_classifier: models/Vela-1.0-Encoder-307M-Domain
      pii_classifier: models/Vela-1.0-Encoder-307M-PII
      fact_check_classifier: models/Vela-1.0-Encoder-307M-FactCheck
      hallucination_detector: models/Vela-1.0-Encoder-307M-Halu
      feedback_detector: models/Vela-1.0-Encoder-307M-Feedback
```

One line brings back one signal, and the others stay on the 0.3B:

| Signal | Line under `global.model_catalog` |
| --- | --- |
| Domain | `system.domain_classifier: models/Vela-1.0-Encoder-307M-Domain` |
| Prompt guard | `system.prompt_guard: models/Vela-1.0-Encoder-307M-Guard` |
| Safety | `system.safety: models/Vela-1.0-Encoder-307M-Safety` |
| Fact check | `system.fact_check_classifier: models/Vela-1.0-Encoder-307M-FactCheck` |
| User feedback | `system.feedback_detector: models/Vela-1.0-Encoder-307M-Feedback` |
| PII | `system.pii_classifier: models/Vela-1.0-Encoder-307M-PII` |
| Hallucination | `system.hallucination_detector: models/Vela-1.0-Encoder-307M-Halu` |
| Modality | `modules.modality_detector.classifier.model_path: models/Vela-1.0-Encoder-307M-Modality` |

A signal moved back takes its Vela 1.0 rule thresholds with it, for example
`mom-v1`'s prompt guard 0.5, safety 0.5 and PII 0.7. See
[Choose a model](../model-runtime/choose-a-model#vela-20).
