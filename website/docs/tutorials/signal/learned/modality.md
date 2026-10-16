# Modality Signal

## Overview

`modality` detects whether a request should stay in text generation, switch into
image generation, or support both. Define its output labels under
`routing.signals.modality`.

The configured `modality_detector` classifies the intended output mode; the
named result is then available to ordinary decisions.

## Key Advantages

- Keeps image-generation routing separate from text-only routes.
- Makes multimodal traffic visible in `routing.decisions`.
- Avoids mixing modality checks into every decision rule.
- Scales from simple request-shape routing into detector-backed multimodal classification.

## What Problem Does It Solve?

Text chat, image generation, and mixed workflows often share the same entrypoint but should not share the same model path. Without a modality signal, route logic becomes brittle and repetitive.

`modality` solves that by exposing output mode as a named routing input.

## When to Use

Use `modality` when:

- the router serves both autoregressive and diffusion-style backends
- some routes should only accept image-generation prompts
- multimodal handling must stay explicit in the route graph
- you want a stable signal name such as `AR`, `DIFFUSION`, or `BOTH`

## Configuration

```yaml
routing:
  signals:
    modality:
      - name: AR
        description: Text-only autoregressive requests.
      - name: DIFFUSION
        description: Requests that should route into image generation flows.
      - name: BOTH
        description: Requests that need both text and image generation behavior.
```

Keep the rule names aligned with the route behavior you want decisions to
reference.

The detector is off by default, so these rules match only with a
`modality_detector` block. With `method: classifier` and no
`classifier.model_path`, it runs Vela 2.0 0.3B, whose calibrated
`confidence_threshold` is 0.51:

```yaml
global:
  model_catalog:
    modules:
      modality_detector:
        enabled: true
        method: classifier
        confidence_threshold: 0.51
```

Without the block, the Router logs `modality_detector_disabled` when it loads
the config, and `vllm-sr config validate` prints the same warning from the
Router's own validation. `method:
keyword` needs `keywords`; `method: hybrid` also needs `lower_threshold_ratio`
and a classifier or keywords.

A decision on `BOTH` must reference both an AR model (`modality: ar` in
`routing.modelCards`) and a diffusion model (`modality: diffusion`), or an omni
model (`modality: omni`).

## Dependencies and Limitations

Without a recipe model binding, `classifier.max_sequence_length: 0` keeps the
512-token default. Long prompts receive representative head, middle, and tail
samples for routing, and the model runtime truncates any input still over the
budget. This policy limits inference cost and does not classify every token.

For unsampled modality input, bind `modality_detector` to a named deployment with
`input.max_tokens` above 512. Set `input.overflow: reject` to reject input beyond
that budget. Choose a budget qualified for the selected artifact and provider;
a model's advertised 32K capacity does not establish 32K execution on every
provider. A positive module `classifier.max_sequence_length` above 512 also
bypasses routing sampling, but retains the module's truncation policy.

With `method: classifier`, an actual inference error leaves modality unknown,
and the decision's `on_unknown` policy controls the outcome. It does not create
an `AR` match. `method: hybrid` explicitly permits keyword fallback.

The modality detector classifies intended output mode; it does not prove that a
backend supports the request's input attachments. Keep model-card capabilities
and provider validation aligned. See a complete example:
[`config/fragments/signal/modality/multimodal.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/modality/multimodal.yaml).
