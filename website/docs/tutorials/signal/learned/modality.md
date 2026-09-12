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
reference. Configure the detector through
`global.model_catalog.modules.modality_detector`.

## Dependencies and Limitations

Set `classifier.max_sequence_length` inside the detector configuration to select
an explicit token budget; zero keeps the historical 512-token default. The
native loader checks that the model supports the budget. Input beyond that
budget produces an error instead of a truncated prediction. With
`method: classifier`, an inference error leaves modality unknown, and a decision's
`on_unknown` policy controls the outcome. It does not create an `AR` match.
`method: hybrid` explicitly permits the existing keyword fallback.

The modality detector classifies intended output mode; it does not prove that a
backend supports the request's input attachments. Keep model-card capabilities
and provider validation aligned. See a complete example:
[`config/fragments/signal/modality/multimodal.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/modality/multimodal.yaml).
