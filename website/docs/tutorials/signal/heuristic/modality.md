# Modality Signal

## Overview

`modality` detects whether a request should stay in text generation, switch into image generation, or support both. It maps to `config/signal/modality/` and is declared under `routing.signals.modality`.

This family sits under `heuristic` because it usually routes from request form and configured modality behavior, even though deployments can later tune the detector through `global.model_catalog.modules.modality_detector`.

## Key Advantages

- Keeps image-generation routing separate from text-only routes.
- Makes multimodal traffic visible in `routing.decisions`.
- Avoids mixing modality checks into every decision rule.
- Scales from simple request-shape routing into hybrid detection.

## What Problem Does It Solve?

Text chat, image generation, and mixed workflows often share the same entrypoint but should not share the same model path. Without a modality signal, route logic becomes brittle and repetitive.

`modality` solves that by exposing request form as a named routing input.

## When to Use

Use `modality` when:

- the router serves both autoregressive and diffusion-style backends
- some routes should only accept image-generation prompts
- multimodal handling must stay explicit in the route graph
- you want a stable signal name such as `AR`, `DIFFUSION`, or `BOTH`

## Configuration

Source fragment family: `config/signal/modality/`

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

Keep the rule names aligned with the route behavior you want decisions to reference.
