# Image Generation

## Overview

`image_gen` is a route-local plugin for handing a matched route off to an image-generation backend.

It aligns to `config/plugin/image-gen/basic.yaml`.

## Key Advantages

- Keeps multimodal or image generation behavior local to the route.
- Exposes backend details clearly in config.
- Lets one router host text and image routes without mixing the behaviors.

## What Problem Does It Solve?

Some routes should not follow the standard chat-completions flow. `image_gen` makes that image-generation handoff explicit for routes that need it.

## When to Use

- a matched route should call an image-generation backend
- the route needs backend-specific generation settings
- text-only routes should remain unaffected

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: image_gen
  configuration:
    enabled: true
    backend: vllm_omni
    backend_config:
      base_url: http://image-router:8005
      model: Qwen/Qwen-Image
      num_inference_steps: 28
      cfg_scale: 4.5
```
