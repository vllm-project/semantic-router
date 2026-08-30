# Image Generation

## Overview

`image_gen` is a route-local plugin for handing a matched route off to an image-generation backend.

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

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: image_gen
    configuration:
      enabled: true
      backend: vllm_omni
      backend_config:
        base_url: http://image-router:8005
        model: Qwen/Qwen-Image
        num_inference_steps: 28
        cfg_scale: 4.5
```

The selected backend receives the image prompt and generation parameters. Use
an authenticated, trusted endpoint and apply request-side safety policy before
this plugin. `max_response_bytes` caps each OpenAI or vLLM-Omni response body;
omitted or `0` uses 64 MiB. See a complete example:
[`config/fragments/plugin/image-gen/basic.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/image-gen/basic.yaml).
