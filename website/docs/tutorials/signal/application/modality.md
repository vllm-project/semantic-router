# Modality Signal

## Overview

`modality` reads which output the request asks for: text, an image, text with an image, speech, audio or video.
It asks the `modality`
[preset](../../../system-one/reference#preset-catalog), "What kind of output
does this request ask for?", over eight options: `text`, `image`,
`text_image`, `image_edit`, `speech`, `audio`, `video` and `video_edit`.

## Key Advantages

- Keeps image-generation routing separate from text-only routes.
- Makes output intent visible in `recipes[].decisions`.
- Reads the written request, so a text-only "draw a cat" reaches an image
  route.

## What Problem Does It Solve?

Text chat, image generation and mixed requests often share one entrypoint but
need different backends. `modality` names the requested output, so a decision
can send each request to a model that can produce it.

## When to Use

Use `modality` when:

- one entrypoint serves text and image-generation backends;
- some routes should accept only requests for a given output;
- an edit request should reach a model that reads the medium it edits.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      modality:
        - name: wants_image
          labels: [image]
        - name: wants_text_image
          labels: [text_image]
    decisions:
      - name: image
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: modality, name: wants_image}
        output: {kind: image, action: generate}
        modelRefs:
          - model: image-model
```

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which `{type: modality, name: ...}` conditions use |
| `labels` | The outputs the rule matches; a rule without `labels` matches nothing itself, and its conditions name one with `label` |
| `predicate` | The least probability of the chosen output; by default the model's calibrated threshold (0.63 on vela3-fast), else 0.5 |
| `deployment`, `timeout_ms` | The deployment that answers, and the answer's deadline |

The signal only routes. A decision that answers with a generated output
declares it in its `output` contract, as `image` does above, and every model
it names must declare the capabilities that output needs (`image_generation`
on `image-model` here); [Output](../../decision/output) covers the contract.
A Responses API client then receives the generated image. An Images API
client (`POST /v1/images/generations`) is routed on its prompt and must reach
such a decision, or it is refused with `image_route_required` before any
model is called. Combine the edit labels, `image_edit` and `video_edit`, with
an [`input_modality`](../heuristic/input-modality) condition that confirms
the request carries the medium to edit; this release refuses edit outputs at
load with `output_not_executable`.

### The model that answers

The decision model answers every modality rule, unless the `modality`
binding or the rule's `deployment` names another; see
[Choose a deployment](../../../system-one/deployments). vela3-fast and the
Vela 3.0 d3 sizes were trained on three of the options, `text`, `image` and
`text_image` (asked in the trained form as `AR`, `DIFFUSION` and `BOTH`, and
mapped back); a question that needs another option is asked zero-shot. It
reads the current user turn, truncated to the model's input.

Configurations written for earlier releases named rules `AR`, `DIFFUSION`
and `BOTH` and let the name choose the output. `vllm-sr config migrate` gives
such rules their `labels`. See a complete example:
[`config/fragments/signal/modality/multimodal.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/modality/multimodal.yaml).
