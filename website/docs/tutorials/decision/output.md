# Output

## Overview

A decision's `output` attribute declares the media its answers carry. A
decision without one answers in text. A decision with
`output: {kind: image, action: generate}` answers with a generated image: when a
request asks for nothing but text, such as "draw a lighthouse at dusk", the
Router adds the hosted `image_generation` operation to it before it selects a
model, so the request only reaches a model that can generate the image.

Output is a decision attribute, not a signal, algorithm, or plugin. It replaces
the output that a model's `modality` keyword (`ar`, `diffusion`, `omni`) and the
`image_gen` plugin used to imply; a configuration that still writes either fails
to load, and `vllm-sr config migrate` rewrites what it can.

## What Problem Does It Solve?

Routing a "draw a cat" request to an image model is not enough: a text-only
request has nothing an image API can encode, so a route that only names the
image model selects a backend that then fails. The output contract makes the
answer's media part of the decision, checks at load that every model it
references can produce it, and gives the request the operation that produces
it.

## When to Use

- A decision answers with a generated image: `output: {kind: image}`.
- One model answers in text and generates an image in the same response:
  `output: {kind: [text, image]}`.

Leave it out for text answers, which every decision gives by default. Routing on
what a request's own images show is a signal, not an output contract (see
[Image Input](#image-input)).

## Configuration

```yaml
providers:
  models:
    - name: painter
      api_format: images
      capabilities: [image_generation]
      modalities: {input: [text], output: [image]}
      backend_refs:
        - {name: omni-server, provider: vllm, endpoint: vllm-omni:8000}
    - name: omni
      api_format: responses
      capabilities: [chat, image_generation]
      modalities: {input: [text, image], output: [text, image]}
      backend_refs:
        - {name: omni-model, provider: vllm, endpoint: omni:8000}
recipes:
  - name: default
    signals:
      modality:
        - {name: wants_image, labels: [image]}
        - {name: wants_text_image, labels: [text_image]}
    decisions:
      - name: image
        rules: {operator: AND, conditions: [{type: modality, name: wants_image}]}
        output: {kind: image, action: generate}
        modelRefs: [{model: painter}]
      - name: text-and-image
        rules: {operator: AND, conditions: [{type: modality, name: wants_text_image}]}
        output: {kind: [text, image], action: generate}
        modelRefs: [{model: omni}]
```

The equivalent DSL is:

```dsl
ROUTE image {
  WHEN modality("wants_image")
  MODEL "painter"
  OUTPUT {
    kind: "image"
    action: "generate"
  }
}
```

`kind` is `text` (the default), `image`, `audio`, or `video`, or a list of them
that one model writes together. `action` is `generate` (the default) or `edit`,
which edits the image or video the request provides.

## Load Checks

Every model a decision references must declare what its output needs, in its
`providers.models[]` entry:

| Output | The model declares |
|---|---|
| text (the default) | text in `modalities.output`, which every model does unless its modalities say otherwise |
| `image`, `generate` | `image_generation` in `capabilities` |
| `image`, `edit` | `image_generation` and `image_input` |
| `audio` | `audio_output` |
| `video` | `video_output`, and `video_input` to edit |
| a list | everything each kind needs, in one model |

A model that lacks one fails the load with `output_capability_missing`, naming
the model and the fact to declare. This release executes text, image
generation, and text with a generated image from one model; image edits, audio,
and video fail with `output_not_executable`, naming the codec they need, until
that codec lands. A generated image also needs an API that carries it: a model
on the images API (`api_format: images`) returns the image, and a model on the
Responses API (`api_format: responses`) returns the image with text. An
executor algorithm (such as `escalate`) answers in text only, so it cannot serve
an image.

## Runtime Effect

- The hosted `image_generation` operation joins the request's capabilities, so
  the candidate filter skips a model whose API or declared capabilities cannot
  carry it, and `candidate_requirements` sees it too.
- An `image` decision forces the operation; a `[text, image]` decision offers
  it and leaves the call to the model.
- A Responses API client receives an `image_generation_call` item with the
  image, and text in a message item when the model writes it.
- An [Images API](../../api/router#images-api) client
  (`POST /v1/images/generations`) receives the image in that API's shape. Its
  request is routed on the prompt and must reach a decision whose output is an
  image; otherwise it gets `image_route_required` before any model is called.
- A Chat Completions or Anthropic Messages client cannot receive an image: the
  Router refuses its request with `unsupported_capability` before it calls a
  model. A request whose `tool_choice` is `none` is refused too.
- The `modality` member of `x-vsr-trace` names the output kinds of the
  selected decision on a debug response.

## Image Input

Routing on what a request's images show is a `systemone` question with
`input.image: true`, answered by a decision model that reads images. See
[System One](../signal/learned/systemone) for the question fields and the
models that answer image questions.
