# Input Modality Signal

## Overview

`input_modality` deterministically matches which kinds of input — `text`,
`image`, `audio`, or `video` — are present in the parsed request. Detection is
purely structural: the router inspects content-part types in user messages and
never runs a classifier, embedding model, or any other ML inference.

Input modality is independent of the *output* the request asks for, which
the [`modality`](../application/modality.md) signal reads. A request may
contain several input modalities at once, and each configured rule matches
independently.

## Key Advantages

- routes any request containing an image (or audio/video) to a capable model
  with zero inference cost
- exposes the full modality set of a request for composition in AND/OR
  decision rules, projections, and traces
- works across Chat Completions, Anthropic Messages, Response API, and the
  classify/eval HTTP APIs (see Detection Rules for per-protocol coverage)

## What Problem Does It Solve?

A common topology sends any request containing an image to a vision-capable
model while text-only requests continue through normal semantic routing.
`input_modality` names what a request carries directly and covers all four
modalities.

Note that `NOT image_input` is not the same as "text-only": a request with no
image may still be audio-only or empty. Combine the `text` modality rule with
negations of the media rules to express text-only precisely.

## When to Use

Use `input_modality` when the routing decision depends on what kinds of input
are present — not on what the media contains. To route on what an image or a
video shows, ask a [`systemone`](../systemone#read-more-than-the-current-turn)
question with `input.image` or `input.video` instead; its deployment must read
that medium, or the configuration is refused with
`input_modality_unsupported`.

## Configuration

```yaml
recipes:
  - name: default
    signals:
      input_modality:
        - name: image_input
          description: Request contains at least one image content part.
          modality: image
        - name: audio_input
          description: Request contains at least one audio content part.
          modality: audio
    decisions:
      - name: vision-request
        description: Send image-bearing requests to the vision pool.
        priority: 1000
        rules:
          operator: AND
          conditions:
            - type: input_modality
              name: image_input
        modelRefs:
          - model: vision-model
```

`modality` must be one of `text`, `image`, `audio`, or `video`. Rule names
must be unique and trimmed.

## Detection Rules

Counting covers user messages only, so system prompts and assistant history
never satisfy a rule. Every ingress protocol is decoded once into the router's
neutral request, and the counts come from its content kinds, so a rule matches
the same way whichever protocol carried the request:

- **Chat Completions**: plain string content or `text` parts count as text,
  `image_url` parts as image, and `input_audio` parts as audio.
- **Anthropic Messages**: `text` and `image` blocks; the protocol has no audio
  or video block types.
- **Response API**: `input_text` counts as text and `input_image` counts as
  image. The Response API decoder accepts no audio content type today.
- **Classify/eval APIs**: the same part types on `messages[].content`, plus
  `input_audio` / `audio_url` for audio and `video_url` / `input_video` for
  video.

`video` is accepted in configuration and the walker recognizes neutral video
content, but no ingress protocol decoder currently accepts a video content
part, so on the data plane a `video` rule cannot match yet. It can match through
the classify/eval APIs.

## Observability

Matched rules are exposed as `input_modality:<name>` in the `signals` member
of `x-vsr-trace`, the `input_modality` field of classify/eval responses, router-replay
records, and the standard per-signal Prometheus metrics
(`llm_signal_match_total{signal_type="input_modality"}`).

## Related Signals

- [`modality`](../application/modality.md) — the output the request asks
  for: text, an image, text with an image, speech, audio or video.
- [`conversation`](conversation.md) — request-shape facts such as message
  counts, tool definitions, and tool loops.
