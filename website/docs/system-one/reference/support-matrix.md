---
title: Support matrix and thresholds
sidebar_label: Support matrix
sidebar_position: 2
description: Which built-in System One questions each model family answers trained, zero-shot or not at all, and the calibrated thresholds a rule takes by default.
---

# Support matrix and thresholds

Every model answers every preset it can in its canonical, zero-shot form. Some
families were trained on a preset, and some have a calibrated threshold for it.
This page shows both, so you can tell what changes when you switch the decision
model.

## Support matrix {#support-matrix}

**TC** trained and calibrated, **T** trained, **B** the base model with every
adapter off, **Z** zero-shot, **—** cannot run.

| Preset | vela3-fast | vela3.1-pico-380m, -150m | vela3-lite | vela3-mini, -flash, -turbo | d3 (local or hosted) | Decision 2.0 | Vela 2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| domain, fact_check, feedback | TC | T | T | T | Z | Z | TC |
| jailbreak | TC | T | B | B | Z | Z | TC |
| harm | TC | T | B at the request stage; T at the response stage | B at the request stage; T at the response stage | Z | Z | TC |
| harm categories | T | T | T at the request stage; Z at the response stage | Z | Z | Z | Z |
| pii | TC | T | T | T | — | — | TC |
| hallucination | T | T | T | T | — | — | T |
| refusal | T | T | T | T | Z | Z | Z |
| modality | TC for `text`, `image`, `text_image`; Z for the others | T for the three; Z | T for the three; Z | T for the three; Z | Z | Z | Z |
| turn_relation, complexity, relevance, same_request, tool_rank, tool_need, answer_check | Z | Z | Z | Z | Z | Z | Z |

- The Vela 3.0 d3 sizes turn trained adapters on by question ID: `domain`,
  `factcheck`, `feedback`, `r_harm`, `r_refusal`, `modality`, and, on
  vela3-lite only, the `hazard:` prefix of the harm categories; vela3-fast, the
  Vela 3.1 picos and Vela 2.0 by wording.
- The Vela 3.0 d3 sizes answer the jailbreak preset and the request-stage harm
  preset with their base, every adapter off. Their `attack` and `p_harm`
  adapters lose an attack that other text surrounds (on vela3-turbo, 0.975
  against the base's 0.9992 for an injection after 8K tokens of prose) and rate
  benign text higher, and guard thresholds such as 0.995 were set on the base's
  answers.
- Image input is read by the d3 family and the Vela 3.0 d3 sizes. No current
  decision model reads audio or video for routing rules.

## Calibrated thresholds

The default `predicate` of a rule that writes none, on the models measured on
the router signal suite:

| Model | jailbreak | domain | pii | fact_check | feedback | harm | modality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vllm-sr/vela3-fast` | 0.88 | 0.41 | 0.18 | 0.9999 | 0.47 | 0.52 | 0.63 |
| `vllm-sr/Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 | 0.46 | 0.51 |
| `vllm-sr/Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 | | |
| `vllm-sr/Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 | | |
| `vllm-sr/Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 | | |

:::note
Every other model uses 0.5, with an `uncalibrated_preset` warning at load. The
`hallucination` preset's spans keep the model's own thresholds. The records
behind these values are in the
[model runtime records](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records),
and each model's page under
[Supported models](../../model-runtime/supported-models/overview.md) lists its
own.
:::
