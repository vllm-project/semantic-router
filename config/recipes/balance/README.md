# Balanced Routing Recipe Model Card

## Overview

Balanced Routing is a general-purpose policy for sharing traffic across local,
fast, specialist, and premium model lanes. It considers task difficulty,
domain, user feedback, verification needs, context, and conversation shape to
trade off quality, latency, and configured cost.

The recipe exposes `vllm-sr/auto`. It is intended as a starting policy that
operators calibrate against their own models and traffic.

## Model details

| Role | Reference alias |
| --- | --- |
| Local default and creative work | `qwen/qwen3.5-rocm` |
| Fast, lower-cost specialist | `google/gemini-2.5-flash-lite` |
| Complex specialist and verification | `google/gemini-3.1-pro` |
| Deep reasoning and formal math | `openai/gpt5.4` |
| Premium legal analysis | `anthropic/claude-opus-4.6` |
| Secondary high-care health reviewer | `anthropic/claude-opus-4.6` |

These are routing aliases, not required vendor choices. Connect the Models you
operate, then review each decision assignment before publishing the Entrypoint.

## Intended use

Use this recipe for mixed assistant traffic where:

- most questions should use an efficient local or fast lane;
- complex reasoning and specialist domains justify stronger models;
- corrections and repeated questions should trigger a recovery path;
- evidence-sensitive requests need additional verification; and
- one request-facing model name is preferred over several client-side tiers.

Choose a more specialized recipe when privacy, minimum latency, or minimum cost
must dominate every decision.

## Routing behavior

| Request pattern | Typical behavior |
| --- | --- |
| Legal, health, formal proof, or deep reasoning | Prefer premium or high-reasoning candidates. |
| Complex specialist or coding work | Use a capable specialist, with a stronger fallback where configured. |
| Explicit correction, repeated dissatisfaction, or verification request | Move to a recovery or reviewed-answer lane. |
| Medium explanation or creative work | Balance local and fast specialist candidates. |
| Short factual question or simple general request | Prefer the local or fast lane. |
| Casual conversation | Use the lowest-priority local fallback. |

Higher-risk and higher-effort routes run before general routes. Signal scores
coordinate semantic and deterministic evidence so a single keyword does not
normally decide the lane by itself.

## Requirements

- Reachable OpenAI-compatible endpoints for the configured aliases.
- Complexity, domain, embedding, feedback, fact-check, language, and re-ask
  signals enabled by [`config.yaml`](config.yaml).
- A response-cache store when cache behavior is desired.
- Calibrated cost and capability metadata for the actual deployment.

## Data handling and safety

The checked-in decisions opt into in-memory Router Replay. Replay captures up
to 4 KiB each from the request and response body for a routed request. The
records are process-local and non-durable, but they can still contain prompts,
answers, and other sensitive application data while the Router is running.

Review body capture, access, and retention before production use. Disable
replay or body capture when the deployment must not retain request content.

## Quick start

```bash
vllm-sr serve
```

In the Dashboard, connect the physical Models, choose **Balanced Routing** in
**Recipes**, assign a Model to each decision, and publish the resulting
Mixture-of-Model Entrypoint. The checked-in YAML and probes remain the
maintainer-facing source and conformance fixtures for this built-in Recipe.

## Evaluation

The probes cover each route family, negative cases, priority collisions,
feedback recovery, verification pressure, multilingual signals, and fallback
behavior. The current scenarios live in [`probes.yaml`](probes.yaml); the
[conformance guide](../CONFORMANCE.md) explains how to evaluate them.

## Limitations

- Quality, latency, and cost trade-offs depend on measured deployment data;
  the reference coefficients are not pricing or performance guarantees.
- Learned classifiers can misclassify unfamiliar domains or languages.
- A route to a stronger model does not guarantee correctness.
- The recipe does not provision the inference backends it references.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
