---
title: Decision 1.0
sidebar_label: Decision 1.0
description: The seven Decision 1.0 models, three 0.6B encoders and four Qwen3.5 decoders, answer choice, noul and score questions about text, zero-shot.
---

# Decision 1.0

Decision 1.0 is the first family of decision models: three 0.6B encoders
(Kai, Lex and Route, on the Vela encoder backbone) and four decoders on Qwen3.5
(Eos-0.8B, Sol-2B, Nox-4B and Lux-9B). They answer your own `choice`, `noul`
and `score` questions about text, zero-shot.

```bash title="Start an encoder on a CPU"
vllm-sr serve vllm-sr/Decision-1.0-Kai-0.6B --engine --platform cpu
```

## Sizes

<!-- BEGIN GENERATED MODEL TABLE -->

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [Decision-1.0-Kai-0.6B](./kai-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Lex-0.6B](./lex-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Route-0.6B](./route-0-6b) | 0.572B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Eos-0.8B](./eos-0-8b) | 0.753B | choice, noul, score | — | text | 4 GiB |
| [Decision-1.0-Sol-2B](./sol-2b) | 1.88B | choice, noul, score | — | text | 8 GiB |
| [Decision-1.0-Nox-4B](./nox-4b) | 4.21B | choice, noul, score | — | text | 16 GiB |
| [Decision-1.0-Lux-9B](./lux-9b) | 7.94B | choice, noul, score | — | text | 24 GiB |

<!-- END GENERATED MODEL TABLE -->

:::tip
For new configurations, prefer [Decision 2.0](../decision-2/overview.md) or
[Decision 3.0](../decision-3/overview.md). Decision 1.0 stays built in, so
configurations that name it keep working.
:::

## How it runs

- **Byte-identical to the released runtime.** With each request alone, the
  family answers byte-identically to each package's bundled runtime, for all
  seven packages, on the Router's ROCm image
  ([parity record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision1-parity.md)).
- **Precision:** the encoders run in FP32 everywhere; the decoders in BF16 on a
  GPU and FP32 on a CPU.
- **Faster encoders on x86 CPUs:** Kai, Lex and Route run merged forwards on
  pre-packed FP32 copies of their linear layers with oneDNN, 1.5–1.7× faster
  with no decision changed.
- **Shared context is off** for the decoders: their records show decisions
  changing, so `--enable-shared-context` is an explicit choice
  ([Engine arguments](../../deploy/engine-args.md#shared-context)).
- **No spans.** Bind `pii` and `hallucination` to a span-capable deployment
  when a Decision 1.0 model is the decision model.
