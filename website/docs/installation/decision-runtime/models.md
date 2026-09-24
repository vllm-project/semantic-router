---
title: Models and backends
description: Choose a Decision 1.0 model, supported execution path, and immutable revision.
---

Use the exact catalog model ID in both `drun run` and every inference request.
The service does not infer a model from the endpoint or pick a default.

| Model ID (`llm-semantic-router/…`) | Runtime family | Profile input limit | Installed ROCm path |
| --- | --- | ---: | --- |
| `Decision-1.0-Kai-0.6B` | Vela | 1,024 tokens | `gfx942` |
| `Decision-1.0-Lex-0.6B` | Vela | 1,024 tokens | `gfx942` |
| `Decision-1.0-Eos-0.8B` | Qwen 3.5 | 16,384 tokens | `gfx942` |
| `Decision-1.0-Sol-2B` | Qwen 3.5 | 16,384 tokens | `gfx942` |
| `Decision-1.0-Nox-4B` | Qwen 3.5 | 16,384 tokens | `gfx942` |
| `Decision-1.0-Lux-9B` | Qwen 3.5 | 16,384 tokens | `gfx942` |

The installed CPU executor is limited to the sub-1B Kai, Lex, and Eos models.
That is an execution scope, not a claim that a CPU image, latency target, or
quality result is released. CUDA and Apple MLX still require separate runtime
and hardware qualification. A CLI backend option alone does not qualify a
device. Confirm a matching image and the model's status before production use.

The catalog supplies each model's default immutable revision. To use another
revision of the **same** model, pass its full 40-character commit SHA with
`--revision`. At launch, the host materializes and verifies that revision's
selected data files and manifest; the runtime verifies the mounted files again
before loading. The packaged profile records a manifest location, while each
selected snapshot supplies the manifest hash, file inventory, and sizes. A different
architecture, prompt contract, or hardware kernel still has to pass structural
and device checks.

All six model configurations start at physical batch size 8. On ROCm, a larger
`--max-batch` must fit the verified kernel-profile envelope; do not infer B16
or B32 support from a larger queue or model size. The optional Sol ROCm B8
graph path is [separate and off by default](./parameters.md#experimental-sol-graph).
