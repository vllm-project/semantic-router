---
title: Runtime architecture
description: Follow a Decision request from catalog resolution through row credits and physical batching.
---

Decision Runtime has a host-side launch boundary and a resident serving
boundary. The CLI resolves the exact catalog ID and revision, verifies and
materializes selected **data-only** model files into a content-addressed
artifact, and mounts that artifact read-only into a backend-matched image.
Model-repository Python is not executed in the new runtime. A missing release
image lock without an explicit image override, unsupported device, manifest
integrity mismatch, or incompatible profile fails before service becomes ready.

Inside the container, an owned Vela or Qwen 3.5 loader verifies the mounted
artifact again and keeps one model resident. The HTTP layer validates a
single-state or shared-question batch request. The scheduler reserves a
running slot and enough row credits for every state × question decision in
that request; a waiting request holds neither. The physical backend then
prepares complete model inputs, interleaves compatible rows from different
requests fairly, and sends at most `--max-batch` rows to one forward. It
restores each result to its original state and question before returning the
strict response envelope.

## Why the model paths differ

Kai and Lex use the Vela encoder path; Eos, Sol, Nox, and Lux use a Qwen 3.5
hybrid backbone. Both paths score a finite set of answer candidates instead of
generating an open-ended response. Each question becomes one prepared row,
so a request with many questions can share resident weights and physical
forwards without changing its question IDs or response order. The Qwen path
keeps its backbone in BF16 on GPU and its candidate head and final scoring in
FP32. The Vela and Qwen input encoders, heads, and device kernels are separate
modules behind the same row-executor interface.

On ROCm, a Qwen artifact that carries a normalization profile binds its
validated launch shapes; an unknown shape fails instead of silently
autotuning during serving.
The optional Sol graph path can replay a previously captured B8 backbone
shape, but it is not enabled by default. These are model-specific execution
choices, not changes to the public API.

This separates three limits that are easy to confuse:

| Boundary | Unit | Default or cap |
| --- | --- | --- |
| Scheduler concurrency | Admitted HTTP requests | `drun` default 8 |
| Waiting queue | Requests not yet admitted | `drun` default 32 |
| Active-row credits | Decisions reserved by admitted requests | Fixed cap 4,096 |
| Physical batch | Compatible rows per model forward | Profile default 8 |

The scheduler prevents a large multi-state request from consuming unbounded
work while smaller requests wait. The physical queue bounds outstanding rows
and does not silently raise the kernel-profile batch envelope. `/api/status`
reports the live scheduler limits and artifact identity; `/metrics` reports
request, preparation, physical-batch, and optional graph events. Successful
response headers bind that answer to the resident model revision and verified
artifact without changing its JSON contract.

For launch behavior, see [parameters](./parameters.md); for request shapes,
see [the API guide](./api.md).
