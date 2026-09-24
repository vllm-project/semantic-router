---
title: Runtime architecture
description: Follow a Decision request from catalog resolution through row credits and physical batching.
---

Decision Runtime has a host-side launch boundary and a resident serving
boundary. The CLI resolves the exact catalog ID and revision, verifies and
materializes selected **data-only** model files into a content-addressed
artifact, and mounts that artifact read-only into a backend-matched image.
Model-repository Python is not executed in the new runtime. A missing release
image lock without an explicit image override, unsupported device, changed
manifest, or incompatible profile fails before model service becomes ready.

Inside the container, an owned Vela or Qwen 3.5 loader verifies the mounted
artifact again and keeps one model resident. The HTTP layer validates a
single-state or shared-question batch request. The scheduler reserves a
running slot and enough row credits for every state × question decision in
that request; a waiting request holds neither. The physical backend then
prepares complete model inputs, interleaves compatible rows from different
requests fairly, and sends at most `--max-batch` rows to one forward. It
restores each result to its original state and question before returning the
strict response envelope.

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
