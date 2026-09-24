---
title: Launch parameters
description: Understand Decision Runtime image, capacity, device, and lifecycle controls.
---

`vllm-sr drun run MODEL` resolves one catalog model and one immutable artifact
before starting a managed container. These are the most consequential options;
the [generated command reference](../../api/cli.md#vllm-sr-drun-run) is the
complete list.

| Option | Behavior |
| --- | --- |
| `--backend` | `auto`, `rocm`, `cuda`, or `cpu`. Select explicitly when device discovery is ambiguous; a choice still needs an installed executor and image. |
| `--revision` | Full commit SHA of the selected catalog model; defaults to its catalog revision. |
| `--image`, `--image-pull-policy` | Released defaults require a packaged digest inventory. An exact local Docker `sha256:` image ID requires `--image-pull-policy never`; a published `repository@sha256:` reference follows the selected pull policy. |
| `--max-batch` | Maximum compatible **physical rows** per model forward; profile default 8. It is not the number of HTTP requests. |
| `--max-concurrency` | Simultaneously admitted requests; `drun` default 8. |
| `--max-queue` | Waiting requests; `drun` default 32. Zero rejects rather than queues when admission is full. |
| `--cpu-threads` | CPU-only Torch/BLAS thread limit; default is the smaller of 8 and the container CPU allowance. |
| `--gpu-device` | ROCm-only numeric device index. Without it, all visible GPUs remain visible; this is not exclusive GPU ownership. |
| `--host`, `--port` | Host binding; defaults to `127.0.0.1:8000`. Use distinct ports for separate model instances. |
| `--detach`, `--restart-policy` | Managed background mode and Docker restart behavior. `unless-stopped` requires detached Docker mode. |
| `--startup-timeout` | Readiness deadline in seconds; default 1,800. |

The request scheduler also reserves one credit for each pending decision row,
up to a fixed 4,096 active rows across admitted requests. A single request is
limited to 1,024 state × question decisions. Queue size counts **requests**;
row credits and physical batch size count **decision rows**. Raising a queue
or concurrency limit cannot bypass the row cap or make a model forward wider.
Check `/api/status` for the live `max_concurrency`, `max_queue`, and
`max_active_rows` values instead of assuming the CLI defaults were used.

## Experimental Sol graph

`--experimental-qwen-rocm-graph-b8` is an explicit opt-in for canonical
Decision Sol on ROCm at physical batch size 8. It is off for every ordinary
launch and is rejected for other models, backends, or batch sizes. Graph
capture/replay has bounded fallback and telemetry; enabling it is not a
general speed guarantee. Keep the eager path as your baseline and qualify the
graph variant against the same model revision and artifact before relying on
it. See [measuring changes](./optimization.md).
