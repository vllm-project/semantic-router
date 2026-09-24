---
title: Configure an instance
description: Choose a device, set request capacity, and manage a Decision model instance.
---

Start with the [basic `decision serve` command](./overview.md). You can
leave the capacity flags unset for the first run. The options below change how
that one model instance starts and handles traffic; the
[CLI reference](../../api/cli.md#vllm-sr-decision-serve) lists every flag.

## Device and model

| Option | When to use it |
| --- | --- |
| `--backend auto` (default) | Detect one visible GPU type, or select CPU when none is visible. Detection does not guarantee that the selected model and image can run there. If both ROCm and CUDA devices are visible, choose a backend explicitly. |
| `--backend rocm` or `--backend cpu` | Choose the intended execution path explicitly. The [model table](./models.md) shows which paths exist in the current build. `--backend cuda` is accepted by the CLI, but no Decision CUDA executor is installed yet. |
| `--gpu-device INDEX` | On ROCm, limit this instance to one visible GPU index. Without it, the container sees all visible GPUs; it does not reserve them for this instance. |
| `--cpu-threads N` | On CPU, set the Torch/BLAS thread limit. The default is the smaller of 8 and the container's CPU allowance. |
| `--revision SHA` | Use a full 40-character commit SHA for another revision of the same catalog model. Omit it to use the catalog revision. |

## Request capacity

One HTTP request can contain many decisions: one state with 10 questions is
one request and 10 decision rows. A batch of three states and four shared
questions is one request and 12 rows. These limits control different things:

| Option | Default | What it limits |
| --- | ---: | --- |
| `--max-concurrency` | 8 | Requests admitted at the same time. |
| `--max-queue` | 32 | Additional requests waiting for admission. `0` rejects a request immediately when capacity is full. |
| `--max-batch` | 8 | Compatible decision rows combined into one model forward. This does not limit the number of HTTP requests. |

For example, this starts Kai with room for four active requests and sixteen
waiting requests. The [starting guide](./overview.md) notes release
availability.

```bash
vllm-sr decision serve llm-semantic-router/Decision-1.0-Kai-0.6B \
  --backend cpu \
  --port 8001 --instance-name kai-8001 \
  --max-concurrency 4 --max-queue 16 --detach
```

At most 1,024 state/question decisions fit in one request, and at most 4,096
decision rows can be active across requests. Those bounds still apply if you
raise concurrency or queue length. When the service is full, it returns `529`
with `Retry-After`. `GET /api/status` shows the live request and row limits.

Keep `--max-batch` at the profile default unless you have verified a larger
value on this exact model and device. In particular, a larger queue does not
make a ROCm kernel support a larger physical batch. See
[measuring changes](./optimization.md).

## Image and lifecycle

| Option | When to use it |
| --- | --- |
| `--image` | Advanced override for a custom or locally built Decision Runtime image. Normally omit it and use the image selected by the installed CLI version. |
| `--image-pull-policy` | Control when to pull an image: `ifnotpresent` (default), `always`, or `never`. Most users can leave the default. |
| `--host`, `--port` | Publish the endpoint at `127.0.0.1:8000` by default. Give separate model instances separate ports. |
| `--instance-name` | Give a detached instance a stable name for `decision status` and `decision stop`. |
| `--detach`, `--restart-policy` | Keep the instance running in the background. Docker restart policy defaults to `no`; `unless-stopped` requires `--detach`. |
| `--startup-timeout` | Allow more time for model loading and readiness than the default 1,800 seconds. |

## Sol on ROCm

Decision Sol on ROCm automatically reuses the model backbone for repeated,
short eight-row batches. The first eligible shape is checked against ordinary
inference and captured before reuse, so that first request can take longer.
Longer inputs, other batch sizes, and captures that fail validation use ordinary
inference. No extra flag is needed.
