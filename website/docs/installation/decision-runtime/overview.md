---
title: Run a Decision model
description: Start one standalone Decision Runtime instance, verify readiness, and manage its lifecycle.
---

Decision Runtime is a standalone service for the six Decision 1.0 models. It is
not the Router's decision-rule engine: `vllm-sr drun` runs one pinned model per
instance and serves the [SystemOne and batch APIs](./api.md). Run separate
instances, on separate ports, when you need more than one model.

## Choose an image before launching

A released CLI can select a default image only after its package contains a
qualified, digest-pinned Decision image inventory. A source checkout or staging
wheel does not contain that release inventory. Its default-image launch fails
closed; it is not evidence that a published image exists.

For source or isolated validation, first build or obtain the matching Decision
Docker image and inspect its **full local image ID**:

```bash
IMAGE_ID=$(docker image inspect --format '{{.Id}}' YOUR_LOCAL_DECISION_IMAGE)
vllm-sr drun run llm-semantic-router/Decision-1.0-Kai-0.6B \
  --backend rocm --port 8001 --instance-name kai-8001 \
  --image "$IMAGE_ID" --image-pull-policy never --detach
```

The override must be a complete `sha256:` Docker image ID already on this
host; it is never pulled. It is for explicit local validation, **not** a
released registry digest or a substitute for release qualification. Podman
does not support this local-ID override. Once a qualified immutable image is
actually published and installed in the CLI's inventory, omit `--image` to use
that released default. See [model and backend support](./models.md) before
choosing another model or device.

## Check and stop the instance

```bash
curl -fsS http://127.0.0.1:8001/ready
curl -fsS http://127.0.0.1:8001/v1/models
vllm-sr drun status kai-8001
vllm-sr drun stop kai-8001
```

`/ready` confirms the resident model can serve requests; container creation
alone does not. The default host binding is loopback. Keep `/api/status` and
`/metrics` on a protected control-plane path if you expose inference through a
Gateway. Detached instances default to Docker restart policy `no`; use
`--restart-policy unless-stopped` with `--detach` only when the Docker service,
model cache, and mounted artifact will remain available after a reboot.

Continue with [models](./models.md), [launch parameters](./parameters.md), and
the [first API request](./api.md). The [generated CLI reference](../../api/cli.md#vllm-sr-drun-run)
lists every current option.
