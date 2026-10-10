---
sidebar_position: 1
title: Quickstart
description: Install vLLM Semantic Router and send your first routed request.
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'
import CodeBlock from '@theme/CodeBlock'
import {
  AGENT_INSTALL_DOC_PATH,
  AGENT_INSTALL_PROMPT,
  AGENT_SKILL_PATH,
  CURL_INSTALL_COMMAND,
  PIP_INSTALL_COMMAND,
  UV_INSTALL_COMMAND,
} from '@site/src/data/installation'

# Quickstart

Install vLLM Semantic Router, connect one model, and send your first routed
request. To serve typed decision questions without a Chat backend, follow the
[System One quickstart](../model-runtime/quickstart.md) instead. Both paths use
the same [frontend and model runtime](../overview/component-architecture).

`vllm-sr serve` runs a local stack in Docker: the Router, which serves an
OpenAI-compatible API on port 8899 itself (standalone mode, the default), the
Dashboard on port 8700, and the
[model runtime](../model-runtime/overview.md) that runs the Router's own
classifiers and embedding models. The models that answer your users run
elsewhere, such as Ollama, a vLLM server or a hosted API, and you connect them
in the Dashboard.

:::note Release channel
These pages follow `main`. Standalone mode and the built-in model runtime came
after `vllm-sr` 0.4.0, the current stable release, which puts Envoy in front
of the Router and has no `--gateway` option. To follow these pages today,
use the development channel. The installation tabs below select it: the curl
installer uses `--channel dev`, and pip/uv allow prereleases. Stable-release
upgrade instructions are separate from this `main` quickstart.
:::

## Requirements

| Host | You need |
| --- | --- |
| Every host | Linux, macOS or WSL2; Docker (Linux can use Podman instead); Python 3.10 or newer; about 5 GB of free disk for the images, plus 1–1.5 GB for each built-in model your routes use |
| CPU | Nothing else. The `vllm-sr` image runs every built-in model on the CPU; plan about 1.3 GB of memory for each 307M task model your routes use |
| AMD Instinct MI300X or MI325X | The ROCm driver on the host, Docker access to `/dev/kfd` and `/dev/dri`, and `--platform rocm`. Its `vllm-sr-rocm` image is a 6.5 GB download and takes about 20 GB of disk |
| NVIDIA | The NVIDIA Container Toolkit and `--platform cuda` (works, not yet validated) |

On macOS the Docker target runs on the CPU only; see
[Gateway Modes](gateway-modes#macos).

For the curl installer, pass `--runtime podman` to force Podman or
`--runtime skip` to skip container-runtime preparation. For example:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --runtime skip
```

These are installer options. `vllm-sr serve --container-runtime` selects a
container runtime (`docker` or `podman`); `skip` is not a `serve` runtime.

## Install

<Tabs groupId="install-method" defaultValue="curl" values={[
  {label: 'curl', value: 'curl'},
  {label: 'pip', value: 'pip'},
  {label: 'uv', value: 'uv'},
  {label: 'Agent', value: 'agent'},
]}>
  <TabItem value="curl">
    <CodeBlock language="bash">{CURL_INSTALL_COMMAND}</CodeBlock>
  </TabItem>
  <TabItem value="pip">
    <CodeBlock language="bash">{PIP_INSTALL_COMMAND}</CodeBlock>
  </TabItem>
  <TabItem value="uv">
    <CodeBlock language="bash">{UV_INSTALL_COMMAND}</CodeBlock>
  </TabItem>
  <TabItem value="agent">
    Copy this prompt into your coding agent:
    <CodeBlock language="text">{AGENT_INSTALL_PROMPT}</CodeBlock>
    The prompt points to the public, self-contained <a href={AGENT_SKILL_PATH}>vLLM SR agent skill</a>.
    See <a href={AGENT_INSTALL_DOC_PATH}>Install with an agent</a> for the workflow and safety boundaries.
  </TabItem>
</Tabs>

Verify the CLI:

```bash
vllm-sr --version
```

## Start the stack

The curl installer starts the stack for you. After a pip or uv install, start
it yourself:

```bash
vllm-sr serve                  # Auto-detect the execution target
vllm-sr serve --platform rocm   # AMD GPUs
```

The first start pulls the images, which takes a few minutes. With no
`config.yaml` in the current directory, the stack starts in setup mode: the
Dashboard runs, and the Router waits for a configuration.

## Set up in the Dashboard

Open [http://localhost:8700](http://localhost:8700). The Dashboard listens on
`127.0.0.1`; on a remote host, run `ssh -L 8700:127.0.0.1:8700 <host>` first.

1. **Create the first administrator:** a name, an email and a password.
2. **Connect a model:** its name, its provider (vLLM, Ollama, OpenAI
   Compatible or Anthropic) and its address as the Router container sees it,
   for example `host.docker.internal:11434` for Ollama on the same host. For a
   first local model, follow [Configure models with Ollama](ollama).
3. **Choose routing:** **From scratch** makes one default route to that model.
4. **Review and Activate.**

`vllm-sr serve` keeps waiting during setup. After you activate, it starts the
Router from that configuration, prints the endpoints and exits. If you stopped
it first, run `vllm-sr serve` again; until then `vllm-sr status` says that
setup is complete. Agents can do the same work through the CLI and the Router
management API without using the Dashboard.

## Send a request

```bash
curl -s -D - http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-vsr-debug: true' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

`vllm-sr/auto` lets the Router choose. The response headers say what it chose:
`x-vsr-selected-decision` names the route and `x-vsr-selected-model` the
model; with `x-vsr-debug: true`, the `x-vsr-matched-*` headers list the
signals that matched. See [Router headers](../troubleshooting/vsr-headers).

## Operate the stack

```bash
vllm-sr status            # what runs, and whether setup or a restart is pending
vllm-sr logs router -f    # Router logs
vllm-sr stop
```

Later changes the Router hot-reloads apply at once. A change the running
containers can't take, such as a listener's new port, is saved and the
Dashboard answers "Restart required: run `vllm-sr serve` to apply."; the next
`vllm-sr serve` applies it, and `vllm-sr status` reports it until then. See
[Configuration Management](configuration-management).

`vllm-sr serve --gateway extproc` puts an Envoy container in front of the
Router, as releases before standalone mode did.
[Gateway Modes](gateway-modes) explains when you need it.

## Next

- [Connect an agent harness](agent-harness)
- [Choose a deployment](deployment-options)
- [Configure models](model-configuration)
- [Configure routing](configuration)
- [Use the built-in model runtime](../model-runtime/overview.md)
- [Use the Router API](../api/router)
- [Troubleshoot installation](../troubleshooting/common-errors)
