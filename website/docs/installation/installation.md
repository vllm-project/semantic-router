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
request. `vllm-sr serve` runs the whole local stack in Docker:

| Component | Port | Role |
| --- | --- | --- |
| **Router** | 8899 | Serves the OpenAI-compatible API itself (standalone mode, the default). |
| **Dashboard** | 8700 | Setup, models, routing and insights. |
| **[Model runtime](../model-runtime/get-started/overview.md)** | internal | Runs the Router's own decision model, and the embedding model when a recall feature needs it. |

The models that answer your users run elsewhere, such as Ollama, a vLLM
server or a hosted API. You connect them in the Dashboard.

:::tip Only need typed decisions?
To serve decision questions without a chat backend, follow the
[System One quickstart](../model-runtime/get-started/quickstart.md) instead. Both paths
use the same [frontend and model runtime](../overview/component-architecture).
:::

:::note Release channel
These pages follow `main`. Standalone mode and the built-in model runtime came
after the stable `vllm-sr` 0.4.0, so the tabs below select the development
channel: the curl installer uses `--channel dev`, and pip and uv allow
prereleases. For stable-release upgrades, see
[Upgrade and Rollback](upgrade-rollback).
:::

## Requirements

| Host | You need |
| --- | --- |
| Every host | Linux, macOS or WSL2; Docker (Linux can use Podman instead); Python 3.10 or newer; about 5 GB of free disk for the images, plus 1–1.5 GB for each built-in model your routes use |
| CPU | Nothing else. The `vllm-sr` image runs the built-in models on the CPU. The default decision model, vela3-fast, holds about 7 GB of memory once loaded, so give the Router about 12 GiB; the 307M embedding model adds about 1.3 GB when a recall feature uses it |
| AMD Instinct MI300X or MI325X | The ROCm driver on the host, Docker access to `/dev/kfd` and `/dev/dri`, and `--platform rocm`. Its `vllm-sr-rocm` image is a 6.5 GB download and takes about 20 GB of disk |
| NVIDIA | The NVIDIA Container Toolkit and `--platform cuda`. CUDA is validated (H200, A100, Tesla T4 records). Fused kernels without bit-exact evidence stay off the exact profile |

On macOS the Docker target runs on the CPU only; see
[Gateway Modes](gateway-modes#macos).

:::tip Installer options
Pass `--runtime podman` to the curl installer to force Podman, or
`--runtime skip` to skip container-runtime preparation:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --runtime skip
```

These are installer options. To choose a runtime when serving, use
`vllm-sr serve --container-runtime docker` or `podman`.
:::

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

`vllm-sr serve` waits during setup. After you activate, it starts the Router
from that configuration, prints the endpoints and exits. If you stopped it
first, run `vllm-sr serve` again; until you do, `vllm-sr status` reports that
setup is complete.

Agents can do the same work through the CLI and the Router management API,
without the Dashboard.

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

`vllm-sr/auto` lets the Router choose. The response headers say what it
chose:

- `x-vsr-selected-decision` names the route;
- `x-vsr-selected-model` names the model;
- `x-vsr-trace` lists the matched signals in its `signals` member, for a
  caller that sends `x-vsr-debug: true` and that `global.router.headers.debug`
  allows.

:::note
No caller receives `x-vsr-trace` by default. On a local stack, set
`global.router.headers.debug: any` to see it. See
[Router headers](../api/headers).
:::

## Operate the stack

```bash
vllm-sr status            # what runs, and whether setup or a restart is pending
vllm-sr logs router -f    # Router logs
vllm-sr stop
```

Changes the Router can hot-reload apply at once. A change the running
containers can't take, such as a listener's new port, is saved, and the
Dashboard says "Restart required: run `vllm-sr serve` to apply.". Run
`vllm-sr serve` to apply it; until then `vllm-sr status` reports it. See
[Configuration Management](configuration-management).

`vllm-sr serve --gateway extproc` puts an Envoy container in front of the
Router, as releases before standalone mode did.
[Gateway Modes](gateway-modes) explains when you need it.

## What the installer leaves behind

`install.sh` writes these paths; `--install-root` and `--bin-dir` move them:

| Path | What it is |
| --- | --- |
| `~/.local/share/vllm-sr/venv/` | The CLI's virtual environment. |
| `~/.local/share/vllm-sr/runtime.env` | The container runtime chosen at install time. `CONTAINER_RUNTIME` overrides it for one run, and a runtime missing from `PATH` falls back to auto-detection with a warning. Delete the file to return to auto-detection. |
| `~/.local/share/vllm-sr/first-run.log` | Kept only when the first `vllm-sr serve` waits for setup or fails. |
| `~/.local/bin/vllm-sr` | The launcher, which pins `VLLM_SR_INSTALL_ROOT`. |
| A line in `~/.bashrc` or `~/.zshrc`, or `~/.config/fish/completions/vllm-sr.fish` | Shell completion from `vllm-sr completion install`; see [Shell Completion](shell-completion). |

To uninstall, run `vllm-sr stop` while the CLI is still installed, which
removes the stack's containers and networks. Then delete the paths above,
including the completion line, or every new shell reports the missing command.

:::note What uninstalling keeps

- System packages the installer added when missing: Python and its venv
  support and, in serve mode, Homebrew `docker` and `colima` on macOS, or the
  Docker package, its service and your `docker` group membership on Linux.
- The images `vllm-sr serve` pulled, and the stack's Redis and Postgres data
  volumes.
- The state the first `vllm-sr serve` wrote where it ran: `config.yaml`,
  `.vllm-sr/` (Dashboard data, recipe store, logs) and `models/`. That is the
  directory the installer ran in, or `$HOME` when it was not writable.

:::

## Next

- [Connect an agent harness](agent-harness)
- [Choose a deployment](deployment-options)
- [Configure models](model-configuration)
- [Configure routing](configuration)
- [Use the built-in model runtime](../model-runtime/get-started/overview.md)
- [Use the Router API](../api/router)
- [Troubleshoot installation](../troubleshooting/common-errors)
