---
sidebar_position: 2
title: Install with an agent
description: Give a coding agent one prompt to install, configure, and verify vLLM Semantic Router on a CPU or GPU host, through its CLI and Router API.
---

import CodeBlock from '@theme/CodeBlock'
import {
  AGENT_INSTALL_PROMPT,
  AGENT_SKILL_PATH,
} from '@site/src/data/installation'

# Install with an agent

Paste this prompt into a coding agent with terminal access to the target machine:

<CodeBlock language="text">{AGENT_INSTALL_PROMPT}</CodeBlock>

The <a href={AGENT_SKILL_PATH}>vLLM SR Skill</a> takes the agent from a host
with Docker to a verified, routed request. Dashboard and Playground checks are
optional. Add the model endpoint to use, such as a local Ollama or vLLM server
or a hosted API, and any constraints; the agent asks before it changes anything
outside the stack.

Once the Router is running, [connect your agent harness](agent-harness).

:::note Release channel
The Skill follows `main`, as these pages do. It installs the stable release
when that release has standalone mode, and the development channel otherwise.
Stable `0.4.0` predates standalone mode and the model runtime, so for now the
agent installs the development channel and tells you so.
:::

## What the agent does

1. **Preflight**, without changing anything: Docker access, Python and its
   `venv` support, free disk and ports, AMD or NVIDIA GPU devices, and any
   `vllm-sr` stack that already runs.
2. **Chooses the path:** the release channel; the platform, `--platform amd` or
   `--platform nvidia` when the host has those GPUs; standalone mode, or
   `--gateway extproc` when you need Envoy; Docker or Kubernetes; and the model
   endpoint, checked the way the Router container will reach it.
3. **Installs the CLI** with the curl installer, without starting a stack.
4. **Writes a configuration** for your model, with the inference API bound to
   `127.0.0.1` and one keyword route that proves signals reach decisions, and
   validates it with `vllm-sr config validate`.
5. **Starts the stack** with `vllm-sr serve --config config.yaml`, never in
   setup mode, which would wait for a person.
6. **Verifies it** against explicit criteria: `vllm-sr status`,
   `GET /v1/models`, `vllm-sr route preview`, a routed request with its
   `x-vsr-selected-decision` and `x-vsr-selected-model` headers, and
   `vllm-sr route probe`. On a GPU host it also serves one Router model on the
   GPU in engine mode.
7. **Hands off** the version and channel, the configuration path, the
   endpoints, the Dashboard's first-administrator step, and what each check
   returned.

Every step is safe to rerun: an existing configuration or running stack is
verified, not replaced. The Skill's references cover GPU details, Envoy,
Kubernetes, configuration changes, troubleshooting, recipe tuning, and
evaluation.

## Direct contracts

The agent works against the same contracts used by the CLI and Dashboard.
Dashboard verification is optional and uses real server responses and streamed
Playground output when requested.

| Purpose | CLI or Router contract |
| --- | --- |
| Discover operations | `GET /api/v1?audience=agent&visibility=primary` |
| Inspect an operation | `GET /openapi.json?path=...&method=...` |
| Discover configuration | `vllm-sr config schema` or `GET /api/v1/config/schema` |
| Discover packaged Recipes | `vllm-sr recipe builtin list` |
| First launch | `vllm-sr config validate`, then `vllm-sr serve --config config.yaml` |
| Check the stack | `vllm-sr status` |
| Plan an existing-stack change | `vllm-sr config validate`, then `vllm-sr config plan` |
| Apply a hot-reloadable change | `vllm-sr config apply`, which plans again before applying |
| Test routing logic | `vllm-sr route preview` |
| Test the complete data path | `vllm-sr route probe` |
| Serve a Router model alone | `vllm-sr serve --mode engine --model ARTIFACT` |

The management origin, port 8080 on a local stack, serves health, discovery,
configuration, and OpenAPI. The inference listener, port 8899, separately
serves the [supported inference protocols](protocol-compatibility). An agent
must discover both rather than infer one from the other.

## Safety boundaries

- Keep API keys and provider credentials in environment variables; do not put
  secret values in prompts, YAML, command arguments, or logs.
- Keep changes within the requested deployment and existing authorization;
  obtain missing authorization before installing packages, publishing a port
  beyond loopback, other destructive changes, or disruption of an unrelated
  service such as a stack the agent didn't start.
- A routing preview runs routing signals without backend generation. A route
  probe is the end-to-end check that reaches the selected backend.
- Use the running Router's discovery, schema, and OpenAPI responses as the
  authority for its installed version.

For deeper configuration work, continue with the
[configuration contract](configuration-contract) and
[configuration workflows](configuration-workflows). For model and
Mixture-of-Models evaluation, use the
[agent evaluation loop](../benchmarking/agent-evaluation-loop).

## Maintaining the Skill

The single authored source is
[`tools/agent/skills/vllm-sr-agent-operations/`](https://github.com/vllm-project/semantic-router/tree/main/tools/agent/skills/vllm-sr-agent-operations),
including its optional references. Edit those files and run
`make agent-skill-sync`; do not edit the public copies directly. Use absolute
URLs in the Skill and every reference so either copy can be installed alone.
The generator changes the public skill name, checks linked documents, and makes
any relative links absolute. Commit the generated files alongside their source; the website
publishes those static files directly. A remote agent can load each reference
without a repository checkout.

`make agent-skill-check`, pre-commit, and `make harness-check` reject missing or
stale generated files. The repository and website therefore share one workflow
while keeping their respective skill names and installation paths.

Change the Skill together with the CLI behavior it describes, and follow it on a
fresh host before publishing: its commands and expected outputs are the
contract an agent runs.
