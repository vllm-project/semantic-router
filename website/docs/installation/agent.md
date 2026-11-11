---
sidebar_position: 2
title: Install with an agent
description: Give a coding agent one prompt to install, configure and verify vLLM Semantic Router on a CPU or GPU host or a Kubernetes cluster, then to maintain, troubleshoot and tune it, through its CLI and Router API.
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
with Docker to a verified, routed request, and stays useful afterwards: the
same skill upgrades the stack, migrates a configuration to v0.5, diagnoses
failures and tunes routing. Dashboard and Playground checks are optional.
Add the model endpoint to use, such as a local
Ollama or vLLM server or a hosted API, and any constraints; the agent asks
before it changes anything outside the stack.

Once the Router is running, [connect your agent harness](agent-harness).

:::note Release channel
The Skill follows `main`, as these pages do. It installs the stable release
when that release reads the v0.5 configuration (`vllm-sr config schema`
reports `"config_version": "v0.5"`), and the development channel otherwise,
and tells you which one it installed.
:::

## What the Skill covers

The skill is a short router over six tasks. Each task states its
preconditions, the reference with its steps, what the output must contain
(exit codes, JSON fields, header values), what to do when a step fails, and
when to stop and ask you.

| Task | What the agent does |
| --- | --- |
| Deploy | Installs the CLI, writes and validates a configuration, starts the stack on Docker (CPU, AMD or NVIDIA) or Kubernetes (Helm or the operator), and checks it |
| Configure | Changes models, signals, decisions, plugins, recipes or entrypoints of a running stack with `config validate`, `config plan` and `config apply` |
| Verify | Previews routing, probes the routed path, reads the response headers and checks that the decision model is ready |
| Maintain | Upgrades, runs `vllm-sr config migrate` for v0.5, rolls back configuration versions, backs up and restores, rotates secrets, reads logs and health, removes the stack |
| Troubleshoot | Matches a symptom to a check and a fix, including the v0.5 load errors, a decision model that isn't ready, header trust and image routes |
| Optimize | Chooses the decision model and its hardware, tunes recipes, caches and judge caps, and measures quality, latency and cost with a short sr-bench run |

## A first install

1. **Preflight**, without changing anything: Docker access and memory, Python
   and its `venv` support, free disk and ports, AMD or NVIDIA GPU devices, and
   any `vllm-sr` stack that already runs.
2. **Chooses the path:** the release channel; the platform (`--platform rocm`
   or `cuda` when the host has those GPUs, `--platform cpu` when you keep them
   for other work); standalone mode, or `--gateway extproc` when you need
   Envoy; Docker or Kubernetes; and the model endpoint, checked the way the
   Router container will reach it.
3. **Installs the CLI** with the curl installer, without starting a stack, and
   checks that it reads the v0.5 configuration.
4. **Writes a configuration** for your model, with the inference API bound to
   `127.0.0.1`, one keyword route, and one route that the decision model
   judges (a `domains` rule), and validates it with `vllm-sr config validate`.
5. **Starts the stack** with `vllm-sr serve --config config.yaml`, never in
   setup mode, which would wait for a person. The first start downloads the
   default decision model, vela3-fast, which holds about 7 GB on the CPU.
6. **Verifies it** against explicit criteria: `vllm-sr status`,
   `GET /v1/models`, the decision model's readiness, `vllm-sr route preview`
   for both routes, a routed request with its `x-vsr-selected-decision`,
   `x-vsr-selected-model` and `x-vsr-response-path` headers, and
   `vllm-sr route probe`. On a GPU host it also serves one decision model on
   the GPU in engine mode.
7. **Hands off** the version and channel, the configuration path, the
   endpoints, the decision model, the Dashboard's first-administrator step,
   and what each check returned.

Every step is safe to rerun: an existing configuration or running stack is
verified, not replaced.

## Direct contracts

The agent works against the same contracts used by the CLI and Dashboard.

| Purpose | CLI or Router contract |
| --- | --- |
| Discover operations | `GET /api/v1?audience=agent&visibility=primary` |
| Inspect an operation | `GET /openapi.json?path=...&method=...` |
| Discover configuration | `vllm-sr config schema` or `GET /api/v1/config/schema` |
| Discover packaged Recipes | `vllm-sr recipe builtin list` |
| First launch | `vllm-sr config validate`, then `vllm-sr serve --config config.yaml` |
| Check the stack | `vllm-sr status`, `GET /ready` |
| Check the decision model | `GET /api/v1/inventory/model-runtime` |
| Plan an existing-stack change | `vllm-sr config validate`, then `vllm-sr config plan` |
| Apply a hot-reloadable change | `vllm-sr config apply`, which plans again before applying |
| Move a configuration to v0.5 | `vllm-sr config migrate` |
| List and restore versions | `vllm-sr config versions`, `vllm-sr config rollback` |
| Test routing logic | `vllm-sr route preview` |
| Test the complete data path | `vllm-sr route probe` |
| Serve a decision model alone | `vllm-sr serve ARTIFACT --engine` |
| Measure quality, latency and cost | `vllm-sr benchmark plan`, `vllm-sr benchmark run` |

The management origin, port 8080 on a local stack, serves health, discovery,
configuration, and OpenAPI. The inference listener, port 8899, separately
serves the [supported inference protocols](protocol-compatibility). An agent
must discover both rather than infer one from the other.

## Safety boundaries

- Keep API keys and provider credentials in environment variables; do not put
  secret values in prompts, YAML, command arguments, or logs.
- Keep changes within the requested deployment and existing authorization;
  obtain missing authorization before installing packages, publishing a port
  beyond loopback, stopping or upgrading a stack you run, deleting volumes or
  state, or disrupting an unrelated service such as a stack the agent didn't
  start.
- A routing preview runs routing signals without backend generation. A route
  probe is the end-to-end check that reaches the selected backend.
- Use the running Router's discovery, schema, and OpenAPI responses as the
  authority for its installed version.

For deeper configuration work, continue with the
[configuration contract](configuration-contract) and
[configuration workflows](configuration-workflows); for v0.5, the
[migration guide](migrate-to-v0-5) and [System One](../system-one/overview).
For model and Mixture-of-Models evaluation, use the
[agent evaluation loop](../benchmarking/agent-evaluation-loop).

## Maintaining the Skill

The single authored source is
[`tools/agent/skills/vllm-sr-agent-operations/`](https://github.com/vllm-project/semantic-router/tree/main/tools/agent/skills/vllm-sr-agent-operations),
including its references. Edit those files and run `make agent-skill-sync`;
do not edit the public copies directly. Use absolute URLs in the Skill and
every reference so either copy can be installed alone. The generator changes
the public skill name, checks linked documents, and makes any relative links
absolute. Commit the generated files alongside their source; the website
publishes those static files directly. A remote agent can load each reference
without a repository checkout.

`make agent-skill-check`, pre-commit, and `make harness-check` reject missing or
stale generated files. Two tests keep the Skill honest: the Router loads
every YAML block of the Skill with its own configuration parser, and the CLI's
unit suite parses every `vllm-sr` command line with the CLI's own argument
parser and checks every command and option the prose names.

Change the Skill together with the CLI behavior it describes, and follow it on a
fresh host before publishing: its commands and expected outputs are the
contract an agent runs.
