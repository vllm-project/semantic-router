---
sidebar_position: 2
title: Install with an agent
description: Give an agent one prompt to install, configure, and verify vLLM Semantic Router through its CLI and Router API.
---

import CodeBlock from '@theme/CodeBlock'
import {
  AGENT_INSTALL_PROMPT,
  AGENT_SKILL_PATH,
} from '@site/src/data/installation'

# Install with an agent

Paste this prompt into a coding agent with terminal access to the target machine:

<CodeBlock language="text">{AGENT_INSTALL_PROMPT}</CodeBlock>

The <a href={AGENT_SKILL_PATH}>vLLM SR Skill</a> handles installation and
verification. Dashboard and Playground checks are optional.

Once the Router is running, [connect your agent harness](agent-harness).

## What the agent does

The Skill directs the agent to:

1. Inspect the host, existing installation, container runtime, accelerator, and
   available model endpoints without changing them.
2. Install the latest published stable CLI when needed, then verify its supported
   commands before changing a runtime. Discover configuration progressively from
   the CLI and the selected Router's schema and OpenAPI contract.
3. Create or update canonical YAML for the available model pool while keeping
   credentials in environment variables. Reuse packaged built-in Recipes through
   `vllm-sr recipe builtin list`, `export`, and `init` when requested.
4. For a new stack, validate locally, launch, and wait for readiness. For an
   existing stack, validate and plan before applying; listener or provider
   topology changes require an authorized deployment restart.
5. Preview the routing decision without backend generation, then send a real
   end-to-end request through the routed inference endpoint.
6. Leave the config path, active revision, validation result, and routing
   evidence for review.

Include known endpoint URLs, routing goals, and deployment constraints. The
agent discovers the rest and asks when a choice or permission is required.

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
| First launch | `vllm-sr config validate`, then `vllm-sr serve` and readiness |
| Plan an existing-stack change | `vllm-sr config validate`, then `vllm-sr config plan` |
| Apply a hot-reloadable change | `vllm-sr config apply`, which plans again before applying |
| Test routing logic | `vllm-sr route preview` |
| Test the complete data path | `vllm-sr route probe` |

The management origin serves health, discovery, configuration, and OpenAPI.
The routed inference origin separately serves the
[supported inference protocols](protocol-compatibility). An agent must discover
both rather than infer one from the other.

## Safety boundaries

- Keep API keys and provider credentials in environment variables; do not put
  secret values in prompts, YAML, command arguments, or logs.
- Keep changes within the requested deployment and existing authorization;
  obtain missing authorization before destructive changes, public exposure, or
  disruption of an unrelated service.
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
