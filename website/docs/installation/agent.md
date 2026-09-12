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

Paste this prompt into a coding agent that can use a terminal and access the
machine where you want to run vLLM Semantic Router:

<CodeBlock language="text">{AGENT_INSTALL_PROMPT}</CodeBlock>

That is the complete bootstrap prompt. It points the agent to the public,
self-contained <a href={AGENT_SKILL_PATH}>vLLM SR Skill</a>; installation details
stay in the Skill instead of being copied into every prompt. The Dashboard is
optional and is not part of the agent workflow.

## What the agent does

The Skill directs the agent to:

1. Inspect the host, existing installation, container runtime, accelerator, and
   available model endpoints without changing them.
2. Install the stable CLI when needed, then discover the running Router's
   supported operations, configuration schema, and OpenAPI contract.
3. Create or update canonical YAML for the available model pool while keeping
   credentials in environment variables.
4. Validate and plan the change before applying it. Changes to listeners or
   provider topology require an explicit deployment restart.
5. Preview the routing decision without calling a model, then send a real
   end-to-end request through the routed inference endpoint.
6. Leave the config path, active revision, validation result, and routing
   evidence for review.

Tell the agent your model endpoint URLs, routing objective, or deployment
constraints in the same message when they are already known. Otherwise, the
agent will discover what it can and ask only when a choice or permission is
required.

## Direct contracts

The agent works against the same contracts used by the CLI and Dashboard; it
does not automate the Dashboard UI.

| Purpose | CLI or Router contract |
| --- | --- |
| Discover operations | `GET /api/v1?audience=agent&visibility=primary` |
| Inspect an operation | `GET /openapi.json?path=...&method=...` |
| Discover configuration | `vllm-sr config schema` or `GET /api/v1/config/schema` |
| Validate and plan | `vllm-sr config validate`, then `vllm-sr config plan` |
| Apply a hot-reloadable change | `vllm-sr config apply` with the planned ETag |
| Test routing logic | `vllm-sr route preview` |
| Test the complete data path | `vllm-sr route probe` |

The management origin serves health, discovery, configuration, and OpenAPI.
The routed inference origin separately serves OpenAI-compatible requests. An
agent must discover both rather than infer one from the other.

## Safety boundaries

- Keep API keys and provider credentials in environment variables; do not put
  secret values in prompts, YAML, command arguments, or logs.
- Review any privileged, destructive, publicly exposed, or service-disrupting
  action before allowing it.
- A routing preview proves the decision path but does not call a model. A route
  probe is the end-to-end check that reaches the selected backend.
- Use the running Router's discovery, schema, and OpenAPI responses as the
  authority for its installed version.

For deeper configuration work, continue with the
[configuration contract](configuration-contract) and
[configuration workflows](configuration-workflows). For model and
Mixture-of-Models evaluation, use the
[agent evaluation loop](../benchmarking/agent-evaluation-loop).
