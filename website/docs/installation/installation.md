---
sidebar_position: 1
title: Quickstart
description: Install vLLM Semantic Router and send your first routed request.
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'
import CodeBlock from '@theme/CodeBlock'
import {
  AGENT_INSTALL_PROMPT,
  AGENT_SKILL_PATH,
  CURL_INSTALL_COMMAND,
  PIP_INSTALL_COMMAND,
  UV_INSTALL_COMMAND,
} from '@site/src/data/installation'

# Quickstart

Install vLLM Semantic Router, start the local stack, and send one request.

## Requirements

- Linux, macOS, or WSL2
- Python 3.10 or newer
- Docker; Linux can fall back to Podman

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
    The prompt points to the self-contained <a href={AGENT_SKILL_PATH}>vLLM SR agent skill</a>.
  </TabItem>
</Tabs>

Verify the CLI:

```bash
vllm-sr --version
```

The curl installer starts the stack automatically. After a pip or uv install,
start it with:

```bash
vllm-sr serve
```

Open [http://localhost:8700](http://localhost:8700), add a model endpoint, and
activate the generated configuration. Agents can do the same work through the
CLI and Router management API without using the Dashboard.

## Send a request

```bash
curl http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Next

- [Choose a deployment](deployment-options)
- [Configure models](model-configuration)
- [Configure routing](configuration)
- [Use the Router API](../api/router)
- [Troubleshoot installation](../troubleshooting/common-errors)
