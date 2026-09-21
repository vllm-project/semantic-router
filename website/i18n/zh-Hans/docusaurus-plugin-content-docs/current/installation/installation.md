---
sidebar_position: 1
title: 快速开始
description: 安装 vLLM Semantic Router 并发送你的第一条已路由请求。
translation:
  source_commit: "12c2aa4feb5c5d40d90104d8b09cade1facf0bf5"
  source_file: "docs/installation/installation.md"
  outdated: true
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

# 快速开始

安装 vLLM Semantic Router，启动本地栈，并发送一条请求。

## 系统要求

- Linux、macOS 或 WSL2
- Python 3.10 或更高版本
- Docker；Linux 可以回退到 Podman

## 安装

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
 将此提示词复制到你的编码 Agent 中：
 <CodeBlock language="text">{AGENT_INSTALL_PROMPT}</CodeBlock>
 该提示词指向公开、自包含的 <a href={AGENT_SKILL_PATH}>vLLM SR agent skill</a>。
 工作流和安全边界见 <a href={AGENT_INSTALL_DOC_PATH}>使用 Agent 安装</a>。
 </TabItem>
</Tabs>

验证 CLI：

```bash
vllm-sr --version
```

curl 安装程序会自动启动栈。使用 pip 或 uv 安装后，用以下命令启动：

```bash
vllm-sr serve
```

打开 [http://localhost:8700](http://localhost:8700)，添加模型端点，并激活生成的配置。Agent 可以通过 CLI 和 Router 管理 API 完成同样的工作，而无需使用控制面板。

## 发送请求

```bash
curl http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 下一步

- [选择部署方式](deployment-options)
- [配置模型](model-configuration)
- [配置路由](configuration)
- [使用 Router API](../api/router)
- [排查安装问题](../troubleshooting/common-errors)
