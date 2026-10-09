---
sidebar_position: 1
title: 快速开始
description: 安装 vLLM Semantic Router 并发送你的第一条已路由请求。
translation:
  source_commit: "b9b183307e97f3ce8448d2838d0f1bf99972d336"
  source_file: "docs/installation/installation.md"
  outdated: false
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

安装 vLLM Semantic Router，接入一个模型，发送第一条路由请求。

`vllm-sr serve` 在 Docker 中运行一个本地栈：Router 自己在 8899 端口提供 OpenAI
兼容 API（standalone 模式，默认），控制面板在 8700 端口，还有运行 Router 自身分类器和
嵌入模型的[模型运行时](../model-runtime/overview.md)。回答用户的模型在别处运行，例如
Ollama、vLLM 服务或托管 API，你在控制面板中接入它们。

:::note 发布渠道
本文档跟随 `main`。standalone 模式和内置模型运行时晚于当前稳定版 `vllm-sr` 0.4.0：
0.4.0 仍在 Router 前面放置 Envoy，也没有 `--gateway` 选项。想按本文档操作，请安装开发渠道：
给 curl 安装脚本传入 `--channel dev`。
:::

## 系统要求

| 主机 | 你需要 |
| --- | --- |
| 所有主机 | Linux、macOS 或 WSL2；Docker（Linux 也可以用 Podman）；Python 3.10 或更高版本；约 5 GB 可用磁盘放镜像，路由用到的每个内置模型另需 1–1.5 GB |
| CPU | 无需其他。`vllm-sr` 镜像在 CPU 上运行所有内置模型；路由用到的每个 307M 任务模型约需 1.3 GB 内存 |
| AMD Instinct MI300X 或 MI325X | 主机上的 ROCm 驱动、Docker 能访问 `/dev/kfd` 和 `/dev/dri`，以及 `--platform rocm`。它使用的 `vllm-sr-rocm` 镜像下载约 6.5 GB，占用约 20 GB 磁盘 |
| NVIDIA | NVIDIA Container Toolkit 和 `--platform cuda`（可用，尚未验证） |

在 macOS 上，docker 目标只使用 CPU；见[网关模式](gateway-modes#macos)。

使用 curl 安装脚本时，可传入 `--runtime podman` 强制使用 Podman，或传入
`--runtime skip` 跳过容器运行时准备。例如：

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel stable --runtime skip
```

这些参数属于安装脚本。`vllm-sr serve --container-runtime` 用于选择容器运行时
（`docker` 或 `podman`）；`skip` 不是 `serve` 的运行时选项。

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

## 启动栈

curl 安装脚本会替你启动栈。使用 pip 或 uv 安装后，自己启动：

```bash
vllm-sr serve                  # Auto-detect the execution target
vllm-sr serve --platform rocm   # AMD GPU
```

首次启动会拉取镜像，需要几分钟。当前目录没有 `config.yaml` 时，栈以设置模式启动：控制面板运行，
Router 等待一份配置。

## 在控制面板中设置

打开 [http://localhost:8700](http://localhost:8700)。控制面板监听 `127.0.0.1`；在远程主机上，
先运行 `ssh -L 8700:127.0.0.1:8700 <host>`。

1. **创建第一个管理员**：名字、邮箱和密码。
2. **接入模型**：模型名、提供方（vLLM、Ollama、OpenAI Compatible 或 Anthropic），以及
   Router 容器看到的地址，例如同一主机上的 Ollama 用 `host.docker.internal:11434`。
   第一个本地模型可按[使用 Ollama 配置模型](ollama)操作。
3. **选择路由**：**From scratch** 会生成一条到该模型的默认路由。
4. **检查并 Activate**。

设置期间 `vllm-sr serve` 保持等待。你激活后，它用该配置启动 Router，打印各端点后退出。
如果你先停止了它，再运行一次 `vllm-sr serve`；在此之前 `vllm-sr status` 会提示设置已完成。
Agent 可以通过 CLI 和 Router 管理 API 完成同样的工作，而无需使用控制面板。

## 发送请求

```bash
curl -s -D - http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-vsr-debug: true' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

`vllm-sr/auto` 让 Router 来选择。响应头说明它选了什么：`x-vsr-selected-decision` 是路由，
`x-vsr-selected-model` 是模型；带上 `x-vsr-debug: true` 时，`x-vsr-matched-*` 响应头列出
命中的信号。见 [Router 响应头](../troubleshooting/vsr-headers)。

## 运维这个栈

```bash
vllm-sr status            # 正在运行什么，以及是否有待完成的设置或重启
vllm-sr logs router -f    # Router 日志
vllm-sr stop
```

之后，Router 能热加载的变更会立即生效。运行中的容器无法接受的变更（例如监听器换了端口）会先保存，
控制面板会提示“Restart required: run `vllm-sr serve` to apply.”；下一次 `vllm-sr serve`
会应用它，在此之前 `vllm-sr status` 会一直报告。见[配置管理](configuration-management)。

`vllm-sr serve --gateway extproc` 会像 standalone 模式之前的版本那样，在 Router 前面放一个
Envoy 容器。[网关模式](gateway-modes)说明了什么时候需要它。

## 下一步

- [接入 Agent Harness](agent-harness)
- [选择部署方式](deployment-options)
- [配置模型](model-configuration)
- [配置路由](configuration)
- [使用内置模型运行时](../model-runtime/overview.md)
- [使用 Router API](../api/router)
- [排查安装问题](../troubleshooting/common-errors)
