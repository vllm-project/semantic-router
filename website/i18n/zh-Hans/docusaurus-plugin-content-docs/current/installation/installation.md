---
is_mtpe: false
sidebar_position: 1
title: 快速开始
description: 安装 vLLM Semantic Router 并发送你的第一条已路由请求。
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
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

# 快速开始 {#quickstart}

安装 vLLM Semantic Router，接入一个模型，发送第一条经过路由的请求。`vllm-sr serve` 在 Docker 中运行整个本地服务：

| 组件 | 端口 | 作用 |
| --- | --- | --- |
| **Router** | 8899 | 自己提供 OpenAI 兼容 API（standalone 模式，默认）。 |
| **Dashboard** | 8700 | 设置、模型、路由和洞察。 |
| **[模型运行时](../model-runtime/get-started/overview.md)** | 内部 | 运行 Router 自己的决策模型；召回功能需要时，还运行嵌入模型。 |

回答用户的模型在别处运行，例如 Ollama、vLLM 服务或托管 API。你在 Dashboard 中接入它们。

:::tip 只需要类型化决策？
如果想在没有对话后端的情况下提供决策问题服务，请改看 [System One 快速开始](../model-runtime/get-started/quickstart.md)。两种用法共用同一套[前端与模型运行时](../overview/component-architecture)。
:::

:::note 发布渠道
本文档跟随 `main`。standalone 模式和内置模型运行时晚于稳定版 `vllm-sr` 0.4.0，因此下面的选项卡选择开发渠道：curl 安装脚本使用 `--channel dev`，pip 和 uv 允许预发布版本。稳定版的升级方式见[升级与回滚](upgrade-rollback)。
:::

## 系统要求 {#requirements}

| 主机 | 你需要 |
| --- | --- |
| 所有主机 | Linux、macOS 或 WSL2；Docker（Linux 也可以用 Podman）；Python 3.10 或更高版本；约 5 GB 可用磁盘放镜像，路由用到的每个内置模型另需 1–1.5 GB |
| CPU | 无需其他。`vllm-sr` 镜像在 CPU 上运行内置模型。默认决策模型 vela3-fast 加载后约占 7 GB 内存，因此请给 Router 约 12 GiB；召回功能用到 307M 嵌入模型时再加约 1.3 GB |
| AMD Instinct MI300X 或 MI325X | 主机上的 ROCm 驱动、Docker 能访问 `/dev/kfd` 和 `/dev/dri`，以及 `--platform rocm`。它使用的 `vllm-sr-rocm` 镜像下载约 6.5 GB，占用约 20 GB 磁盘 |
| NVIDIA | NVIDIA Container Toolkit 和 `--platform cuda`。CUDA 已验证（H200、A100、Tesla T4 记录）。没有 bit-exact 证据的融合 kernel 不进入精确配置 |

在 macOS 上，Docker 目标只使用 CPU；见[网关模式](gateway-modes#macos)。

:::tip 安装脚本选项
向 curl 安装脚本传入 `--runtime podman` 可强制使用 Podman，传入 `--runtime skip` 可跳过容器运行时准备：

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --runtime skip
```

这些是安装脚本的选项。运行服务时要选择容器运行时，请使用 `vllm-sr serve --container-runtime docker` 或 `podman`。
:::

## 安装 {#install}

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

## 启动服务 {#start-the-stack}

curl 安装脚本会替你启动服务。使用 pip 或 uv 安装后，请自行启动：

```bash
vllm-sr serve                  # Auto-detect the execution target
vllm-sr serve --platform rocm   # AMD GPUs
```

首次启动会拉取镜像，需要几分钟。当前目录没有 `config.yaml` 时，服务以设置模式启动：Dashboard 运行，Router 等待一份配置。

## 在 Dashboard 中设置 {#set-up-in-the-dashboard}

打开 [http://localhost:8700](http://localhost:8700)。Dashboard 监听 `127.0.0.1`；在远程主机上，请先运行 `ssh -L 8700:127.0.0.1:8700 <host>`。

1. **创建第一个管理员：** 名字、邮箱和密码。
2. **接入模型：** 模型名、提供方（vLLM、Ollama、OpenAI Compatible 或 Anthropic），以及 Router 容器看到的地址，例如同一主机上的 Ollama 用 `host.docker.internal:11434`。第一个本地模型可按[使用 Ollama 配置模型](ollama)操作。
3. **选择路由：** **From scratch** 会生成一条到该模型的默认路由。
4. **检查并 Activate。**

设置期间 `vllm-sr serve` 会保持等待。你激活后，它用该配置启动 Router，打印各端点后退出。如果你先停止了它，请再运行一次 `vllm-sr serve`；在此之前，`vllm-sr status` 会报告设置已完成。

Agent 可以通过 CLI 和 Router 管理 API 完成同样的工作，无需使用 Dashboard。

## 发送请求 {#send-a-request}

```bash
curl -s -D - http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-vsr-debug: true' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

`vllm-sr/auto` 让 Router 来选择。响应头说明它选了什么：

- `x-vsr-selected-decision` 指明路由；
- `x-vsr-selected-model` 指明模型；
- 对于发送了 `x-vsr-debug: true` 且被 `global.router.headers.debug` 允许的调用方，`x-vsr-trace` 的 `signals` 成员会列出命中的信号。

:::note
默认情况下，任何调用方都不会收到 `x-vsr-trace`。在本地服务上，设置 `global.router.headers.debug: any` 即可看到它。见 [Router 响应头](../api/headers)。
:::

## 运维服务 {#operate-the-stack}

```bash
vllm-sr status            # 正在运行什么，以及是否有待完成的设置或重启
vllm-sr logs router -f    # Router 日志
vllm-sr stop
```

Router 能热加载的变更会立即生效。运行中的容器无法接受的变更（例如 listener 换了端口）会先保存，Dashboard 会提示“Restart required: run `vllm-sr serve` to apply.”。运行 `vllm-sr serve` 即可应用；在此之前，`vllm-sr status` 会一直报告。见[配置管理](configuration-management)。

`vllm-sr serve --gateway extproc` 会像 standalone 模式之前的版本那样，在 Router 前面放一个 Envoy 容器。[网关模式](gateway-modes)说明了什么时候需要它。

## 安装脚本留下了什么 {#what-the-installer-leaves-behind}

`install.sh` 会写入以下路径；`--install-root` 和 `--bin-dir` 可以改变它们的位置：

| 路径 | 内容 |
| --- | --- |
| `~/.local/share/vllm-sr/venv/` | CLI 的虚拟环境。 |
| `~/.local/share/vllm-sr/runtime.env` | 安装时选定的容器运行时。`CONTAINER_RUNTIME` 可对单次运行覆盖它；它记录的运行时不在 `PATH` 中时，会回退到自动检测并给出警告。删除该文件即可恢复自动检测。 |
| `~/.local/share/vllm-sr/first-run.log` | 仅在第一次 `vllm-sr serve` 等待设置或失败时保留。 |
| `~/.local/bin/vllm-sr` | 启动器，固定了 `VLLM_SR_INSTALL_ROOT`。 |
| `~/.bashrc` 或 `~/.zshrc` 中的一行，或 `~/.config/fish/completions/vllm-sr.fish` | `vllm-sr completion install` 写入的 shell 补全；见 [Shell 补全](shell-completion)。 |

卸载时，请在 CLI 仍然安装的情况下运行 `vllm-sr stop`，它会删除服务的容器和网络。然后删除上面的路径，包括补全那一行，否则每个新 shell 都会报告找不到命令。

:::note 卸载后仍会保留

- 安装脚本在缺失时添加的系统软件包：Python 及其 venv 支持；在 serve 模式下，还有 macOS 上的 Homebrew `docker` 和 `colima`，或 Linux 上的 Docker 软件包、它的服务以及你的 `docker` 组成员身份。
- `vllm-sr serve` 拉取的镜像，以及服务写入的 Redis 和 Postgres 数据卷。
- 第一次 `vllm-sr serve` 在运行目录中写入的状态：`config.yaml`、`.vllm-sr/`（Dashboard 数据、配方存储、日志）和 `models/`。该目录就是安装脚本运行的目录；它不可写时为 `$HOME`。

:::

## 下一步 {#next}

- [接入 agent harness](agent-harness)
- [选择部署方式](deployment-options)
- [配置模型](model-configuration)
- [配置路由](configuration)
- [使用内置模型运行时](../model-runtime/get-started/overview.md)
- [使用 Router API](../api/router)
- [排查安装问题](../troubleshooting/common-errors)
