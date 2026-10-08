---
sidebar_position: 2
title: 使用 Agent 安装
description: 向编码 Agent 提供一条提示词，即可通过 CLI 和 Router API 在 CPU 或 GPU 主机上完成 vLLM Semantic Router 的安装、配置与验证。
translation:
  source_commit: "c34ca03a83d8b95eadfd86ab0a4fb11ac17af8d4"
  source_file: "docs/installation/agent.md"
  outdated: false
---

import CodeBlock from '@theme/CodeBlock'
import {
  AGENT_INSTALL_PROMPT,
  AGENT_SKILL_PATH,
} from '@site/src/data/installation'

# 使用 Agent 安装

将提示词粘贴到能使用目标机器终端的编码 Agent 中：

<CodeBlock language="text">{AGENT_INSTALL_PROMPT}</CodeBlock>

<a href={AGENT_SKILL_PATH}>vLLM SR Skill</a> 带领 Agent 从一台装有 Docker 的主机走到一条经过验证的路由请求。控制面板和 Playground 检查按需进行。请附上要使用的模型端点，例如本地的 Ollama 或 vLLM 服务、托管 API，以及部署约束；改动栈以外的任何东西之前，Agent 都会先询问。

Router 启动后，[接入 Agent Harness](agent-harness)。

:::note 发布渠道
Skill 与本文档一样跟随 `main`。稳定版具备 standalone 模式时，它安装稳定版，否则安装开发渠道。稳定版 `0.4.0` 早于 standalone 模式和模型运行时，所以目前 Agent 会安装开发渠道，并告诉你这一点。
:::

## Agent 会做什么

1. **预检**，不做任何改动：Docker 访问权限、Python 及其 `venv` 支持、可用磁盘和端口、AMD 或 NVIDIA GPU 设备，以及已在运行的 `vllm-sr` 栈。
2. **选择路径**：发布渠道；平台，主机有相应 GPU 时用 `--platform amd` 或 `--platform nvidia`；standalone 模式，需要 Envoy 时用 `--gateway extproc`；Docker 或 Kubernetes；以及模型端点，并按 Router 容器访问它的方式检查连通性。
3. **安装 CLI**：使用 curl 安装脚本，不启动栈。
4. **编写配置**：面向你的模型，推理 API 绑定到 `127.0.0.1`，并带一条关键词路由来证明信号能到达决策；然后用 `vllm-sr config validate` 校验。
5. **启动栈**：运行 `vllm-sr serve --config config.yaml`，绝不进入会等待人工操作的设置模式。
6. **按明确标准验证**：`vllm-sr status`、`GET /v1/models`、`vllm-sr route preview`、一条路由请求及其 `x-vsr-selected-decision` 和 `x-vsr-selected-model` 响应头，以及 `vllm-sr route probe`。在 GPU 主机上，它还会以引擎模式在 GPU 上运行一个 Router 模型。
7. **交付结果**：版本与渠道、配置路径、各端点、控制面板创建第一个管理员的步骤，以及每项检查的结果。

每一步都可以安全地重复执行：已有的配置或正在运行的栈只会被验证，不会被替换。Skill 的参考文档覆盖 GPU 细节、Envoy、Kubernetes、配置变更、故障排查、配方调优和评估。

## 直接契约

Agent 使用的契约与 CLI 和控制面板相同。需要控制面板验证时，它会检查真实的服务响应和 Playground 流式输出。

| 用途 | CLI 或 Router 契约 |
| --- | --- |
| 发现操作 | `GET /api/v1?audience=agent&visibility=primary` |
| 检查某项操作 | `GET /openapi.json?path=...&method=...` |
| 发现配置 | `vllm-sr config schema` 或 `GET /api/v1/config/schema` |
| 发现内置配方 | `vllm-sr recipe builtin list` |
| 首次启动 | `vllm-sr config validate`，然后 `vllm-sr serve --config config.yaml` |
| 检查栈 | `vllm-sr status` |
| 规划已有部署的变更 | `vllm-sr config validate`，然后 `vllm-sr config plan` |
| 应用可热重载的变更 | `vllm-sr config apply`，执行前会重新规划 |
| 测试路由逻辑 | `vllm-sr route preview` |
| 测试完整数据路径 | `vllm-sr route probe` |
| 单独运行一个 Router 模型 | `vllm-sr serve MODEL`（引擎模式） |

管理源（本地栈的 8080 端口）提供健康检查、发现、配置和 OpenAPI。推理监听器（8899 端口）单独提供[受支持的推理协议](protocol-compatibility)。Agent 必须分别发现两者，而不能从其中一个推断另一个。

## 安全边界

- 将 API 密钥和 provider 凭据保留在环境变量中；不要把密钥值写入提示词、YAML、命令参数或日志。
- 将变更限制在目标部署和已有授权范围内；安装软件包、在回环地址之外开放端口、其他破坏性操作，或中断无关服务（例如不是 Agent 启动的栈）之前，先补齐缺少的授权。
- 路由预览执行路由信号推理，但不执行后端生成。route probe 是到达所选后端的端到端检查。
- 以正在运行的 Router 的发现、schema 和 OpenAPI 响应作为其已安装版本的权威来源。

更深入的配置工作，请继续阅读[配置契约](configuration-contract)和[配置工作流](configuration-workflows)。模型和 Mixture-of-Models 评估请使用 [Agent 评估循环](../benchmarking/agent-evaluation-loop)。

## 维护 Skill

唯一编写源位于 [`tools/agent/skills/vllm-sr-agent-operations/`](https://github.com/vllm-project/semantic-router/tree/main/tools/agent/skills/vllm-sr-agent-operations)，包含可选参考文档。修改这些文件后运行 `make agent-skill-sync`，不要直接修改公开副本。在 Skill 及每份参考文档中使用绝对 URL，让任一副本都可独立安装。生成器调整公开 Skill 名称、检查链接文档，并将相对链接转换为绝对 URL。源文件与生成文件一同提交后，网站直接发布这些静态文件；远程 Agent 无须检出仓库即可加载参考文档。

`make agent-skill-check`、pre-commit 和 `make harness-check` 会检查生成文件缺失或过期。仓库与网站由此共享同一套工作流，同时保留各自的 Skill 名称和安装路径。

Skill 应与它描述的 CLI 行为一起修改，并在发布前于一台全新主机上照着执行一遍：它的命令和预期输出就是 Agent 执行的契约。
