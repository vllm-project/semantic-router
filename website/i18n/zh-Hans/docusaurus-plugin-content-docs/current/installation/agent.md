---
is_mtpe: false
sidebar_position: 2
title: 使用 Agent 安装
description: 向编码 Agent 提供一条提示词，即可通过 CLI 和 Router API 在 CPU 或 GPU 主机或 Kubernetes 集群上安装、配置并验证 vLLM Semantic Router，之后再用它维护、排障和调优。
translation:
  source_commit: "8e8840126a20bab8b92c38aafac91b7589eda187"
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

<a href={AGENT_SKILL_PATH}>vLLM SR Skill</a> 带领 Agent 从一台装有 Docker 的主机走到一条经过验证的路由请求，之后依然有用：同一个 Skill 负责升级栈、把配置迁移到 v0.5、诊断故障和调优路由。控制面板和 Playground 检查按需进行。请附上要使用的模型端点，例如本地的 Ollama 或 vLLM 服务、托管 API，以及部署约束；改动栈以外的任何东西之前，Agent 都会先询问。

Router 启动后，[接入 Agent Harness](agent-harness)。

:::note 发布渠道
Skill 与本文档一样跟随 `main`。稳定版能读取 v0.5 配置时（`vllm-sr config schema` 报告 `"config_version": "v0.5"`），它安装稳定版，否则安装开发渠道，并告诉你装的是哪一个。
:::

## Skill 覆盖的任务

Skill 是一个简短的路由，指向六类任务。每类任务都写明前提条件、包含具体步骤的参考文档、输出必须包含什么（退出码、JSON 字段、响应头的值）、某一步失败时怎么做，以及何时停下来询问你。

| 任务 | Agent 做什么 |
| --- | --- |
| 部署 | 安装 CLI，编写并校验配置，在 Docker（CPU、AMD 或 NVIDIA）或 Kubernetes（Helm 或 Operator）上启动栈并检查 |
| 配置 | 用 `config validate`、`config plan` 和 `config apply` 修改运行中的栈的模型、信号、决策、插件、配方或入口 |
| 验证 | 预览路由，探测完整路由路径，读取响应头，并检查决策模型是否就绪 |
| 维护 | 升级，运行 `vllm-sr config migrate` 迁移到 v0.5，回滚配置版本，备份与恢复，轮换密钥，查看日志与健康状态，移除栈 |
| 排障 | 把症状对应到检查方法和修复方法，包括 v0.5 的加载错误、未就绪的决策模型、响应头信任和图像路由 |
| 优化 | 选择决策模型及其硬件，调优配方、缓存和评判上限，并用一次简短的 sr-bench 运行衡量质量、延迟和成本 |

## 首次安装

1. **预检**，不做任何改动：Docker 访问权限与内存、Python 及其 `venv` 支持、可用磁盘和端口、AMD 或 NVIDIA GPU 设备，以及已在运行的 `vllm-sr` 栈。
2. **选择路径**：发布渠道；平台（主机有相应 GPU 时用 `--platform rocm` 或 `cuda`，GPU 要留作他用时用 `--platform cpu`）；standalone 模式，需要 Envoy 时用 `--gateway extproc`；Docker 或 Kubernetes；以及模型端点，并按 Router 容器访问它的方式检查连通性。
3. **安装 CLI**：使用 curl 安装脚本，不启动栈，并检查它能读取 v0.5 配置。
4. **编写配置**：面向你的模型，推理 API 绑定到 `127.0.0.1`，包含一条关键词路由和一条由决策模型判断的路由（一条 `domains` 规则），然后用 `vllm-sr config validate` 校验。
5. **启动栈**：运行 `vllm-sr serve --config config.yaml`，绝不进入会等待人工操作的设置模式。首次启动会下载默认决策模型 vela3-fast，它在 CPU 上约占 7 GB 内存。
6. **按明确标准验证**：`vllm-sr status`、`GET /v1/models`、决策模型是否就绪、两条路由的 `vllm-sr route preview`、一条路由请求及其 `x-vsr-selected-decision`、`x-vsr-selected-model` 和 `x-vsr-response-path` 响应头，以及 `vllm-sr route probe`。在 GPU 主机上，它还会以引擎模式在 GPU 上运行一个决策模型。
7. **交付结果**：版本与渠道、配置路径、各端点、决策模型、控制面板创建第一个管理员的步骤，以及每项检查的结果。

每一步都可以安全地重复执行：已有的配置或正在运行的栈只会被验证，不会被替换。

## 直接契约

Agent 使用的契约与 CLI 和控制面板相同。

| 用途 | CLI 或 Router 契约 |
| --- | --- |
| 发现操作 | `GET /api/v1?audience=agent&visibility=primary` |
| 检查某项操作 | `GET /openapi.json?path=...&method=...` |
| 发现配置 | `vllm-sr config schema` 或 `GET /api/v1/config/schema` |
| 发现内置配方 | `vllm-sr recipe builtin list` |
| 首次启动 | `vllm-sr config validate`，然后 `vllm-sr serve --config config.yaml` |
| 检查栈 | `vllm-sr status`、`GET /ready` |
| 检查决策模型 | `GET /api/v1/inventory/model-runtime` |
| 规划已有部署的变更 | `vllm-sr config validate`，然后 `vllm-sr config plan` |
| 应用可热重载的变更 | `vllm-sr config apply`，执行前会重新规划 |
| 把配置迁移到 v0.5 | `vllm-sr config migrate` |
| 列出并恢复版本 | `vllm-sr config versions`、`vllm-sr config rollback` |
| 测试路由逻辑 | `vllm-sr route preview` |
| 测试完整数据路径 | `vllm-sr route probe` |
| 单独运行一个决策模型 | `vllm-sr serve ARTIFACT --engine`（引擎模式） |
| 衡量质量、延迟和成本 | `vllm-sr benchmark plan`、`vllm-sr benchmark run` |

管理源（本地栈的 8080 端口）提供健康检查、发现、配置和 OpenAPI。推理监听器（8899 端口）单独提供[受支持的推理协议](protocol-compatibility)。Agent 必须分别发现两者，而不能从其中一个推断另一个。

## 安全边界

- 将 API 密钥和 provider 凭据保留在环境变量中；不要把密钥值写入提示词、YAML、命令参数或日志。
- 将变更限制在目标部署和已有授权范围内；安装软件包、在回环地址之外开放端口、停止或升级你在运行的栈、删除卷或状态，或中断无关服务（例如不是 Agent 启动的栈）之前，先补齐缺少的授权。
- 路由预览执行路由信号，但不执行后端生成。route probe 是到达所选后端的端到端检查。
- 以正在运行的 Router 的发现、schema 和 OpenAPI 响应作为其已安装版本的权威来源。

更深入的配置工作，请继续阅读[配置契约](configuration-contract)和[配置工作流](configuration-workflows)；关于 v0.5，请阅读[迁移指南](migrate-to-v0-5)和 [System One](../system-one/overview)。模型和 Mixture-of-Models 评估请使用 [Agent 评估循环](../benchmarking/agent-evaluation-loop)。

## 维护 Skill

唯一编写源位于 [`tools/agent/skills/vllm-sr-agent-operations/`](https://github.com/vllm-project/semantic-router/tree/main/tools/agent/skills/vllm-sr-agent-operations)，包含其参考文档。修改这些文件后运行 `make agent-skill-sync`，不要直接修改公开副本。在 Skill 及每份参考文档中使用绝对 URL，让任一副本都可独立安装。生成器调整公开 Skill 名称、检查链接文档，并将相对链接转换为绝对 URL。源文件与生成文件一同提交后，网站直接发布这些静态文件；远程 Agent 无须检出仓库即可加载参考文档。

`make agent-skill-check`、pre-commit 和 `make harness-check` 会检查生成文件缺失或过期。另有两项测试保证 Skill 与实现一致：Router 用自己的配置解析器加载 Skill 中的每个 YAML 块；CLI 单元测试用 CLI 自己的参数解析器解析每条 `vllm-sr` 命令行，并检查正文提到的每个命令和选项都存在。

Skill 应与它描述的 CLI 行为一起修改，并在发布前于一台全新主机上照着执行一遍：它的命令和预期输出就是 Agent 执行的契约。
