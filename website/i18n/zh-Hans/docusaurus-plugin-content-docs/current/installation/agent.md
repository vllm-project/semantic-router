---
sidebar_position: 2
title: 使用 Agent 安装
description: 向 Agent 提供一条提示词，即可通过 CLI 和 Router API 完成 vLLM Semantic Router 的安装、配置与验证。
translation:
  source_commit: "12c2aa4feb5c5d40d90104d8b09cade1facf0bf5"
  source_file: "docs/installation/agent.md"
  outdated: false
---

import CodeBlock from '@theme/CodeBlock'
import {
  AGENT_INSTALL_PROMPT,
  AGENT_SKILL_PATH,
} from '@site/src/data/installation'

# 使用 Agent 安装

将此提示词粘贴到可以使用终端、并能访问目标机器的编码 Agent 中，即可安装 vLLM Semantic Router：

<CodeBlock language="text">{AGENT_INSTALL_PROMPT}</CodeBlock>

这就是完整的引导提示词。它会指向公开、自包含的 <a href={AGENT_SKILL_PATH}>vLLM SR Skill</a>；安装细节保留在 Skill 中，而不是复制到每条提示词里。控制面板是可选的，不属于 Agent 工作流。

## Agent 会做什么

Skill 会引导 Agent：

1. 检查主机、现有安装、容器运行时、加速器和可用模型端点，且不改动它们。
2. 在需要时安装稳定版 CLI，然后发现正在运行的 Router 所支持的操作、配置 schema 和 OpenAPI 契约。
3. 为可用的模型池创建或更新 canonical YAML，并将凭据保留在环境变量中。
4. 在应用变更前先校验并规划。更改 listeners 或 provider 拓扑需要显式重启部署。
5. 在不调用模型的情况下预览路由决策，再通过已路由的推理端点发送一次真实的端到端请求。
6. 留下配置路径、活动 revision、校验结果和路由证据，供人工复核。

如果已经知道模型端点 URL、路由目标或部署约束，请在同一条消息中告诉 Agent。否则，Agent 会尽量自行发现，并仅在需要做选择或获得许可时提问。

## 直接契约

Agent 使用的契约与 CLI 和控制面板相同；它不会自动化控制面板 UI。

| 用途 | CLI 或 Router 契约 |
| --- | --- |
| 发现操作 | `GET /api/v1?audience=agent&visibility=primary` |
| 检查某项操作 | `GET /openapi.json?path=...&method=...` |
| 发现配置 | `vllm-sr config schema` 或 `GET /api/v1/config/schema` |
| 校验并规划 | `vllm-sr config validate`，然后 `vllm-sr config plan` |
| 应用可热重载的变更 | 使用已规划的 ETag 执行 `vllm-sr config apply` |
| 测试路由逻辑 | `vllm-sr route preview` |
| 测试完整数据路径 | `vllm-sr route probe` |

管理源提供健康检查、发现、配置和 OpenAPI。已路由的推理源单独提供 OpenAI 兼容请求。Agent 必须分别发现两者，而不能从其中一个推断另一个。

## 安全边界

- 将 API 密钥和 provider 凭据保留在环境变量中；不要把密钥值写入提示词、YAML、命令参数或日志。
- 在允许任何特权、破坏性、对外暴露或会中断服务的操作之前，先进行复核。
- 路由预览可以证明决策路径，但不会调用模型。route probe 才是到达所选后端的端到端检查。
- 以正在运行的 Router 的发现、schema 和 OpenAPI 响应作为其已安装版本的权威来源。

更深入的配置工作，请继续阅读[配置契约](configuration-contract)和[配置工作流](configuration-workflows)。模型和 Mixture-of-Models 评估请使用 [Agent 评估循环](../benchmarking/agent-evaluation-loop)。
