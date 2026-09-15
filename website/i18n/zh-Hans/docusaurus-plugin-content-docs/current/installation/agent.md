---
sidebar_position: 2
title: 使用 Agent 安装
description: 向 Agent 提供一条提示词，即可通过 CLI 和 Router API 完成 vLLM Semantic Router 的安装、配置与验证。
translation:
  source_commit: "aa8c4a7d17848ba060a509ea9917b380a1c25f7f"
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

这就是完整的引导提示词。它会指向公开、自包含的 <a href={AGENT_SKILL_PATH}>vLLM SR Skill</a>；安装细节保留在 Skill 中，而不是复制到每条提示词里。控制面板是可选的；需要验证控制面板或 Playground 时，可以一并交给 Agent。

## Agent 会做什么

Skill 会引导 Agent：

1. 检查主机、现有安装、容器运行时、加速器和可用模型端点，且不改动它们。
2. 在需要时安装最新发布的 dev CLI，先确认支持的命令，再通过 CLI 和目标 Router 的 schema、OpenAPI 逐步了解配置。
3. 为可用模型池创建或更新规范 YAML，并将凭据保留在环境变量中。需要内置配方时，使用 `vllm-sr recipe builtin list`、`export` 和 `init`。
4. 新部署先在本地校验，再启动并等待就绪。已有部署先校验和规划，再应用变更；更改监听器或提供方拓扑时，使用已获授权的部署重启流程。
5. 预览路由决策但不执行后端生成，再通过推理端点发送一次真实的端到端请求。
6. 留下配置路径、活动 revision、校验结果和路由证据，供人工复核。

如果已经知道模型端点 URL、路由目标或部署约束，请在同一条消息中告诉 Agent。否则，Agent 会尽量自行发现，并仅在需要做选择或获得许可时提问。

## 直接契约

Agent 使用的契约与 CLI 和控制面板相同。需要控制面板验证时，它会检查真实的服务响应和 Playground 流式输出。

| 用途 | CLI 或 Router 契约 |
| --- | --- |
| 发现操作 | `GET /api/v1?audience=agent&visibility=primary` |
| 检查某项操作 | `GET /openapi.json?path=...&method=...` |
| 发现配置 | `vllm-sr config schema` 或 `GET /api/v1/config/schema` |
| 发现内置配方 | `vllm-sr recipe builtin list` |
| 首次启动 | `vllm-sr config validate`，然后 `vllm-sr serve` 并检查就绪状态 |
| 规划已有部署的变更 | `vllm-sr config validate`，然后 `vllm-sr config plan` |
| 应用可热重载的变更 | `vllm-sr config apply`，执行前会重新规划 |
| 测试路由逻辑 | `vllm-sr route preview` |
| 测试完整数据路径 | `vllm-sr route probe` |

管理源提供健康检查、发现、配置和 OpenAPI。已路由的推理源单独提供 OpenAI 兼容请求。Agent 必须分别发现两者，而不能从其中一个推断另一个。

## 安全边界

- 将 API 密钥和 provider 凭据保留在环境变量中；不要把密钥值写入提示词、YAML、命令参数或日志。
- 将变更限制在目标部署和已有授权范围内；破坏性操作、对外暴露或中断无关服务前，补齐缺少的授权。
- 路由预览执行路由信号推理，但不执行后端生成。route probe 是到达所选后端的端到端检查。
- 以正在运行的 Router 的发现、schema 和 OpenAPI 响应作为其已安装版本的权威来源。

更深入的配置工作，请继续阅读[配置契约](configuration-contract)和[配置工作流](configuration-workflows)。模型和 Mixture-of-Models 评估请使用 [Agent 评估循环](../benchmarking/agent-evaluation-loop)。

## 维护 Skill

唯一编写源位于 [`tools/agent/skills/vllm-sr-agent-operations/`](https://github.com/vllm-project/semantic-router/tree/main/tools/agent/skills/vllm-sr-agent-operations)，包含可选参考文档。修改这些文件后运行 `make agent-skill-sync`，不要直接修改公开副本。生成器只调整公开 Skill 名称，并将相对参考链接转换为同站点的绝对 URL。源文件与生成文件一同提交后，网站直接发布这些静态文件；远程 Agent 无须检出仓库即可加载参考文档。

`make agent-skill-check`、pre-commit 和 `make harness-check` 会检查生成文件缺失或过期。仓库与网站由此共享同一套工作流，同时保留各自的 Skill 名称和安装路径。
