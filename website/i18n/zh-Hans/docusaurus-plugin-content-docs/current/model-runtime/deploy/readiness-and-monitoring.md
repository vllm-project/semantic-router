---
title: 就绪与监控
sidebar_label: 就绪与监控
sidebar_position: 4
description: Router 何时等待其模型、模型未就绪时功能看到什么，以及显示运行状态的指标和 inventory。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/deploy/readiness-and-monitoring.md"
  outdated: false
---

# 就绪与监控 {#readiness-and-monitoring}

只有当 Router 为某份配置运行的模型都加载完毕后，它才会提供这份配置的服务。之后，请求绝不会等待一个无法回答的模型：未就绪的模型会让其信号成为未知，由每条规则的策略决定其含义。

```bash title="What is loading, and what is ready"
curl -s localhost:8080/startup-status
curl -s localhost:9190/metrics | grep '^vsr_model_runtime_ready'
```

## 模型未就绪时 {#when-a-model-is-not-ready}

### 启动时 {#at-startup}

启动时，Router 等待它管理的每个部署：其路由使用的任务模型（domain、PII、guard、safety、fact-check、feedback 和 hallucination 模型，以及你自己的分类器），以及 `systemone` 信号和 `systemone` 选择算法询问的决策模型。

- 等待期间，`/health` 会回答，`/ready` 返回 `503`。
- `/startup-status` 列出每个托管部署及其状态（`starting`、`loading`、`warming`、`ready` 等），`vllm-sr serve` 打印它仍在等待的部署。
- 等待（包括第一次下载）受 `VLLM_SRUN_READY_TIMEOUT` 限制（默认 10 分钟）。
- 如果模型加载失败或未能按时就绪，Router 不会启动；其日志和最后的启动状态（`phase: error`）会指出部署和原因（参见[故障排查](./troubleshooting.md#the-runtime-reports-failed)）。

:::note
由附加的运行时作为决策模型回答的信号，或通过选择某个 head 的绑定回答的信号，在准备阶段也需要模型元数据，因此请在 Router 之前启动这些运行时。
:::

自定义 `systemone` 问题可以指向离线的附加运行时，而不会阻塞启动。在当前模型卡就绪并支持所需任务之前，它们保持未知。恢复不会绕过能力检查：就绪但不兼容的模型卡会拒绝该任务。选择某个 head 的绑定仍然需要其特定的元数据；就绪状态不会悄悄把 PII span 检测器换成只判断是否存在的判断。

声明了但未使用的部署永远不会启动，也不会拖慢启动。

### 重新加载时 {#on-a-reload}

添加模型的配置重新加载或 `vllm-sr config apply` 会以同样的方式等待，期间之前的配置继续提供服务：

- `/ready` 保持 `200`，`GET /api/v1/config/hash` 把这次激活报告为 `pending`。
- 新配置在其模型就绪后开始提供服务。
- 如果有模型加载失败，这次重新加载会被拒绝，之前的配置继续提供服务。

### 服务期间 {#while-serving}

| 情形 | 功能看到的结果 |
| --- | --- |
| 附加运行时的决策模型仍在下载或加载 | 未知 |
| 答案在功能的超时之后才到达 | 未知 |
| 运行时过载 | 未知 |
| 运行时进程崩溃 | 在 Router 重启它之前为未知（退避时间从 1 秒到 60 秒） |
| 某个进程的所有模型都加载失败 | 未知；Router 以同样的退避重启该进程，直到模型加载成功 |
| 输入长于部署允许的长度且 `overflow: reject` | 该输入返回错误 |

未知的信号不匹配。决策的 `rules.on_unknown` 选择未知信号对该路由意味着什么（默认 `no_match`，或 `fail_request`），guard 和 PII 模块有 `on_error: allow | block`。`systemone` 选择算法回退到 `modelRefs` 中的第一个模型。

## 查看运行状态 {#check-what-is-running}

- Router 在其指标端口上导出 `vsr_model_runtime_ready{deployment="..."}`（部署能回答时为 1）、`vsr_model_runtime_restarts_total`、按结果统计的 `vsr_model_runtime_requests_total`，以及按原因统计的 `vsr_model_runtime_unknown_answers_total`。
- Router 的管理 API 列出每个部署及其进程、状态、重启次数和它提供的模型（标签、设备、引擎参数）：`curl -s localhost:8080/api/v1/inventory/model-runtime`。
- 附加的运行时直接回答 `GET /v1/models` 和 `GET /health`。

[指标参考](../reference/metrics.md)列出了每个指标，[配置参考](../reference/configuration.md#router-managed-runtimes)列出了改变托管运行时的命令、套接字目录和模型缓存的环境变量。
