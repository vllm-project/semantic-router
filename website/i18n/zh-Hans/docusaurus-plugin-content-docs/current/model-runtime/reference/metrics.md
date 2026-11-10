---
title: 指标与安全
sidebar_label: 指标与安全
sidebar_position: 4
description: Router 前端和运行时导出的指标、如何解读传输与服务端时间，以及运行时的安全特性。
is_mtpe: true
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/reference/metrics.md"
  outdated: true
---

# 指标与安全 {#metrics-and-security}

Router 前端为它调用的每个部署导出指标，每个运行时也在 `GET /metrics` 上导出自己的指标。本页先列出两者，再列出运行时的安全特性。

## 前端指标 {#metrics}

router（9190 端口）：

| 指标 | 标签 | 含义 |
| --- | --- | --- |
| `vsr_model_runtime_ready` | `deployment` | 部署能答题期间为 1。 |
| `vsr_model_runtime_requests_total` | `deployment`、`outcome` | 按结果分的调用：`ok`、`timeout`、`unavailable`、`overloaded`、`rejected`、`failed`。 |
| `vsr_model_runtime_request_duration_seconds` | `deployment` | 到达 runtime 的调用延迟。 |
| `vsr_model_runtime_unknown_answers_total` | `deployment`、`reason` | 按原因分的未知答案。 |
| `vsr_model_runtime_restarts_total` | `deployment` | 托管进程的重启。 |

## 运行时指标 {#runtime-metrics}

每个运行时（`GET /metrics`）导出：按端点和状态分的 `vllm_srun_requests_total`、`vllm_srun_request_duration_seconds`、`vllm_srun_queue_duration_seconds`、`vllm_srun_forward_duration_seconds`、`vllm_srun_batch_rows`、`vllm_srun_batch_tokens`、`vllm_srun_bundle_tasks`、按模型和结果分的 `vllm_srun_result_cache_total`、`vllm_srun_queue_depth`、`vllm_srun_ready`、`vllm_srun_model_info`（按模型、revision、家族、引擎、加速器、设备、`dtype` 和 `numerics`），以及 `vllm_srun_model_memory_bytes`（模型加载后的权重占的字节数，降精度副本也算）。

## 安全 {#security}

- 模型仓库里的代码（`modeling_*.py` 之类）永不导入、永不运行，`trust_remote_code` 永不使用。模型代码是 runtime 或你装的插件的一部分。
- 内置模型的每个文件，加载前都按记录在案的 SHA-256 核对。符号链接只许指向装着该模型的缓存内部。
- 令牌来自 `HF_TOKEN` 或 Hugging Face 的 token 文件，绝不来自命令行参数，也绝不进日志。
- 请求文本永不进日志，指标不带任何请求内容。
- `vllm-sr serve MODEL` 默认把 runtime 发布在 `127.0.0.1`，除非传 `--host`。runtime 没有认证，只暴露在可信网络里。
