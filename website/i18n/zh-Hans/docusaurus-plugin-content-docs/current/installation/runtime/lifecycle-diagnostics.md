---
title: 运维与故障排查
description: 检查就绪状态、限制模型并发并更新运行中的模型。
translation:
  source_commit: "dc7f402642a8b8ecec8218e2086a4c6f186ea406"
  source_file: "docs/installation/runtime/lifecycle-diagnostics.md"
  outdated: false
---

配置好[进程内模型](in-process.md)或[外部服务](external.md)后，使用本页进行检查。

## 检查启动状态 {#check-startup}

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
curl -fsS http://localhost:8080/startup-status
```

验证命令检查配置。启动时，Router 再加载或连接模型，并检查已启用功能所需的能力。GPU 内核编译可能使首次启动比后续请求耗时更长。

| 问题 | 检查项 |
| --- | --- |
| 模型无法加载 | 权重或 ONNX 文件是否完整，以及分词器、标签和挂载路径 |
| 引擎或设备不可用 | 镜像和主机是否匹配所选 CPU/GPU 运行环境 |
| 标签不匹配 | 模型标签顺序，以及规则的标签或映射文件 |
| 不支持嵌入层或维度 | 导出所需层，并与使用方或已存索引匹配 |
| 远程推理失败 | 服务地址、凭据、超时、响应格式和大小限制 |
| 置信度为 `null` | 结果没有模型分数；见[安全模型](safety.md#handle-failures-and-missing-scores) |

## AMD 启动问题 {#amd-startup-problems}

使用维护的 ROCm 镜像，确保 ORT 与 MIGraphX 库匹配。镜像包含 ORT 1.22.1 / MIGraphX 2.13，并设置 `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention`。使用该镜像时保留此设置；它会禁用 MLIR attention fusion。

| 错误或现象 | 处理方式 |
| --- | --- |
| SDPA 图中的 `IsNaN` 不受支持 | 使用兼容的标准 `onnx/model.onnx` 图 |
| GPU 嵌入缺少输入预算 | 将部署的 `input.max_tokens` 设为正数；见[嵌入模型](embeddings.md#amd-gpu) |
| 模型在 GPU 准备阶段失败 | 检查图和运行库；请求的 GPU 执行不会回退到 CPU |
| 首次启动耗时明显较长 | 为编译和每个所需嵌入层的预热留出时间 |

取消设置以下进程环境变量，改用 deployment 配置精度：

```text
ORT_MIGRAPHX_FP16_ENABLE
ORT_MIGRAPHX_BF16_ENABLE
ORT_MIGRAPHX_FP8_ENABLE
ORT_MIGRAPHX_INT8_ENABLE
ORT_MIGRAPHX_MODEL_CACHE_PATH
```

任何非空值（包括 `0`）都会被拒绝，因为它们可能覆盖配置的精度或已编译模型。维护的镜像不设置这些变量。

## 限制推理并发 {#limit-concurrent-inference}

对于[进程内模型](in-process.md)示例中的 `email-risk-cpu` 部署，以下设置允许两个并发调用、八个排队请求，排队超时为一秒：

```yaml
global:
  model_catalog:
    admission:
      email-risk-cpu:
        max_concurrency: 2
        max_queue: 8
        queue_timeout_ms: 1000
        on_overflow: shed
```

`shed` 拒绝超出容量的请求，`wait` 等待队列空位，`fail_open` 在满载时绕过限制。`wait` 要求队列大小非零。不配置 admission 时，不施加并发准入限制。共享同一个模型的调用也共享其容量；请求期限包含排队时间。

## 更新运行中的模型 {#update-a-running-model}

将新的模型 revision 放入新目录，更新配置，再通过 Dashboard 或现有管理流程重载。Router 在激活前准备新模型。准备失败时，当前配置继续运行；旧模型资源会在已有请求完成后释放。

Dashboard 更改可能已保存但尚未激活。对于返回 `202` 的更新，通过 Dashboard 轮询 `GET /api/router/api/v1/config/hash`，等待 `active_runtime_hash` 与该次更新的 `generated_runtime_hash` 相同。第一项更新尚未激活时，另一项写入会返回 `409`。

知识库更新会保留旧资产版本，供仍在运行的读取使用。旧版本保留在磁盘上，目前不自动清理。请求和响应详情见[管理 API 参考](/zh-Hans/docs/api/apiserver)。
