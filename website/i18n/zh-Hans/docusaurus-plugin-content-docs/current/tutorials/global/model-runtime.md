---
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/global/model-runtime.md"
  outdated: false
---

# 模型运行时

内置模型在 `vllm-srun` worker 中运行。Router 通过模型服务接口使用任务能力；前端保持控制与请求入口，deployment 声明模型资源，副本池提供多个独立 worker。

## 配置部署与绑定

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
    system:
      decision_model:
        deployment: primary
```

命名部署集中保存产物、设备与输入策略；任务绑定引用部署。相同模型可以供多个任务使用，而不必为每个消费者加载一份。目录中仅存在条目不会自动启动未被使用的模型。

在 CLI 中，`vllm-sr serve MODEL` 使用 Router 模式；添加 `--engine`（`-e`）启动不要求 Router YAML 或 Chat 后端的 Engine 模式。`--platform cpu|cuda|rocm` 选择运行平台。模式由启动方式决定，Dashboard 用于管理模型、任务和副本。

## 副本、状态与输入

部署可配置多个副本，由运行时池根据就绪状态和队列负载分发。副本调度不改变 Router 的 signals → decisions → algorithm 语义。增加副本需要足够的计算与内存资源，不保证在同一张 GPU 上线性提速。

未就绪、失败或超时的模型任务按可用性策略处理。截止时间约束调用方等待，不保证正在进行的底层 forward 立即中止。长输入在少核 CPU 上可能超过请求预算，应按实际文本长度、并发和任务选择硬件与部署规模。

继续阅读[部署指南](../../model-runtime/deploy)、[选择模型](../../model-runtime/choose-a-model.md)与[输入限制](../../model-runtime/reference#long-inputs)。
