---
title: 在 Kubernetes 上运行
sidebar_label: 在 Kubernetes 上运行
sidebar_position: 3
description: 在 Router pod 内用 CPU 或 GPU 运行模型运行时，确定其模型卷的大小，并让多个 Router 共享一个 GPU 运行时。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/deploy/kubernetes.md"
  outdated: false
---

# 在 Kubernetes 上运行 {#run-on-kubernetes}

Router 镜像已经包含 CPU 运行时，因此托管的部署可以在任何集群中工作。要使用 GPU，请使用 ROCm 或 CUDA 版 Router 镜像，并给 Router pod 分配一块 GPU；要在多个 Router 之间共享 GPU 模型，请把运行时作为独立的 Deployment 运行，并把每个 Router 附加到它。

```bash title="Write the GPU image and limit into the chart's values"
vllm-sr serve --target kubernetes --platform rocm
```

## 在 Router pod 的 GPU 上运行模型 {#models-on-the-router-pods-gpu}

ROCm 版 Router 镜像 `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm` 包含带 ROCm 版 PyTorch 的运行时，`vllm-sr-cuda` 包含带 CUDA 版 PyTorch 的运行时：

1. 给 Router pod 分配一块 GPU（在其资源限制中设置 `amd.com/gpu` 或 `nvidia.com/gpu`）。
2. 在部署上设置 `device: rocm:0` 或 `device: cuda:0`，Router 就会自己在 GPU 上运行该模型。

`vllm-sr serve --target kubernetes --platform rocm|cuda` 会替你把镜像和 GPU 限制写入 chart 的 values。

## 模型卷 {#the-model-volume}

第一次启动时，运行时把 Router 使用的模型下载到其模型卷中（`/app/models`，即 chart 的 `persistence` 声明，默认 10 GiB）。

- 按你路由使用的模型确定声明的大小：仅 Vela Omni Mini 就占 4.3 GB。
- 在隔离网络的集群中，请先把模型下载到卷中（[故障排查](./troubleshooting.md#the-runtime-stays-in-loading-or-warming)）。

## ROCm 镜像中的 CPU 模型 {#cpu-models-in-the-rocm-image}

:::warning
如果 Router 的所有模型都在 CPU 上运行，请使用 CPU 镜像。
:::

- 在 ROCm 镜像中保留在 CPU 上的模型运行在其 ROCm 版 PyTorch 上。对于大多数 Vela 任务模型，这个构建无法通过运行时在加载时进行的“单独与批量回答一致”检查，因此它们像 GPU 模型一样批处理：并发请求在批处理窗口之后共享前向计算，答案可能因舍入而改变。
- ROCm 镜像的 PyTorch 没有 CPU LAPACK，因此带 gated delta rule 层的解码器（基于 Qwen3.5 的模型，例如 Decision 2.0 Lux-9B 或 Vela 2.0 4B）无法在其 CPU 上加载。运行时在启动时、加载权重之前就拒绝为它们设置 `device: cpu`，并说明原因。请在 GPU 上运行它们，或使用 CPU 镜像。

## 在多个 Router 之间共享运行时 {#share-a-runtime-between-routers}

要在多个 Router 之间共享 GPU 模型，请用同一个镜像把运行时作为独立的 Deployment 运行，并把每个 Router 附加到它的 Service。镜像中的 `vllm-srun` 命令负责启动它：

```yaml title="model-runtime.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-runtime
spec:
  replicas: 1
  selector:
    matchLabels: {app: model-runtime}
  template:
    metadata:
      labels: {app: model-runtime}
    spec:
      containers:
        - name: runtime
          image: ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest
          command: ["vllm-srun"]
          args: ["serve", "vllm-sr/d3-edge", "--device", "rocm:0", "--host", "0.0.0.0", "--port", "8100"]
          resources:
            limits: {amd.com/gpu: 1}
          ports:
            - {name: http, containerPort: 8100}
          readinessProbe:
            httpGet: {path: /health, port: http}
          livenessProbe:
            httpGet: {path: /health/live, port: http}
---
apiVersion: v1
kind: Service
metadata:
  name: model-runtime
spec:
  selector: {app: model-runtime}
  ports:
    - {name: http, port: 8100, targetPort: http}
```

把部署的 `endpoint` 设为 `http://model-runtime.<namespace>.svc.cluster.local:8100`，方法同[附加你自己运行的运行时](./connect-the-router.md#attach-to-a-runtime-you-run)。

:::tip
用 `/health` 作为就绪探针：只有当每个模型都加载完毕并通过自检后它才会成功。`/health/live` 在进程处理请求期间都会回答，无论其模型是否就绪。
:::
