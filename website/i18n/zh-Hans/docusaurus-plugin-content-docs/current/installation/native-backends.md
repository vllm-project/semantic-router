---
sidebar_position: 5
description: Candle、ONNX 与非 CGO 构建的原生后端能力与生命周期契约。
---

# 原生后端（Native Backends）

Semantic Router 使用原生 Rust 与 CGo 绑定来实现学习型分类器、基于 embedding 的信号、多模态 embedding 以及基于 MLP 的模型选择。Router 通过 `CurrentNativeBackendCapabilities` 暴露所选后端的运行时能力契约；调用方应以该契约为准，而不是通过 build tag 或包名来猜测是否支持某项能力。

## 后端选择

| 构建形态 | 后端名称 | 选择方式 |
|-------------|--------------|--------------------|
| 默认 CGO 构建 | `candle` | 未启用 `onnx` tag 且 CGO 开启。 |
| ONNX 构建 | `onnx` | 启用 `onnx` tag 且 CGO 开启。 |
| 非 CGO 或 Windows 构建 | `stub` | `CGO_ENABLED=0` 或在 Windows 上构建。 |

示例：

```bash
# Default Candle-backed build from the Go router module.
cd src/semantic-router
go build ./cmd

# ONNX-backed build.
cd src/semantic-router
go build -tags=onnx ./cmd

# Non-CGo stub build for environments where native bindings are unavailable.
cd src/semantic-router
CGO_ENABLED=0 go build ./cmd
```

## 运行时能力（Runtime capabilities）

| 能力 | `candle` | `onnx` | `stub` |
|------------|----------|--------|--------|
| 统一批量分类（Unified batch classification） | Yes | No | No |
| LoRA 批量分类（LoRA batch classification） | Yes | No | No |
| 批量 embedding（Batched embedding） | Yes | Yes | No |
| 多模态 embedding（Multimodal embedding） | Yes | No | No |
| 模态路由（Modality routing） | Yes | No | No |
| MLP selector | Yes | No | No |
| 显式重置（Explicit reset） | No | No | No |

当所选后端未声明支持“统一批量分类”或“LoRA 批量分类”时，Router 会在分类层尽早失败。不要因为 Go 包能编译就假设 ONNX 或非 CGO 构建具备与 Candle 一致的分类器行为。

## 生命周期预期（Lifecycle expectations）

当前没有任何后端声明支持 `explicit_reset`。在部署与热加载规划中，应将原生模型状态视为进程级别所有：

- 若要切换后端，或进行需要干净原生状态的模型家族变更，优先使用进程重启
- 控制面在决定是否启用后端特定功能时，应以运行时能力输出为准
- ONNX 部署应限制在其声明支持的能力范围内：在 ONNX 分类器达到一致性之前，主要是批量 embedding

后端生命周期重置支持与更深层的 ONNX 分类器一致性，会作为原生绑定工作流的架构债务持续跟踪。

