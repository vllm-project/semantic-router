---
title: VSR 决策追踪 Header
sidebar_label: VSR Header
translation:
  source_commit: "c7d360e"
  source_file: "docs/troubleshooting/vsr-headers.md"
  outdated: false
---

# VSR 决策追踪 Header

本文档描述了 VSR（Vector Semantic Router，向量 Semantic Router ）决策追踪 Header，这些 Header 会自动添加到成功的响应中，用于调试和监控目的。

## 概述

 Semantic Router 会自动添加响应 Header 以追踪 VSR 决策信息。这些 Header 帮助开发者和运维团队了解请求是如何被处理和路由的。

**Header 仅在以下情况下添加：**

1. 请求成功（HTTP 状态码 200-299）
2. 请求未命中缓存
3. VSR 在请求处理期间做出了路由决策

## 添加的 Header

### `x-vsr-selected-category`

**描述**：VSR 在分类期间选择的类别。

**示例值**：

- `math`
- `business`
- `biology`
- `computer_science`

**添加时机**：当 VSR 成功将请求分类到某个类别时。

### `x-vsr-selected-reasoning`

**描述**：是否确定对此请求使用推理模式。

**值**：

- `on` - 启用了推理模式
- `off` - 禁用了推理模式

**添加时机**：当 VSR 做出推理模式决策时（适用于自动和显式模型选择）。

### `x-vsr-selected-model`

**描述**：VSR 选择用于处理请求的模型。

**示例值**：

- `deepseek-v31`
- `phi4`
- `gpt-4`

**添加时机**：当 VSR 选择模型时（通过自动路由或显式模型指定）。

## 用例

### 调试

这些 Header 帮助开发者了解：

- VSR 将其请求分类到哪个类别
- 是否应用了推理模式
- 最终选择了哪个模型

### 监控

运维团队可以使用这些 Header：

- 追踪跨请求的类别分布
- 监控推理模式使用模式
- 分析模型选择模式

### 分析

产品团队可以分析：

- 请求分类准确性
- 推理模式有效性
- 按类别划分的模型性能

## 响应示例

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-selected-category: math
x-vsr-selected-reasoning: on
x-vsr-selected-model: deepseek-v31

{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "deepseek-v31",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "x^2 + 3x + 1 的导数是 2x + 3。"
      },
      "finish_reason": "stop"
    }
  ]
}
```

## 不添加 Header 的情况

以下情况不添加 Header：

1. **缓存命中**：当响应来自缓存时，不进行 VSR 处理
2. **错误响应**：当上游返回 4xx 或 5xx 状态码时
3. **缺少 VSR 信息**：当 VSR 决策信息不可用时（正常操作中不应发生）
