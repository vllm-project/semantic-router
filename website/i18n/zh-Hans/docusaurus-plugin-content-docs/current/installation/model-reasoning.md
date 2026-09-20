---
title: 模型推理
description: 继承、复用或定义模型推理契约，并从路由决策中选择其 mode 和 effort。
translation:
  source_commit: "bef43a20a96e7b329d314473083be6aa4bad0f03"
  source_file: "docs/installation/model-reasoning.md"
  outdated: false
---

# 模型推理

推理家族告诉 Router 模型如何表示思考控制。它描述原生参数、支持的 mode、可选的 effort 阶梯、默认值和禁用值。所选 Provider 映射随后将该模型契约投影到 Provider 的请求协议上。

## 选择谁拥有家族

| 模型类型 | 配置 | 所有者 |
| --- | --- | --- |
| 目录支持 | 仅设置 `catalog`。 | 内置 Model Card 和 Provider 映射。 |
| 使用已知家族的自定义模型 | 设置 `reasoning.family`。 | 内置推理家族目录。 |
| 使用新家族的自定义模型 | 在 `reasoning` 下设置内联字段。 | 此模型声明。 |
| 自定义透传 | 省略 `reasoning`。 | 上游请求和后端行为。 |

目录支持的 Model 不能覆盖推理。自定义 Model 必须选择内置家族引用或内联定义，二者不能同时使用。

## 复用内置家族

```yaml
providers:
  models:
    - name: private-qwen
      provider_model_id: Qwen3-32B
      reasoning:
        family: qwen3
      backend_refs:
        - provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http
```

在控制面板中，打开 **Build → Models → Add Model**，选择 Provider，然后选择 **Advanced settings → Reasoning family**。Models 页面上的 **Built-in Reasoning Families** 表显示可用 ID 及其原生参数。

## 内联定义自定义家族

当没有匹配的内置家族时，将完整契约附加到自定义 Model：

```yaml
providers:
  models:
    - name: private-reasoner
      provider_model_id: private-reasoner-v2
      reasoning:
        type: reasoning_effort
        parameter: reasoning_effort
        activation_parameter: enable_thinking
        levels: [low, medium, high]
        default: medium
        modes: [enabled, disabled]
        default_mode: enabled
        disabled: none
      backend_refs:
        - provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http
```

Canonical v0.3 没有用户编写的全局家族注册表。旧的 `providers.defaults.reasoning_families` 加上按模型的 `reasoning_family` 组合会迁移到 `providers.models[].reasoning`：内置 ID 变为 `family` 引用，而自定义定义会被内联复制到每个使用它们的自定义 Model 上。

在控制面板中使用 **Manual setup** 编写内联家族。将 **Built-in Catalog Model** 和 **Reasoning Family** 留空，然后填写 **Inline Reasoning** 字段。

## 理解内联字段

| 字段 | 必需行为 |
| --- | --- |
| `type` | `chat_template_kwargs`、`reasoning_effort`、`reasoning_mode` 或 `top_level_reasoning_effort` 之一。 |
| `parameter` | 该家族控制的原生请求参数。与 `type` 一起必需。 |
| `levels` | 允许的 effort 名称。基于 effort 的类型必需。 |
| `default` | 默认 effort；设置时必须出现在 `levels` 中。 |
| `modes` | 来自 `enabled`、`disabled` 和 `adaptive` 的任何受支持值。 |
| `default_mode` | 设置 `modes` 时必需，并且必须出现在该列表中。 |
| `disabled` | 用于禁用推理的原生值；必须支持 `disabled` mode。 |
| `activation_parameter` | `reasoning_effort` 的单独开关参数；必须与 `parameter` 不同。 |
| `effort_flags` | 从 effort 名称到布尔参数的高级映射。它需要 `activation_parameter`，并且每个键都必须出现在 `levels` 中。 |

对于 `top_level_reasoning_effort`，`parameter` 必须是 `reasoning_effort`。仅 mode 的家族在决策中不接受 `reasoning_effort`。始终开启的家族不能用 `use_reasoning: false` 选择。

## 在决策中选择推理

Model 拥有家族后，配置其决策引用：

```yaml
routing:
  decisions:
    - name: complex-request
      rules:
        operator: AND
        conditions:
          - type: complexity
            name: needs_reasoning:hard
      modelRefs:
        - model: private-reasoner
          use_reasoning: true
          reasoning_mode: enabled
          reasoning_effort: high
          max_completion_tokens: 2048
```

effort 必须由家族列出，并且被该 Model 的每个 Provider 绑定允许。`reasoning_mode` 必须与 `use_reasoning` 一致。省略 effort 以使用家族默认值，或在需要一个有效的部署级默认值优先时设置 `providers.defaults.reasoning_effort`。

`max_completion_tokens` 是可选的按模型补全上限。设置后，Provider 派发会取客户端请求、决策插件 `request_params.max_tokens_limit`、此 ModelRef 值、任何 Looper 算法/阶段上限、`default_max_tokens: auto` 渲染后的剩余容量，以及未来的请求级 ledger 中的最严格值（最小值）。省略则保持现有行为。同一上限适用于普通路由和 Looper hop（包括 Confidence 升级），因此每个被选中的模型都在派发时独立封顶。合成上限低于 Responses API 最小值 16 时，派发会失败，而不是去掉该约束。

在控制面板决策编辑器中，先选择 Model。UI 从其有效家族派生 mode 和 effort 选项，并禁用 Model 或 Provider 不支持的控件。没有家族的自定义 Model 保持为透传模型，并且没有控制面板推理控件。
