---
is_mtpe: false
title: 模型推理
description: 继承、复用或定义模型推理契约，并从路由决策中选择其 mode 和 effort。
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
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

在控制面板中，打开 **Models → Backend models**，选择 **Connect model**，选择 Provider，然后选择 **Advanced settings → Reasoning family**。Backend models 页面上的 **Reasoning families** 标签页显示可用 ID 及其原生参数。

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

Canonical v0.5 没有用户编写的全局家族注册表。旧的 `providers.defaults.reasoning_families` 加上按模型的 `reasoning_family` 组合会迁移到 `providers.models[].reasoning`：内置 ID 变为 `family` 引用，而自定义定义会被内联复制到每个使用它们的自定义 Model 上。

在控制面板中使用 **Add manually** 编写内联家族。将 **Built-in Catalog Model** 和 **Reasoning Family** 留空，然后填写 **Inline Reasoning** 字段。

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

## 在决策中选择推理 {#select-reasoning-in-a-decision}

Model 拥有家族后，配置其决策引用：

```yaml
recipes:
  - name: default
    decisions:
      - name: complex-request
        rules:
          operator: AND
          conditions:
            - type: complexity
              name: needs_reasoning
              label: hard
        modelRefs:
          - model: private-reasoner
            use_reasoning: true
            reasoning_mode: enabled
            reasoning_effort: high
```

effort 必须由家族列出，并且被该 Model 的每个 Provider 绑定允许。`reasoning_mode` 必须与 `use_reasoning` 一致。省略 effort 以使用家族默认值，或在需要一个有效的部署级默认值优先时设置 `providers.defaults.reasoning_effort`。该默认值适用于家族列出它的每个 Model，因此这些 Model 的 Provider 绑定也必须允许它；家族未列出它的 Model 继续使用家族默认值。

在控制面板决策编辑器中，先选择 Model。UI 从其有效家族派生 mode 和 effort 选项，并禁用 Model 或 Provider 不支持的控件。没有家族的自定义 Model 保持为透传模型，并且没有控制面板推理控件。透传模型的路由日志、指标和追踪不报告推理强度，其 `routing_decision` 日志行为 `reasoning_projected: false`。

## 为较小的输出上限关闭推理 {#turn-reasoning-off-for-small-output-limits}

后端会把推理计入请求的输出上限。决策开启推理时，输出上限较小的请求可能把上限全部用于思考，最终没有回答：Messages 响应只有 thinking 块，Chat 响应的 `content` 为 `null`，Responses 响应只有 reasoning 项。设置保留决策推理所需的最小上限：

```yaml
providers:
  defaults:
    reasoning_min_output_tokens: 512
```

- **未表明偏好且上限较小：关闭推理。** 未表明推理偏好、且输出上限低于该阈值的请求，会在模型推理关闭的情况下运行。输出上限指 Messages 的 `max_tokens`、Chat 的 `max_tokens` 或 `max_completion_tokens`，以及 Responses 的 `max_output_tokens`。
- **哪些算作偏好。** 任意类型的 Messages `thinking`；Chat 的 `reasoning_effort`（包括 `none` 和 `minimal`）、`reasoning_budget_tokens`，或 `chat_template_kwargs` 中的 `enable_thinking`、`thinking`、`reasoning_effort` 成员；Responses 的 `reasoning.effort` 或 `reasoning.summary`。单独的 Messages `output_config.effort` 不算，因为它设置的是整个回答的 effort。表明了偏好的请求无论上限多少都照常路由。
- **通过模型自己的控制关闭。** 后端收到模型推理家族的关闭控制，与 `use_reasoning: false` 相同。家族没有关闭控制的模型（例如始终开启的家族）保持推理。回退候选遵循同一规则。
- **其他情况不变。** 没有输出上限、或上限不低于阈值的请求保持决策的推理，thinking 照常返回。不设置或设为 `0`（默认值）时该规则关闭。
- **会报告。** `x-vsr-trace` 的 `reasoning` 成员为 `off;reason=output_budget`，Router Replay 记录的推理模式为 `off`。
