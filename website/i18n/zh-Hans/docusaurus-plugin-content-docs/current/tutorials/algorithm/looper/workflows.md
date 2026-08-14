---
translation:
  source_commit: "8fc0fb9c"
  source_file: "docs/tutorials/algorithm/looper/workflows.md"
  outdated: false
---

# Router Flow

## 概览

`workflows` 是 Router Flow 的一种 **looper** 算法：单个模型名称可以在
OpenAI 兼容 API 背后运行一个有界的微智能体工作流。

它与 `config/fragments/algorithm/looper/workflows.yaml` 保持一致。

运行时还支持通过 `global.integrations.looper.flow.model_names` 设置直接调用的
Flow 模型 slug。内置默认值是 `vllm-sr/flow`。对 Flow 的直接调用仅评估
`algorithm.type=workflows` 的决策；它们不会悄然回退到普通的单模型路由。

## 主要优势

- 以一个模型名称 `vllm-sr/flow` 暴露多步骤智能体工作流。
- 明确工作模型边界：动态规划器只能使用决策的 `modelRefs`。
- 同时支持静态角色计划和由动态规划器生成的工作流。
- 记录包含计划、工作模型步骤、响应、失败模型和用量的 Flow 追踪信息。

## 它解决什么问题？

有些请求需要编排，而不是单步路由决策：拆分任务，让多个工作模型分别完成
有针对性的工作，验证或协调输出，并通过同一个聊天补全 API 返回一个最终答案。
`workflows` 将这种编排变为由路由器负责的策略，同时将公开模型接口精简至
`vllm-sr/flow`。

## 何时使用

- 一条路由应公开单个模型名称，但运行一个有界的微智能体流程。
- 工作模型池应来自决策的 `modelRefs`。
- 对于可预测的任务，你希望使用低延迟的静态模板。
- 对于难度更高的推理、编码或验证任务，你希望使用由动态规划器生成的工作流。

## 配置

注册直接模型 slug：

```yaml
global:
  integrations:
    looper:
      endpoint: http://localhost:8899/v1/chat/completions
      max_response_bytes_mb: 32 # 可选；限制单个上游响应正文的大小（默认值为 32 MiB）
      flow:
        model_names:
          - vllm-sr/flow
        state:
          store_backend: file
          ttl_seconds: 1800
          file:
            directory: .vllm-sr/flow-state
```

配置动态 Flow 决策：

```yaml
routing:
  decisions:
    - name: coding_flow
      output_contract: Preserve any explicit output format exactly.
      modelRefs:
        - model: openrouter/gemini-pro
        - model: openrouter/deepseek
        - model: qwen/qwen3.6-rocm
      algorithm:
        type: workflows
        workflows:
          mode: dynamic
          planner:
            model: qwen-coordinator
            max_completion_tokens: 2048
          max_steps: 6
          max_parallel: 3
          round_timeout_seconds: 90
          min_successful_responses: 2
          on_error: skip
```

`output_contract` 是决策作用域内的提示文本。对于应统一应用于静态 Flow、
动态 Flow、Fusion 和 ReMoM 的基准测试或应用格式要求，应使用它，而不要将
特定于任务的提示硬编码到算法中。对于类型化、可由路由器执行的规范化和
后处理（例如选项提取、终止动作 JSON 规范化或引用解引用），请使用
`output_contract_spec`。提取默认要求与 `content` 完全匹配；仅当决策明确允许
使用范围更宽的解析器时，才使用 `extract.sources` 或
`extract.mode: json_object`。

规划器模型是控制平面模型。它不需要出现在 `modelRefs` 中。工作模型调用仅限于
`modelRefs`；如果规划器指定了该列表之外的模型，执行器会拒绝该计划。

静态模式使用显式角色计划。每个角色模型都必须位于决策的 `modelRefs` 中。

```yaml
routing:
  decisions:
    - name: static_flow
      modelRefs:
        - model: qwen-worker
        - model: deepseek-worker
      algorithm:
        type: workflows
        workflows:
          mode: static
          roles:
            - name: thinker
              models: [qwen-worker]
            - name: worker
              models: [deepseek-worker]
            - name: verifier
              models: [qwen-worker]
          final:
            model: qwen-worker
          max_steps: 3
          max_parallel: 1
          round_timeout_seconds: 90
          on_error: skip
```

## 参数

| 参数 | 类型 | 默认值 | 说明 |
| ------ | ------ | -------- | ------ |
| `model_names` | list[string] | `["vllm-sr/flow"]` | 触发 Flow 执行的直接请求模型 slug |
| `state.store_backend` | string | `file` | 待处理工具调用工作流的状态后端：`memory`、`file` 或 `redis` |
| `state.ttl_seconds` | int | `1800` | 待处理工具调用工作流状态的 TTL |
| `mode` | string | `static` | `static` 角色执行或由 `dynamic` 规划器生成的执行 |
| `template` | string | `micro_agent` | 静态工作流模板名称 |
| `roles` | list[object] | 静态模式必需 | 按顺序排列的静态角色，每个角色包含 `name`、`models`、可选的 `prompt`，以及由较早角色 ID 或智能体 ID 组成的可选 `access_list` |
| `final.model` | string | 首个工作模型响应 | 来自 `modelRefs` 的可选静态最终综合模型 |
| `final.prompt` | string | 内置综合提示 | 可选的静态最终综合指令 |
| `planner.model` | string | 动态模式必需 | 用于生成工作流计划的控制平面模型 |
| `planner.max_completion_tokens` | int | `2048` | 仅用于规划器 JSON 计划的最大补全 token 数 |
| `max_steps` | int | `3` | 规划器生成的工作流中可接受的最大步骤数 |
| `max_parallel` | int | `2` | 每个步骤的最大工作模型数 |
| `max_completion_tokens` | int | 请求默认值 | 工作模型和最终综合调用的最大补全 token 数 |
| `round_timeout_seconds` | int | 未设置 | 等待每个工作流步骤或最终综合的最长秒数 |
| `min_successful_responses` | int | 全部模型 | 达到此数量的工作模型成功后继续执行并行步骤 |
| `temperature` | float | 请求默认值 | 规划器、工作模型和综合调用的温度 |
| `include_intermediate_responses` | bool | `true` | 在响应追踪信息中包含 Flow 计划和工作模型输出 |
| `on_error` | string | `fail` | 工作模型出错时使用 `fail` 终止；当至少一个工作模型成功时，使用 `skip` 跳过失败的工作模型 |

## 工具与函数调用

Router Flow 为客户端保留常规的 OpenAI 兼容工具调用契约。像向单个模型发送请求
一样，将 `tools` 或旧版 `functions` 置于 `vllm-sr/flow` 请求中。

当工作模型或最终综合器返回 `tool_calls` 时，Flow 会：

1. 存储待处理的工作流状态，包括计划、已完成步骤的输出、当前智能体请求，
   以及该智能体的私有工具轨迹；
2. 使用 Flow 状态前缀重写每个 `tool_call_id`，并将工具调用返回给客户端；
3. 当客户端在下一个请求中发送匹配的末尾 `tool` 消息时，取用该状态；
4. 将这些工具结果路由回发起请求的确切工作模型或最终智能体，而不重放无关的
   工作模型；
5. 继续该智能体的工具循环，直到它生成内容，然后恢复剩余的工作流。

每个工作模型都有自己的消息历史。后续步骤的 `access_list` 仅公开先前步骤或
先前智能体的输出，而不会公开其他工作模型的原始工具调用或工具结果轨迹。省略
`access_list` 会公开之前所有步骤的输出；将它设置为 `[]` 会使该步骤与先前输出
隔离。使用 `solver` 之类的角色 ID 可公开该角色的所有输出，使用
`solver:1:deepseek-worker` 之类的智能体 ID 则只公开并行角色中一个工作模型的
输出。同一个智能体 ID 也会作为 `flow.steps[].responses[].agent_id` 发出，
前提是启用了 `include_intermediate_responses`。

对于本地单进程开发，使用 `memory` 即可。对于本地重启，请使用 `file`。对于
多副本部署，请使用 `redis`，这样无论哪个路由器实例收到工具结果轮次，都能认领
该轮次。

## 请求

```json
{
  "model": "vllm-sr/flow",
  "messages": [{"role": "user", "content": "Debug this flaky test and propose a patch."}]
}
```

## 设计说明

Router Flow 有意保持面向用户的 API 精简。决策的 `modelRefs` 是工作模型池。
`algorithm.workflows` 描述如何编排该模型池，而不是定义第二个模型目录。
