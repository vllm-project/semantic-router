---
translation:
  source_commit: "9fbd7d85183341c61c25dfc526160eb0599434f8"
  source_file: "docs/tutorials/algorithm/looper/workflows.md"
  outdated: false
---

# 路由工作流

## 概览

`workflows` 在一个 OpenAI 兼容模型名背后，运行有界的多步 Router Flow。

运行时也支持通过 `global.integrations.looper.flow.model_names` 使用直接 Flow 模型 slug。内置默认值是 `vllm-sr/flow`。直接 Flow 调用只评估 `algorithm.type=workflows` 的决策；它们不会静默回退到普通单模型路由。

## 主要优势

- 把多步智能体工作流暴露为一个模型名：`vllm-sr/flow`。
- 保持 worker 边界显式：动态规划器只能使用决策的 `modelRefs`。
- 同时支持静态角色计划和动态规划器生成的工作流。
- 记录包含计划、worker 步骤、响应、失败模型和用量的 Flow 追踪。

## 解决什么问题？

有些请求需要编排，而不是一步路由决策：拆分任务、让多个 worker 做针对性工作、验证或调和输出，再通过同一个 chat completions API 返回一个最终答案。`workflows` 把这种编排变成 Router 策略的一部分，同时把公开模型表面保持得像 `vllm-sr/flow` 一样小。

## 何时使用

- 路由应暴露单个模型名，但运行有界的微型智能体流程。
- worker 池应来自决策的 `modelRefs`。
- 希望为可预测任务使用静态低延迟模板。
- 希望为更难的推理、编码或验证任务使用动态规划器生成的工作流。

## 配置

注册直接模型 slug：

```yaml
global:
  integrations:
    looper:
      endpoint: http://localhost:8899/v1/chat/completions
      max_response_bytes_mb: 32 # optional; caps a single upstream response body (default 32 MiB)
      flow:
        model_names:
          - vllm-sr/flow
        state:
          store_backend: file
          ttl_seconds: 1800
          file:
            directory: .vllm-sr/flow-state
```

配置一条动态 Flow 决策：

```yaml
routing:
  decisions:
    - name: coding_flow
      description: Coordinate coding work through planned worker steps.
      priority: 100
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

`output_contract` 是决策范围的提示词文本。把它用于应同时作用于静态 Flow、动态 Flow、Fusion 和 ReMoM 的基准或应用格式要求，而不是把任务特定提示词硬编码进算法。使用 `output_contract_spec` 做类型化的路由器可执行归一化和后处理，例如 choice 提取、终端动作 JSON 归一化或引用解引用。提取默认精确匹配 `content`；仅当决策明确允许更宽的解析器时，才使用 `extract.sources` 或 `extract.mode: json_object`。

规划器模型生成控制计划。省略 `planner.model` 时，路由器按声明顺序选择首个满足完整规划请求要求的已分配 worker，
包括 JSON 输出能力和实际输出、上下文预算。扫描过程不调用模型。
显式指定的规划器保持原目标并接受相同阶段检查，失败时不会替换为其他模型；没有合格规划器时请求直接失败。
显式规划器可以是 worker `modelRefs` 之外单独配置的辅助模型，但必须有运营者分配的后端。
worker 调用始终限制在 `modelRefs` 内，执行器会拒绝包含范围外 worker 的计划。
规划器选择不会降低已配置的不同成功 worker 最小数量。

静态模式使用显式角色计划。每个角色模型都必须在决策的 `modelRefs` 中。

```yaml
routing:
  decisions:
    - name: static_flow
      description: Coordinate a fixed sequence of worker roles.
      priority: 100
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
|-----------|------|---------|-------------|
| `model_names` | list[string] | `["vllm-sr/flow"]` | 触发 Flow 执行的直接请求模型 slug |
| `state.store_backend` | string | `file` | 待处理工具调用工作流状态后端：`memory`、`file` 或 `redis` |
| `state.ttl_seconds` | int | `1800` | 待处理工具调用工作流状态的 TTL |
| `mode` | string | `static` | `static` 角色执行或 `dynamic` 规划器生成执行 |
| `template` | string | `micro_agent` | 静态工作流模板名 |
| `roles` | list[object] | static 必填 | 有序静态角色，每个含 `name`、`models`，可选 `prompt`，以及可选的更早角色 id 或智能体 id 的 `access_list` |
| `final.model` | string | 第一个 worker 响应 | 可选的静态最终合成模型，来自 `modelRefs` |
| `final.prompt` | string | 内置合成提示 | 可选的静态最终合成指令 |
| `planner.model` | string | 首个合格的已分配 worker | 可选的显式规划模型，用于生成工作流计划 |
| `planner.max_completion_tokens` | int | `2048` | 仅用于规划器 JSON 计划的最大补全 token 数 |
| `minimum_candidates` | int | 未设置 | 配方物化和上下文资格过滤后，决策 `modelRefs` 所需的最少不同模型数 |
| `max_steps` | int | `3` | 规划器可接受的最大工作流步数 |
| `max_parallel` | int | `2` | 每步最大 worker 模型数 |
| `max_completion_tokens` | int | 请求默认值 | worker 和最终合成调用的最大补全 token 数 |
| `round_timeout_seconds` | int | 未设置 | 每个工作流步骤或最终合成最多等待的秒数 |
| `min_successful_responses` | int | 全部模型 | 达到该成功 worker 数后即可继续并行步骤 |
| `temperature` | float | 请求默认值 | 规划器、worker 和合成调用的温度 |
| `include_intermediate_responses` | bool | `true` | 在响应追踪中包含 Flow 计划和 worker 输出 |
| `on_error` | string | `fail` | worker 出错时 `fail`，或在至少一个 worker 成功时 `skip` 失败的 worker |

每个静态角色和每个规划器生成的步骤都必须包含至少 `min_successful_responses` 个模型。无法满足配置法定人数的计划会被拒绝，而不是以静默降低的法定人数运行。

## 工具与函数调用

Router Flow 为客户端保留普通的 OpenAI 兼容工具调用契约。像对待单个模型一样，在 `vllm-sr/flow` 请求上发送 `tools` 或旧版 `functions`。

当 worker 或最终合成器返回 `tool_calls` 时，Flow：

1. 存储待处理工作流状态，包括计划、已完成步骤输出、当前智能体请求，以及该智能体的私有工具轨迹；
2. 用 Flow 状态前缀改写每个 `tool_call_id`，并把工具调用返回给客户端；
3. 在客户端发送匹配的尾随 `tool` 消息时，在下一次请求中消费该状态；
4. 把这些工具结果路由回恰好请求它们的那个 worker 或最终智能体，而不重放无关 worker；
5. 继续该智能体的工具循环，直到它产生内容，然后恢复剩余工作流。

每个 worker 有自己的消息历史。后续步骤的 `access_list` 只暴露先前步骤或先前智能体的输出，不暴露另一个 worker 的原始工具调用或工具结果轨迹。省略 `access_list` 会暴露所有更早步骤的输出；设为 `[]` 则使该步骤与先前输出隔离。使用角色 id（例如 `solver`）可暴露该角色的全部输出，或使用智能体 id（例如 `solver:1:deepseek-worker`）只暴露并行角色中的一个 worker。启用 `include_intermediate_responses` 时，同一智能体 id 会作为 `flow.steps[].responses[].agent_id` 发出。

本地单进程开发用 `memory` 即可。本地重启使用 `file`。多副本部署使用 `redis`，以便收到工具结果回合的任意路由器实例都能认领它。

## 请求

```json
{
  "model": "vllm-sr/flow",
  "messages": [{"role": "user", "content": "Debug this flaky test and propose a patch."}]
}
```

## 设计说明

Router Flow 有意保持面向用户的 API 很小。决策的 `modelRefs` 就是 worker 池。`algorithm.workflows` 描述如何编排该池，而不是第二份模型目录。

规划器和 worker 模型会按工作流计划收到请求派生内容。工具调用状态可以持久化到内存、文件或 Redis；请为这些内容选择合适的后端、TTL、认证和加密。完整示例见：
[`config/fragments/algorithm/looper/workflows.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/workflows.yaml)。
