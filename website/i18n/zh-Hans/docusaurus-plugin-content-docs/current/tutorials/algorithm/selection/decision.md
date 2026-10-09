---
translation:
  source_commit: "dc40c9a164b35316778c66982e6184d9f45cc97c"
  source_file: "docs/tutorials/algorithm/selection/decision.md"
  outdated: false
---

# Decision 选模算法

## 概述

`decision` 向决策模型询问当前路由决策的哪个 `modelRef` 应该回答请求。为每个候选编写一句描述，模型读取请求后为各候选给出概率，由概率最高的候选回答。

## 主要优势

- 使用理解完整请求的模型选择候选。
- 默认复用 Router 的决策模型，无需为选模加载另一个模型。
- 选择跟踪中可以查看每个候选的概率。
- 模型无法及时回答时，回退到第一个 `modelRef`。

## 解决什么问题？

固定顺序忽略了请求内容，而让 Chat 模型选择需要一次生成调用及输出解析。决策模型可以在一次推理中直接给出概率，无需生成文本。

## 何时使用

当一个决策有至少两个候选，且可以分别用一句话描述其优势，而正确选择取决于请求内容时使用。顺序固定时优先使用 `static`；按成本、延迟或负载选择时使用 `multi_factor`。

## 配置

未设置 `deployment` 时，选择由 `global.model_catalog.system.decision_model` 指向的 Router 决策模型完成，默认为 Vela 2.0 0.3B，也可以[选择其他规模](../../../model-runtime/choose-a-model#choose-a-size)。它已经负责内置信号和未指定部署的 Decision 问题，因此选模无需加载第二份模型：

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        device: rocm
    system:
      decision_model:
        deployment: primary

routing:
  signals:
    decision:
      - name: request_kind
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - key: code
              description: Writing, reviewing or debugging code
            - key: chat
              description: Anything else
  decisions:
    - name: code-route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: decision
            name: request_kind
            label: code
      modelRefs:
        - model: qwen3-8b
          use_reasoning: false
        - model: qwen3-32b
          use_reasoning: true
      algorithm:
        type: decision
        decision:
          instructions: Which model should answer this request?
          candidates:
            qwen3-8b: Fast general model for routine code
            qwen3-32b: Strong reasoning model for hard code
          timeout_ms: 1000
```

选择器在决策匹配之后向同一个部署发起独立调用。默认绑定可以选择 Vela 或 Decision 1.0/2.0，但选定资源必须支持 choice 任务。在选择器上显式指定部署可以覆盖默认绑定。

要让另一个模型（例如 Decision 2.0）选择候选，先声明 `model_runtime` 部署：

```yaml alternative
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B

routing:
  signals:
    decision:
      - name: request_kind
        deployment: decision-kai
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - key: code
              description: Writing, reviewing or debugging code
            - key: chat
              description: Anything else
  decisions:
    - name: code-route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: decision
            name: request_kind
            label: code
      modelRefs:
        - model: qwen3-8b
          use_reasoning: false
        - model: qwen3-32b
          use_reasoning: true
      algorithm:
        type: decision
        decision:
          deployment: decision-kai
          instructions: Which model should answer this request?
          candidates:
            qwen3-8b: Fast general model for routine code
            qwen3-32b: Strong reasoning model for hard code
          timeout_ms: 1000
```

命名部署必须使用 `provider: model_runtime`。决策需要 2–255 个不同的 `modelRefs`，`candidates` 只能描述这些模型；未提供候选描述时，使用模型已配置的描述。

模型未就绪、响应超时、过载或返回无效答案时，Router 会记录选模回退并使用第一个 `modelRef`。选中的模型通过响应头 `x-vsr-selected-model` 报告。Routing Preview 会向决策模型提出同样的问题并报告选中的模型，但不调用 Chat 后端。

选择模型及运行位置，请参阅 [Decision 模型指南](../../../model-runtime/guides/decisions)。
