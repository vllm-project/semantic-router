---
translation:
  source_commit: "dc40c9a164b35316778c66982e6184d9f45cc97c"
  source_file: "docs/tutorials/algorithm/selection/decision.md"
  outdated: false
---

# Decision 选模算法

## 概览 {#overview}

`decision` 向判断模型询问：某个路由决策的 `modelRefs` 中，哪个模型应该回答当前请求。你用一句话描述每个候选模型；判断模型读取请求，为各候选给出概率，再由概率最高的模型回答。

## 主要优势 {#key-advantages}

- 通过读取整个请求的模型来选择后端。
- 默认复用 Router 的判断模型，选模无需额外加载模型。
- 为各候选报告概率，可在选模追踪中查看。
- 模型无法及时回答时，回退到第一个 `modelRef`。

## 解决什么问题？ {#what-problem-does-it-solve}

固定顺序不考虑请求内容，而让聊天模型选择后端需要一次生成往返和输出解析。判断模型可以在一次前向计算中回答同样的问题，直接给出概率，无需生成文本。

## 何时使用 {#when-to-use}

当一个决策有两个或更多候选模型、每个模型的优势都能用一句话描述，且正确选择取决于请求内容时，可以使用此算法。顺序始终不变时，优先使用 `static`；主要依据成本、延迟或负载选择时，优先使用 `multi_factor`。

## 配置 {#configuration}

未指定 `deployment` 时，由 Router 的判断模型选模，即 `global.model_catalog.system.decision_model`。默认模型为 Vela 2.0 0.3B，也可以[选择其他规模](../../../model-runtime/choose-a-model.md#choose-a-size)。这个模型已经负责请求中的内置信号和未指定部署的 `decision` 问题，因此选模不需要再加载一份模型：

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

选择器在决策匹配之后，通过独立调用访问同一部署。默认绑定可以选择 Vela 或 Decision 1.0/2.0；所选资源必须支持 choice 任务。选择器显式指定的部署会覆盖默认绑定。

若要让其他模型（例如 Decision 2.0）负责选模，声明一个 `model_runtime` 部署：

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

命名部署必须使用 `provider: model_runtime`。决策需要 2 到 255 个不同的 `modelRefs`，`candidates` 只能描述这些模型；没有提供候选描述时，使用模型配置中的描述。

模型未就绪、响应超时、过载或返回无效答案时，Router 会记录选模回退并使用第一个 `modelRef`。所选模型会通过 `x-vsr-selected-model` 响应头报告。Routing Preview 会向判断模型提出相同问题，并将其选择报告为所选模型，但不会调用后端。

模型与运行位置的选择方法见 [Decision 模型指南](/docs/model-runtime/guides/decisions)。
