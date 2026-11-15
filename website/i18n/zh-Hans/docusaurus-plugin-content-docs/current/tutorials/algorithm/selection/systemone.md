---
is_mtpe: false
translation:
  source_commit: "085b44d375765d4935a1e152a2d14d59c1e64b75"
  source_file: "docs/tutorials/algorithm/selection/systemone.md"
  outdated: false
---

# System One 选模算法

## 概述 {#overview}

`systemone` 向 System One 模型询问：当前路由决策的 `modelRefs` 中，哪个模型最适合回答请求？你用一句话描述每个候选模型；模型读取请求，为每个候选给出概率，再由概率最高的模型回答。

## 主要优势 {#key-advantages}

- 由能够理解整个请求的模型做选择。
- 默认复用 Router 的判断模型，无需为选模再加载一份模型。
- 为每个候选报告概率，可在选模轨迹中查看。
- 模型未能及时给出答案时，回退到第一个 `modelRef`。

## 解决什么问题？ {#what-problem-does-it-solve}

固定顺序不会考虑请求内容，而让 Chat 模型选模需要一次文本生成调用，还要解析输出。System One 模型只需一次前向计算即可回答同样的问题，直接返回概率，无需生成文本。

## 何时使用 {#when-to-use}

当一个 decision 有两个或更多候选模型，能够用一句话描述各自优势，而且合适的选择取决于请求内容时，使用此算法。如果选择顺序始终不变，优先使用 `static`；如果主要依据成本、延迟或负载选模，则使用 `multi_factor`。

## 配置 {#configuration}

未指定 `deployment` 时，使用 `global.model_catalog.system.decision_model` 指向的 Router 判断模型，默认为 Vela 3.0 vela3-fast，也可以[选择其他规模](/model-runtime/get-started/choose-a-model.md#choose-a-size)。该模型已经负责请求的内置信号和未指定部署的 `systemone` 问题，因此选模无需再加载第二份模型：

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
recipes:
  - name: default
    signals:
      systemone:
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
            - type: systemone
              name: request_kind
              label: code
        modelRefs:
          - model: qwen3-8b
            use_reasoning: false
          - model: qwen3-32b
            use_reasoning: true
        algorithm:
          type: systemone
          systemone:
            instructions: Which model should answer this request?
            candidates:
              qwen3-8b: Fast general model for routine code
              qwen3-32b: Strong reasoning model for hard code
            timeout_ms: 1000
```

选择器在 decision 匹配后，向同一部署发起独立调用。默认绑定可以选择 Vela 或 Decision 1.0/2.0，前提是所选资源支持 Choice 任务。选择器显式配置的部署会覆盖默认绑定。

为早期版本编写的配置把这个选择器称为 `decision`（`algorithm.type: decision` 和 `algorithm.decision` 块）。这些配置仍可加载，但会给出 `renamed_algorithm_type` 警告；`vllm-sr config migrate` 会把它们改写为 `systemone`。

要让另一个模型选模，例如 Decision 2.0，可以声明一个 `model_runtime` 部署：

```yaml alternative
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
recipes:
  - name: default
    signals:
      systemone:
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
            - type: systemone
              name: request_kind
              label: code
        modelRefs:
          - model: qwen3-8b
            use_reasoning: false
          - model: qwen3-32b
            use_reasoning: true
        algorithm:
          type: systemone
          systemone:
            deployment: decision-kai
            instructions: Which model should answer this request?
            candidates:
              qwen3-8b: Fast general model for routine code
              qwen3-32b: Strong reasoning model for hard code
            timeout_ms: 1000
```

命名部署必须使用 `provider: model_runtime`，或在使用[远程 System One API](../../../system-one/deploy/deployments#remote-system-one-apis) 时使用 `provider: systemone`；未设置自身 `timeout_ms` 的选择器使用部署的值。decision 需要 2–255 个不同的 `modelRefs`，`candidates` 只能描述这些模型；没有在其中提供描述的模型会使用自身配置的描述。

模型未就绪、响应超时、过载或返回无效答案时，Router 会记录选模回退，并使用第一个 `modelRef`。选中的模型会在 `x-vsr-selected-model` 响应头中返回。Routing Preview 会向 System One 模型提出同样的问题并展示它选择的模型，但不会调用后端。

关于选择模型及其运行位置，请参阅[选择部署](../../../system-one/deploy/deployments)。
