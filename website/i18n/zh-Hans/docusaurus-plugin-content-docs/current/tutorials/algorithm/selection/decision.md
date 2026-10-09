---
translation:
  source_commit: "9649d02f2582471642196bde2e086c34ccb9c88c"
  source_file: "docs/tutorials/algorithm/selection/decision.md"
  outdated: false
---

# 决策模型选择

## 概览 {#overview}

`decision` 问决策模型：一条路由决策的 `modelRefs` 里该由谁来答这个请求。每个候选你用一句话描述；模型读请求，给每个候选一个概率，最可能的那个答。

## 关键优势 {#key-advantages}

- 挑模型的模型读得到整个请求。
- 每个候选报一个概率，选择追踪里看得见。
- 模型答不及，随时退回第一个 `modelRef`。

## 解决什么问题 {#what-problem-does-it-solve}

固定顺序不看请求；让聊天模型来挑，又要一轮生成、又要解析输出。决策模型一趟前向就答完同一个问题，带概率，不生成文本。

## 什么时候用 {#when-to-use}

一条决策有两个以上候选、它们的长处你都能一句话说清、而选哪个取决于请求问什么——这时用它。顺序永远不变就用 `static`；选的是成本、延迟或负载，就用 `multi_factor`。

## 配置 {#configuration}

```yaml
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

部署必须用 `provider: model_runtime`。决策要 2 到 255 个不同的 `modelRefs`，`candidates` 只能描述这些模型；没写描述的模型用自己配的描述。模型没就绪、答晚了、过载了或答得无效，router 记一次选择回退，用第一个 `modelRef`。选中的模型报在 `x-vsr-selected-model` 响应头里。

选模型和跑在哪，见[决策模型](../../../model-runtime/guides/decisions.md)。
