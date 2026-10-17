---
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/signal/learned/hallucination.md"
  outdated: false
---

# 幻觉检测信号 {#hallucination-signal}

## 概览 {#overview}

`hallucination` 对照请求携带的依据上下文（例如工具结果或检索文档）检查模型回答，并报告上下文不支持的主张。在 `routing.signals.hallucination` 下定义其规则。

该族为学习型：依赖 `global.model_catalog.modules.hallucination_mitigation.hallucination_model` 下的幻觉检测器（默认是 Vela 2.0 0.3B 的片段头，也可以是 Vela Halu），并在[模型运行时](../../../model-runtime/guides/hallucination)中执行。NLI 解释器已退役。

## 主要优势 {#key-advantages}

- 将检测不受支持的主张与 Router 对此采取的动作分开。
- 将 detected、not-detected 与 unavailable 作为不同状态报告，因此无法检查的回答绝不会被读成已核验。
- 像其他规则一样，把检查范围限定在请求解析到的配方。
- 为每条规则记录一条路由回放结果，包含判定、置信度、片段数量以及插件应用的动作。

## 解决什么问题？ {#what-problem-does-it-solve}

即使给了证据，模型仍可能答出范围之外的内容：错误数字、编造的名字、上下文从未陈述的细节。在绑定到 `hallucination` 插件内检查回答，会把检测与执行绑在一起：检查只在启用插件的地方运行，结果只对该插件可见，失败或无法检查看起来与干净回答一样。

`hallucination` 把检查作为响应阶段观察发布在 `hallucination:<name>` 键下，形状与路由回放和插件读取其他信号时相同。

## 何时使用 {#when-to-use}

在以下情况使用 `hallucination`：

- 依据工具结果或检索上下文的回答必须在信任前被检查
- 决策的 `hallucination` 插件应在证据上执行，而不是产生证据
- 路由回放应记录回答是否被检查、结果如何，无论插件是否据此行动
- 不受支持的主张应连同其所依据的片段一起报告

## 配置 {#configuration}

```yaml
routing:
  signals:
    hallucination:
      - name: ungrounded_claims
        description: Detect claims the grounding context does not support.
```

规则自身没有阈值：检测器在 `hallucination_model` 上的 `threshold`、`min_span_length` 与 `min_span_confidence` 决定什么算不受支持的片段，检测器找到一个时规则匹配。早期版本可以用 `use_nli` 请求片段级 NLI 解释；该解释器已退役，`vllm-sr config migrate` 会移除这一设置。

### 阶段 {#stage}

幻觉规则在响应阶段观察：它检查模型回答，因此只有模型回答后才存在。它不是决策输入。决策在请求路由时、模型回答前选定，因此在规则中或通过投影直接读取该规则的决策会在配置加载时被拒绝。观察由请求所选决策的 `hallucination` 插件消费：匹配时应用 `hallucination_action`，对没有可对照检查内容的回答应用 `unverified_factual_action`。

仅当请求阶段的 [fact-check](./fact-check) 信号表明提示提出值得锚定的主张、且请求携带了可对照的上下文时，检测器才有东西可检查：

- 回答有依据上下文并已检查：`detected` 或 `not_detected`，检测器置信度在 `hallucination:<name>` 下
- 提示需要锚定，但请求没有携带工具结果或检索上下文：unavailable，通过 `SignalErrors` 报告为 `hallucination_context_unavailable`，这是插件的 `unverified_factual_action` 所作用的情况
- 检测器失败或从未供给，或响应没有可检查的文本：unavailable，报告为 `hallucination_evaluation_failed`
- 提示没有提出值得锚定的主张：规则不适用且不发布；路由回放将其记录为 `not_applicable`

使用 `x-vsr-debug` 时，`x-vsr-matched-hallucination` 请求头携带匹配的规则。

流式回答在流结束后检查，并以 `enforcement: not_enforced_streaming` 代替动作记录。此时字节已到达客户端，因此 `hallucination` 插件不运行，`hallucination_action` 与 `unverified_factual_action` 都不适用。从未到达终端回答的流不检查，也不记录任何内容。

声明规则就足以为本配方供给检测器，即使没有决策启用插件。决策的 `hallucination` 插件在未声明规则时运行会在加载时报告：插件随后自己对回答分类，这是兼容路径。

## Vela 2.0 {#vela-20}

把 `hallucination_detector` 绑定到 Vela 2.0 部署后，检测器会就回答向模型提出它内置的幻觉问题（由其路由片段头回答），
请求与依据上下文作为该问题状态中的其他部分：

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        device: cpu
    bindings:
      hallucination_detector:
        deployment: vela2
        contract: token_spans.v1
```

模型读取完整回答并使用自身校准后的阈值，因此检测器的 `threshold` 只作用于 Vela 1.0 Halu；
`min_span_length` 与 `min_span_confidence` 对两者的片段都生效。检查在响应阶段进行，使用该阶段对该部署的调用。
同一个部署还可以回答请求的 [`decision`](decision.md) 与 [`pii`](pii.md#vela-20) 问题。

## 仅提供整体判断的模型 {#verdict-only-decision-models}

若使用通用判断模型进行整体检查，将 `routing.model_bindings.hallucination_detector` 绑定到该部署，并设置 `contract: decision.v1`。`hallucination` 任务使用 `noul` 问题，将请求、依据上下文和回答作为独立状态部分。结果供响应观察与插件使用，不会虚构主张位置。`threshold` 作用于该概率；片段过滤参数只作用于实际返回的片段。

`token_spans.v1` 绑定要求原生片段能力。两条路径都要求完整的准入输入和依据上下文；失败或未完成的检查是 unavailable，不能证明回答有据可依。

## 依赖与限制 {#dependencies-and-limitations}

检测器通过 `global.model_catalog.modules.hallucination_mitigation.hallucination_model` 处理回答与提供的依据上下文。它可以识别上下文不支持的文本；没有权威证据时无法确立真相，没有可对照上下文的回答报告为 unavailable，而不是干净。完整示例见：
[`config/fragments/signal/hallucination/grounded-answer.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/hallucination/grounded-answer.yaml)。
