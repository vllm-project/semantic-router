---
translation:
  source_commit: "bce357c513f391824e8320267d03977794c20f76"
  source_file: "docs/tutorials/signal/learned/classifier.md"
  outdated: false
---

# 分类器信号 {#classifier-signal}

## 概览 {#overview}

`classifier` 暴露本地原生序列分类器、远程序列分类器或已配置外部 LLM 的可复用标签分数。决策用必需的数值谓词测试已声明标签。

专用的 domain、PII、jailbreak、fact-check、KB 与 preference 信号仍是各自领域的首选接口。

## 主要优势 {#key-advantages}

- 接入任意序列分类头，而不添加领域逻辑
- 将 LLM 分类器约束到已声明标签与确定性 JSON 输出
- 计算一份标签映射，多条决策可按不同分数门控

## 解决什么问题？ {#what-problem-does-it-solve}

有些已训练分类器不属于内置信号分类体系。classifier 信号把这些标签与分数暴露给决策，而不把分类与路由结果混在一起。

## 何时使用 {#when-to-use}

对真正可复用的分类头或提示式 LLM 标注器使用该信号。参考短语相似度请优先用 embedding/KB 信号，响应风格路由请用 preference 信号。

## 配置 {#configuration}

```yaml
routing:
  signals:
    classifiers:
      - name: phishing
        type: local
        model_path: models/phishing-email
        labels: [BENIGN, PHISHING]
        use_cpu: true

  decisions:
    - name: phishing-local
      description: Keep suspected phishing requests on the local model.
      priority: 200
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: classifier
            name: phishing
            label: PHISHING
            predicate:
              gte: 0.5
      modelRefs:
        - model: local-small
          use_reasoning: false
```

LLM 分类器引用命名的 `global.model_catalog.external` 条目，并添加 `instructions`。运行时固定温度、输出 schema、精确标签校验，以及默认 1 MiB 的响应上限。在外部模型条目上设置 `max_response_bytes` 可覆盖该上限。因为运行时拥有输出 schema，该条目上的 `parser_type` 必须是 `json` 或未设置；其他值会在配置加载时被拒绝。模型必须为每个已声明标签报告分数；每个分数必须在 `0` 与 `1` 之间，完整分布必须大约合计为 `1.0`。这些是模型报告的置信度分数，不是校准过的分类器概率。Classifier 叶是唯一接受 `on_error` 的决策谓词；失败会在 eval/回放诊断中暴露有界的 `classifier_evaluation_failed` 代码。

失败时，决策树将该叶评估为 `Unknown`，直到完整 AND/OR/NOT 表达式已知。根级 `rules.on_unknown` 再选择 `no_match`、`match` 或 `fail_request`。`no_match` 与 `match` 只解析自身决策；`fail_request` 是全局失败即拒绝：即使另一条决策干净匹配，也会以 503 拒绝整个请求，与优先级无关。省略 `rules.on_unknown` 时，条件级 `on_error`（`no_match` 或 `match`）保留先前通用分类器结果。设置 `rules.on_unknown` 会禁用该树中所有条件级 `on_error`，因此 Router 会拒绝同时设置两者的配置。`prompt_guard.on_error`（`allow` 或 `block`）仍是 jailbreak 规则的兼容默认。诊断同时包含信号错误与已应用的任何终端策略。见[安全模型](../../../installation/runtime/safety)。

`sequence_classifier` 分类器也引用命名外部模型，但使用共享的 `http_classify` 约定，并保留其完整标签分布。响应必须恰好包含已声明标签，分数合计大约为 `1.0`；sigmoid 多标签输出与标签子集会被拒绝。它们至少需要两个标签，并且不接受 `instructions`、`model_path` 或 `use_cpu`。

本地分类器使用 `model_path`，并支持两个或更多已声明标签。每条规则拥有一个已准备的模型句柄，因此一个配方可以声明多个本地分类器。本地决策谓词保留 `gte: 0.5` 或更高。模型或标签变更会在激活前准备候选代；失败的候选会留下当前代可用。

配方可以用 `model_bindings` 中的 `classifier.<rule name>` 条目显式选择执行；这会替换规则的 `model` 或 `model_path` 选择器。本地与序列规则支持本地序列部署或 HTTP `http_classify`，而 LLM 规则保留其计分提取指令并要求 HTTP `http_chat`。全部使用 `label_distribution.v1`，以规则的有序 `labels` 作为映射。见[进程内模型](../../../installation/runtime/in-process)。

本地路径在 Router 内处理请求文本。`llm` 与 `sequence_classifier` 都会把该文本发给已配置的外部模型，因此请相应选择提供方与保留策略。标签与阈值必须作为同一版本化约定一起评估。完整示例见
[`llm`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/label-score.yaml)
与
[`sequence_classifier`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/sequence-label-score.yaml)。
