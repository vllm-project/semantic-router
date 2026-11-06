---
is_mtpe: false
translation:
  source_commit: "3e9e4262e5c4e286ede3fb75cdac83edb985b6aa"
  source_file: "docs/tutorials/signal/learned/classifier.md"
  outdated: false
---

# 分类器信号 {#classifier-signal}

## 概览 {#overview}

`classifier` 暴露本地原生序列分类器、远程序列分类器或已配置外部 LLM 的可复用标签分数。决策通过数值谓词，或部署所固定的工作点，测试已声明的标签。

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

每条分类器规则都提出自己的问题，因此用 `deployment` 指定回答它的部署。没有 `deployment` 的规则询问决策模型。规则如何解析见[为信号选择模型](../../../installation/choose-a-model-for-a-signal.md)。

```yaml
global:
  model_catalog:
    deployments:
      phishing-model:
        provider: model_runtime
        artifact: /models/phishing-email
        device: cpu

routing:
  signals:
    classifiers:
      - name: phishing
        deployment: phishing-model
        labels: [BENIGN, PHISHING]

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

部署的 `provider` 决定如何询问该规则：

| `provider` | 规则如何得到回答 |
| --- | --- |
| `model_runtime` | 模型运行时中的序列分类器为已声明标签打分 |
| `classify` | Router 把文本发送到 `<endpoint>/classify`，读取完整的标签分布 |
| `openai_chat` | chat 模型按规则的 `instructions` 为已声明标签打分 |
| （没有 `deployment`） | 决策模型回答一个标签问题 |

### 把 chat 模型用作分类器 {#chat-models-as-classifiers}

由 `openai_chat` 部署回答的规则需要添加 `instructions`。Router 固定输出 schema 和精确标签校验，读取的响应上限为 1 MiB，除非部署的 `limits.max_response_bytes` 设置了其他上限。Router 以温度 0 提问；部署的 `generation` 可以设置其他温度或补全长度。

```yaml
global:
  model_catalog:
    deployments:
      risk-judge:
        provider: openai_chat
        endpoint: https://llm.example.com/v1
        model: qwen/qwen3.8-27b
        auth: {api_key_env: RISK_JUDGE_KEY}
        timeout_ms: 5000
        generation: {max_tokens: 256}

routing:
  signals:
    classifiers:
      - name: tool-risk
        deployment: risk-judge
        labels: [SAFE, RISKY]
        instructions: Choose RISKY only for destructive or irreversible tool operations.
```

在这样的规则上设置 `disable_rationale: true`，只请求 `{"scores": {...}}`。响应可以省略 `rationale` 或给出空字符串；如果出现，它仍必须是字符串。其他响应字段会被拒绝。省略该标志或设为 `false` 时，提示词同时请求 `scores` 和 `rationale`，响应必须包含非空的 rationale。该标志只适用于由 `openai_chat` 部署回答的规则。模型必须为每个已声明标签报告分数；每个分数必须在 `0` 与 `1` 之间，完整分布必须大约合计为 `1.0`。这些是模型报告的置信度分数，不是校准过的分类器概率。`classifier` 和 `systemone` 叶均接受条件级 `on_error`；分类器失败会在 eval/回放诊断中暴露有界的 `classifier_evaluation_failed` 代码。

失败时，决策树将该叶评估为 `Unknown`，直到完整 AND/OR/NOT 表达式已知。根级 `rules.on_unknown` 再选择 `no_match`、`match` 或 `fail_request`。`no_match` 与 `match` 只解析自身决策；`fail_request` 是全局失败即拒绝：即使另一条决策干净匹配，也会以 503 拒绝整个请求，与优先级无关。省略 `rules.on_unknown` 时，条件级 `on_error`（`no_match` 或 `match`）保留先前通用分类器结果。设置 `rules.on_unknown` 会禁用该树中所有条件级 `on_error`，因此 Router 会拒绝同时设置两者的配置。`prompt_guard.on_error`（`allow` 或 `block`）仍是 jailbreak 规则的兼容默认。诊断同时包含信号错误与已应用的任何终端策略。见[安全模型](../../../model-runtime/guides/safety.md)。

### 远程分类服务 {#remote-classification-services}

由 `classify` 部署回答的规则保留服务返回的完整标签分布。响应必须恰好包含已声明标签，分数合计大约为 `1.0`；sigmoid 多标签输出与标签子集会被拒绝。这样的规则至少需要两个标签，并且不接受 `instructions`。

### 本地分类器 {#local-classifiers}

由 `model_runtime` 部署回答的规则在[模型运行时](../../../model-runtime/guides/classify.md#use-your-own-classifier)中运行，它加载 Hugging Face 的 ModernBERT 和 mmBERT 序列分类器；其他架构需要[家族插件](../../../model-runtime/plugins.md)。部署的 `device` 决定它在哪里运行。规则声明两个或更多标签，且必须与 checkpoint 的数字 `id2label` 顺序一致。每条规则拥有一个已准备的模型句柄，因此一个配方可以声明多个本地分类器。本地决策谓词保留 `gte: 0.5` 或更高。模型或标签变更会在激活前准备候选代；失败的候选会留下当前代可用。

配方自己的 `routing.signals` 中的规则以同样的方式指定部署，因此每个配方为自己的规则选择模型。规则的有序 `labels` 就是它的标签映射。见[与 Router 一起运行](../../../model-runtime/deploy.md)。

## 使用固定工作点的独立标签 {#independent-labels-with-a-frozen-operating-point}

Hazard 等多标签分类器可独立于 Safety 信号运行。用一个模型运行时部署回答该规则，其模型包附带版本 2 的 `operating_point.json`。模型、分词器、执行方式、标签顺序、窗口几何和阈值都绑定在该文件中，运行时在提供服务前会校验它。部署的 `operating_point` 固定该文件：`path` 指明它在部署产物中的位置，`sha256` 是其字节的哈希。规则随后读取相互独立的标签分数。如果运行时提供的策略摘要不同，准备阶段会拒绝该部署。

```yaml
routing:
  signals:
    classifiers:
      - name: content-risk
        deployment: content-risk-cpu
        labels: [violence, criminal_activity, sexual_content, child_exploitation,
                 hate, harassment_abuse, regulated_substances, weapons, self_harm,
                 privacy, specialized_advice, misinformation]
  decisions:
    - name: weapon-risk
      priority: 100
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: classifier
            name: content-risk
            label: weapons
      modelRefs:
        - model: answer-model
global:
  model_catalog:
    deployments:
      content-risk-cpu:
        provider: model_runtime
        artifact: /models/content-risk
        device: cpu
        input:
          max_tokens: 32768
          overflow: reject
        operating_point:
          path: operating_point.json
          sha256: <SHA256 of the exact version-2 sidecar>
```

替换 SHA256 占位符，并使用产物声明的完整有序标签。文档 token 预算必须与工作点文件一致。省略 `predicate` 时，使用该标签的固定阈值（`score >= threshold`）；可同时匹配多个标签，也可全部不匹配。显式谓词查询原始独立分数，分数不要求合计为一。没有工作点的规则仍要求谓词。

模型运行时以模型发布的 FP32 精度（`dtype: auto`）执行该策略。工作点文件以 SHA256 绑定权重、tokenizer 和窗口几何；产物变化会被拒绝。文档总预算与每个窗口的执行预算相互独立。

模型只分词一次，按声明的重叠窗口覆盖原始内容 token，恢复特殊 token 并重置位置，然后对每个标签取各窗口 sigmoid 分数的最大值。文档超长、扫描不完整、产物变化或执行方式不受支持时会报错。版本 1 缺少必要的身份信息，因此会被拒绝；现有 Safety/Hazard 组合方式不变。

Eval 的 `metrics.classifier.rules` 记录工作点 SHA256、实际提供方、设备、精度、token 用量、内容窗口偏移、阈值和耗时，延迟包含完整扫描。工作点中的参考 batch size 记录校准条件。执行错误保持为 `Unknown`（包括在 `NOT` 下），并遵循现有的 `on_unknown` 策略。

绑定了已退役 ONNX Runtime CK Flash Attention 路径的版本 3 策略不再被接受；请在模型运行时上以 `dtype: auto` 重新评定该策略。

如需将已经选定的仅运行时评分策略绑定到最终原生文件，在 `src/semantic-router` 下运行打包工具：

```bash
go run ../../tools/models/classifier-operating-point/main.go \
  --model /path/to/native-model \
  --policy /path/to/selected-score-policy.json \
  --output /path/to/new-operating-point.json
```

工具输出工作点文件的 SHA256，保留评分和窗口字段，并核验已有权重身份。它补齐最终配置与分词器哈希；版本 2 的执行声明会被保留，所引用的文件会被核验。工具不选择阈值、不评定模型，且拒绝覆盖已有文件。请将工作点与对应的原生文件一起发布，不要跨 checkpoint 复制阈值。

`model_runtime` 部署在本地处理请求文本。`openai_chat` 和 `classify` 部署会把该文本发送给远程服务，因此请相应选择提供方与保留策略。标签与阈值必须作为同一版本化约定一起评估。完整示例见[chat 模型](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/label-score.yaml)与[分类服务](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/classifier/sequence-label-score.yaml)。
