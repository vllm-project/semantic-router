---
translation:
  source_commit: "701e3631aa14a840646b8feb78caef3b5d9a91e3"
  source_file: "docs/tutorials/signal/learned/complexity.md"
  outdated: false
---

# 复杂度信号 {#complexity-signal}

## 概览 {#overview}

`complexity` 通过与已配置示例集比较，估计请求是 `easy`、`medium` 还是 `hard`。它独立于主题：同一领域内的两个请求仍可能需要不同模型档。

## 主要优势 {#key-advantages}

- 将估计难度与主题分类分开。
- 同一 easy/medium/hard 策略可被多条决策复用。
- 用示例调路由，而不需要自定义分类器 schema。

## 解决什么问题？ {#what-problem-does-it-solve}

仅靠领域路由无法区分简短事实请求与多步分析请求。Complexity 提供可复用的难度信号，使决策只升级真正需要的流量。

## 何时使用 {#when-to-use}

当模型成本或推理模式应随估计任务难度变化时，使用 complexity。不要把它当作正确性或安全保证；那些问题请用领域特定评估与安全信号。

## 配置 {#configuration}

```yaml
routing:
  signals:
    complexity:
      - name: needs_reasoning
        threshold: 0.10
        description: Escalate multi-step reasoning or synthesis-heavy prompts.
        hard:
          candidates:
            - solve this step by step
            - compare multiple tradeoffs
            - analyze the root cause
        easy:
          candidates:
            - answer briefly
            - quick summary
            - simple rewrite
```

`threshold` 是裕量，不是相似度截止。Router 分别对 `hard` 与 `easy` 候选库打分，再比较两者：

```text
signal = hard_bank_score - easy_bank_score

signal >  threshold  -> hard
signal < -threshold  -> easy
otherwise            -> medium
```

因为两个库可能给出相近的基线分数，相减得到的裕量可能远小于任一单独分数。在一次使用上述候选库的校准中，八条提示上观察到的最大绝对裕量是 `0.197`；阈值为 `0.75` 时没有一条进入 `hard` 或 `easy` 档。

把 `0.10` 当作起点，而不是通用默认。在代表性流量上测量裕量，再为已配置的候选库与嵌入模型调阈值。

规则会发出带后缀的名称。决策必须引用 `<rule>:easy`、`<rule>:medium` 或 `<rule>:hard`：

```yaml
routing:
  decisions:
    - name: escalate-hard-prompts
      description: Route hard prompts to the reasoning model.
      priority: 150
      rules:
        operator: AND
        conditions:
          - type: complexity
            name: needs_reasoning:hard
      modelRefs:
        - model: reasoning-model
          use_reasoning: true
```

如需可选的原型库调参，在族级模块上配置一次：

```yaml
global:
  model_catalog:
    modules:
      complexity:
        prototype_scoring:
          enabled: true
          max_prototypes: 8
          top_m: 2
```

### 本地与远程打分 {#local-and-remote-scoring}

没有 `backend` 时，complexity 保持现有本地行为：`hard` 与 `easy` 候选列表在启动时嵌入一次，每个请求按它更像难示例还是易示例的程度打分。该差值是以零为中心的有符号裕量，因此 `threshold` 是对称的——高于 `+threshold` 为 hard，低于 `-threshold` 为 easy，中间档为 medium。

远程打分器使用共享 backend 块，声明在 `prototype_scoring` 旁边，而不是写在规则上。每个路由配方会整段替换 `routing.signals`，因此写在规则上的 backend 会在未重复声明它的配方下消失，留下仍在运行但已悄悄退回本地的信号。其 `model` 是 `global.model_catalog.external[]` 中的显式名称，该条目需要 `model_role: classification`。

Complexity 读取两种约定，选哪一种取决于模型返回什么。因为有两种，`contract` 不能默认，必须写明。

#### 返回分数的模型：`score.v1` {#a-model-that-returns-a-score-scorev1}

回归模型——例如查询难度打分器——返回一个数字，Router 用每条规则的边界把它变成判定。无论多少规则读取它，一次请求只调用一次。

分数以模型自身单位到达，不是裕量，因此 `threshold` 不适用。改为写明两个边界，所用的一对字段表示难度方向：

```yaml
global:
  model_catalog:
    external:
      - name: difficulty-scorer
        model_role: classification
        llm_endpoint:
          address: difficulty-scorer.default.svc
          port: 8080
        llm_model_name: query-difficulty-v1
    modules:
      complexity:
        backend:
          protocol: http_classify
          contract: score.v1
          model: difficulty-scorer
          deadline_ms: 5000

routing:
  signals:
    complexity:
      - name: needs_reasoning
        hard_above: 0.85          # 分数越高越难
        easy_below: 0.60
      - name: extreme
        hard_above: 0.95          # 同一分数，更严的边界
        easy_below: 0.30
```

对于分数随难度上升而下降的模型——例如预测正确答案概率的模型——改用 `hard_below` 与 `easy_above`。把方向编码在字段名里，就不必再维护单独的方向设置，也不会意外写出重叠档。这两个字段要求 `score.v1` backend：本地裕量是 hard 减 easy，因此更高值本身就更难；要在本地反转它，交换候选列表即可。

想要比三个判定更细的分级？对同一分数写更多规则，各有自己的边界。判定词表仍是 `hard|easy|medium`，因为决策匹配 `<rule>:<verdict>`，规则仍可通过各自的 `composer` 条件区分。

上面两条规则不是互斥替代：打分器只调用一次，每条规则用自己的边界读同一分数，因此一次请求为每条规则产生一个判定。

| 分数 | `needs_reasoning`（0.85 / 0.60） | `extreme`（0.95 / 0.30） |
| ----- | ------------------------------- | ----------------------- |
| 0.20 | easy | easy |
| 0.50 | easy | medium |
| 0.70 | medium | medium |
| 0.90 | **hard** | medium |
| 0.99 | hard | **hard** |

这给决策一条可匹配的阶梯——最顶端用 `extreme:hard`，超过 0.85 用 `needs_reasoning:hard`，底部用 `needs_reasoning:easy`。分数 0.99 同时满足两个 `hard` 条件，因此 `extreme:hard` 的决策需要更高 `priority`，否则更松的规则会拿走请求。

给每一档显式 `priority`，并把更严的一档设为更大数字。`priority` 可以省略；两条决策在同一优先级都匹配时，按置信度分开——而 `score.v1` 从不报告置信度，因此比较会落到决策名的字母序。重命名决策就会改变请求到达哪个模型，且没有任何报告说明发生了这件事——#3658 跟踪让最终比较可观测。

`score.v1` 不报告置信度。刚低于 `hard_above` 的分数是最不确定的位置，而不是强位置，因此不会从中推导置信度，任何以该规则为门控的决策会按引擎的结构默认排序，而不是报告分数。Router 在启动时对这种情况发出警告。

端点可以用两种形状回答：恰好一条的 HuggingFace 文本分类数组 `[{"label": "difficulty", "score": 0.73}]`，或裸对象 `{"score": 0.73}`。缺失或空分数、多于一条，或非有限值都是错误，绝不是零——零是 `[0,1]` 范围易端的真实分数，故障不得按零路由。

#### 返回判定的模型：`label_distribution.v1` {#a-model-that-returns-the-verdict-label_distributionv1}

三分类模型直接返回 `hard`、`easy` 与 `medium`。不查阅边界——获胜标签*就是*判定——其概率是真实置信度，因此基于置信度的排序不受影响。标签是固定判定词表，不可配置：

```yaml
    modules:
      complexity:
        backend:
          protocol: http_classify
          contract: label_distribution.v1
          model: difficulty-classifier
          deadline_ms: 5000
```

无论哪种方式，一旦 backend 提供分数，`hard` 与 `easy` 候选列表就不再被读取，Router 会在启动时说明，而不是让你去编辑已无效果的示例。

#### 观察远程打分器 {#watching-a-remote-scorer}

网络依赖会以本地模型不会的方式失败，因此远程路径会报告它们做了什么：

- `llm_remote_connector_requests_total{operation, outcome}` 与 `llm_remote_connector_request_duration_seconds{operation}` 统计并计时共享连接器上的每次调用。`outcome` 是 `success` 或错误种类（`transport`、`status`、`authorization`、`request`、`response`），`llm_remote_connector_retries_total` 统计重试次数。
- `llm_complexity_verdict_total{rule, verdict, source}` 统计判定，其中 `source` 是 `local`、`remote_score` 或 `remote_labels`——与失败计数器携带的同一标签，因此两者可以一起读。该分布的形状是尺度不匹配的检查：`hard_above: 0.85` 后面的 `[1,10]` 打分器会表现为该规则的每个请求都是 `hard`，单条日志行看不出来。
- `llm_complexity_evaluation_failures_total{source}` 统计未产生判定的评估。

打分器不可达时，请求仍会路由。complexity 信号缺失，每条 complexity 规则在请求的信号错误中标记为 `complexity_evaluation_failed`，并由 `llm_complexity_evaluation_failures_total` 计数。

默认情况下，条件读取失败 complexity 规则的决策把它当作未匹配，因此请求落到下一条匹配。要有意决定，请在该决策的根 `rules` 节点上设置 `on_unknown`：

```yaml
decisions:
  - name: deep-reasoning
    priority: 100
    rules:
      on_unknown: match       # no_match（默认）| match | fail_request
      operator: AND
      conditions:
        - type: complexity
          name: needs_reasoning:hard
```

`match` 把无法评级的请求送到更强模型，通常是更安全的默认；`fail_request` 直接拒绝。`on_unknown` 属于根 `rules` 节点，并作用于整条决策——逐条件 `on_error` 字段只在 `classifier` 条件上接受，因此把它放在 `complexity` 条件上会在配置加载时被拒绝。

打分器的数字以模型自身单位发布为 `complexity:<rule>:score`，而本地打分发布 `complexity:<rule>:margin` 及其文本/图像分量。这些到达回放记录与可观测性；决策条件匹配判定（`<rule>:<verdict>`），而不是数字，因此对 `:score` 的数值谓词不是路由机制。

## 依赖与限制 {#dependencies-and-limitations}

- Complexity 使用已配置的语义嵌入运行时。远程嵌入提供方会收到用于分类的请求文本。
- 候选短语与阈值必须对照已标注流量一起校准。嵌入模型变更时请重新评估。
- 模糊提示可能落到 `medium` 档；请为你依赖的每一档定义路由或回退。
- 完整示例见：
 [`config/fragments/signal/complexity/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/complexity/escalation.yaml)。
