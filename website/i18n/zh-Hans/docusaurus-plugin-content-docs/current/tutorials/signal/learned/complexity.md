---
is_mtpe: false
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/signal/learned/complexity.md"
  outdated: false
---

# 复杂度信号 {#complexity-signal}

## 概览 {#overview}

`complexity` 通过与已配置示例集比较，估计请求是 `easy`、`medium` 还是 `hard`。它独立于主题：同一领域内的两个请求仍可能需要不同模型档。

本地打分复用已配置的 Embedding 模型（包括 Vela），不需要另外下载 Complexity 模型。也可以接入[远程打分器](#local-and-remote-scoring)。

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
        description: Estimate the work needed to produce a supported answer.
        hard:
          candidates:
            - Find a solution satisfying interacting constraints and justify why alternatives fail.
            - Reconcile conflicting evidence and defend a conclusion under uncertainty.
            - Derive an algorithm and prove its correctness.
        easy:
          candidates:
            - Retrieve a fact explicitly stated in the supplied text.
            - Copy the supplied information into the requested format.
            - Apply one specified local edit while preserving everything else.
```

`threshold` 是裕量，不是相似度截止。Router 分别对 `hard` 与 `easy` 候选库打分，再比较两者：

```text
signal = hard_bank_score - easy_bank_score

signal >  threshold  -> hard
signal < -threshold  -> easy
otherwise            -> medium
```

两个库可能给出相近的分数，因此即使单个分数很高，裕量也可能很小。裕量衡量的是相似度的差距，而不是难度估计正确的概率。

示例和 `0.10` 阈值只是起点。请在所支持的语言中，用简单、普通和高要求的请求校准它们。比较同一主题下需要不同操作的请求，并单独改变回答风格：简短的回答仍可能需要困难的推理。同时检查不必要的升级，以及被送到较弱模型池的困难请求。如果示例的难度分数顺序本身就错了，调整阈值也无法修正。

规则会发出带后缀的名称。决策必须引用 `<rule>:easy`、`<rule>:medium` 或 `<rule>:hard`：

```yaml
routing:
  decisions:
    - name: reasoning
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

复杂度规则也可在 `hard` 和 `easy` 旁声明 `prototype_scoring`。
省略时继承上述族级配置。声明的对象会完整覆盖：未填写的字段，包括空对象 `{}`
中的所有字段，使用内置默认值，不继承族级覆盖值。这样，规则的原始配置会随配方
导出或初始化一同保留。

如需保留两个候选库中的所有不同候选：

```yaml
prototype_scoring:
  enabled: false
  best_weight: 0.75
  top_m: 2
```

此设置关闭聚类和原型数量上限，保留最高相似度与支持分数的聚合。
文本和图像的 hard/easy 候选库及其评分使用同一份解析后的配置。
启用压缩时，`max_prototypes: 0` 使用默认上限 8。这些设置影响本地原型评分，
不影响远程打分器返回的分数；阈值和显式难度边界保持不变。

### 判断模型评分 {#decision-model-scoring}

每条 complexity 规则都提出自己的问题，因此在规则上用 `deployment` 指定它的模型。带有 `hard`/`easy` 样例库且没有 `deployment` 的规则继续使用嵌入比较。在规则上指定一个 `model_runtime` 或 `systemone` 部署，即可改为向它提出打分问题；此时候选样例作为各等级的示例。既没有样例库、也没有 `deployment` 的规则询问决策模型。部署必须支持 `score` 问题。

```yaml
routing:
  signals:
    complexity:
      - name: needs_reasoning
        deployment: primary
        threshold: 0.10
        description: Estimate the work needed to produce a supported answer.
```

任务提出 easy、medium、hard 等级问题，将原生等级分数归一化到 `[0, 1]` 后应用显式边界；仅设置 `threshold` 时先映射到 `[-1, 1]` 再应用对称区间。这是模型判断，不是余弦差值或校准后的正确率；切换任务或模型时应重新评估边界。

### 本地与远程打分 {#local-and-remote-scoring}

没有远程部署时，complexity 在本地打分：`hard` 与 `easy` 候选列表在启动时嵌入一次，每个请求按它更像难示例还是易示例的程度打分。该差值是以零为中心的有符号裕量，因此 `threshold` 是对称的——高于 `+threshold` 为 hard，低于 `-threshold` 为 easy，中间档为 medium。

远程打分器是在 complexity 规则上指定的 `classify` 部署。Router 把请求文本发送到 `<endpoint>/classify`，服务返回一个难度分数（`score.v1`）。所有 complexity 规则由同一个分数驱动，因此一个路由配置中的每条规则都指定同一个部署：如果规则指向不同的远程部署，或者一些规则询问打分器而另一些不询问，Router 会拒绝该配置。替换 `routing.signals` 的配方在它自己的规则上指定部署。

#### 返回分数的模型 {#a-model-that-returns-a-score}

回归模型——例如查询难度打分器——返回一个数字，Router 用每条规则的边界把它变成判定。无论多少规则读取它，一次请求只调用一次。

分数以模型自身单位到达，不是裕量，因此 `threshold` 不适用。改为写明两个边界，所用的一对字段表示难度方向：

```yaml
global:
  model_catalog:
    deployments:
      difficulty-scorer:
        provider: classify
        endpoint: https://difficulty-scorer.example.com
        auth: {api_key_env: DIFFICULTY_SCORER_KEY}
        timeout_ms: 5000

routing:
  signals:
    complexity:
      - name: needs_reasoning
        deployment: difficulty-scorer
        hard_above: 0.85          # a higher score is harder
        easy_below: 0.60
      - name: extreme
        deployment: difficulty-scorer
        hard_above: 0.95          # same score, stricter boundaries
        easy_below: 0.30
```

对于分数随难度上升而下降的模型——例如预测正确答案概率的模型——改用 `hard_below` 与 `easy_above`。把方向编码在字段名里，就不必再维护单独的方向设置，也不会意外写出重叠档。这两个字段适用于远程分数或判断模型评分。本地样例库的裕量是 hard 减 easy，因此更高值本身就更难；要在本地反转它，交换候选列表即可。

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

远程分数不报告置信度。刚低于 `hard_above` 的分数是最不确定的位置，而不是强位置，因此不会从中推导置信度，任何以该规则为门控的决策会按引擎的结构默认排序，而不是报告分数。Router 在启动时对这种情况发出警告。

端点可以用两种形状回答：恰好一条的 HuggingFace 文本分类数组 `[{"label": "difficulty", "score": 0.73}]`，或裸对象 `{"score": 0.73}`。缺失或空分数、多于一条，或非有限值都是错误，绝不是零——零是 `[0,1]` 范围易端的真实分数，故障不得按零路由。

一旦远程部署提供分数，`hard` 与 `easy` 候选列表就不再被读取，Router 会在启动时说明，而不是让你去编辑已无效果的示例。

#### 观察远程打分器 {#watching-a-remote-scorer}

网络依赖会以本地模型不会的方式失败，因此远程路径会报告它们做了什么：

- `llm_remote_connector_requests_total{operation, outcome}` 与 `llm_remote_connector_request_duration_seconds{operation}` 统计并计时共享连接器上的每次调用。`outcome` 是 `success` 或错误种类（`transport`、`status`、`authorization`、`request`、`response`），`llm_remote_connector_retries_total` 统计重试次数。
- `llm_complexity_verdict_total{rule, verdict, source}` 统计判定，其中 `source` 是 `local`、`remote_score` 或 `remote_labels`——与失败计数器携带的同一标签，因此两者可以一起读。该分布的形状是尺度不匹配的检查：`hard_above: 0.85` 后面的 `[1,10]` 打分器会表现为该规则的每个请求都是 `hard`，单条日志行看不出来。
- `llm_complexity_evaluation_failures_total{source}` 统计未产生判定的评估。

打分器不可达时，请求仍会路由。complexity 信号缺失，每条 complexity 规则在请求的信号错误中标记为 `complexity_evaluation_failed`，并由 `llm_complexity_evaluation_failures_total` 计数。

默认情况下，条件读取失败 complexity 规则的决策把它当作未匹配，因此请求落到下一条匹配。要有意决定，请在该决策的根 `rules` 节点上设置 `on_unknown`：

```yaml
decisions:
  - name: reasoning
    priority: 100
    rules:
      on_unknown: match       # no_match (default) | match | fail_request
      operator: AND
      conditions:
        - type: complexity
          name: needs_reasoning:hard
```

`match` 把无法评级的请求送到更强模型，通常是更安全的默认；`fail_request` 直接拒绝。`on_unknown` 属于根 `rules` 节点，并作用于整条决策——逐条件 `on_error` 字段只在 `classifier` 和 `decision` 条件上接受，因此把它放在 `complexity` 条件上会在配置加载时被拒绝。

打分器的数字以模型自身单位发布为 `complexity:<rule>:score`，而本地打分发布 `complexity:<rule>:margin` 及其文本/图像分量。这些到达回放记录与可观测性；决策条件匹配判定（`<rule>:<verdict>`），而不是数字，因此对 `:score` 的数值谓词不是路由机制。

## 依赖与限制 {#dependencies-and-limitations}

- 本地打分使用 `embeddings.default` 指定的 embedding 部署。远程 embedding 部署会收到用于分类的请求文本。
- 候选短语与阈值必须对照已标注流量一起校准。嵌入模型变更时请重新评估。
- 模糊提示可能落到 `medium` 档；请为你依赖的每一档定义路由或回退。
- 完整示例见：
 [`config/fragments/signal/complexity/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/complexity/escalation.yaml)。
