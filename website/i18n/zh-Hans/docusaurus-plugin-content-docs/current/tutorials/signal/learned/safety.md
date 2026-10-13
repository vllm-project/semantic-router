---
translation:
  source_commit: "fddd53d7c446c30ae183d16b6d63ec54f7a00a3e"
  source_file: "docs/tutorials/signal/learned/safety.md"
  outdated: false
---

# 安全

## 概览 {#overview}

`safety` 信号预测内容风险。它补 `jailbreak`（预测提示词攻击）和 `pii`（找实体片段）的位。每条安全规则有一个总的不安全阈值，还可以点名具体的风险类别。

## 解决什么问题 {#what-problem-does-it-solve}

内容风险和提示词攻击要的是不同的标签。把一个有害话题聊得无害，可以是安全的；而一个有害请求未必含任何想覆盖系统指令的成分。Safety 出内容风险分，越狱信号由 Guard 另出一路。

## 什么时候用 {#when-to-use}

宽泛的内容策略用一条二元规则；只让选中的类别去选中决策，就挂一个 Hazard 条件。启用拒绝类策略之前，先按自己应用的语言和输入长度把阈值验一遍。

## 配置 {#configuration}

```yaml
routing:
  signals:
    safety:
      - name: unsafe-content
        threshold: 0.5
      - name: unsafe-privacy
        threshold: 0.5
        hazard:
          labels: [violence, criminal_activity, sexual_content, child_exploitation,
                   hate, harassment_abuse, regulated_substances, weapons,
                   self_harm, privacy, specialized_advice, misinformation]
          categories: [privacy]
          threshold: 0.6
```

不写 `model` 用内置的本地 Safety 或 Hazard 模块。显式写 `model` 则解析 `global.model_catalog.external` 里的一个分类端点。两种形式的分语义一样。本地产物，`labels` 必须和 `config.json` 的 `id2label` 顺序完全一致；加载器还会查头的 `problem_type`。外部端点按名字对齐标签。

- `labels` 和 `unsafe_labels` 默认是 `[safe, unsafe]` 和 `[unsafe]`。
- 二元规则把选中的不安全标签分求和，和 `threshold` 比。相等即命中。
- 带 `hazard` 的规则先要求二元条件成立，再把 `categories` 里的最高分和 `hazard.threshold` 比。Hazard 各概率互相独立；分数绝不相加、也不重新归一。
- 模型、标签、激活契约都相同的多条规则，一个请求共享一次预测。它们的阈值和命中各自算。
- 总不安全阈值没够，Hazard 整个跳过。这省推理，但类别的召回也受这道二元门制约。

上面列的类别是 Vela 的分类法。它不为每类政策关切（版权、高风险治理、操纵）提供专门标签。提到一个话题不等于有害；阈值请结合模型的评测报告和你自己的验证集来定。

在决策里引用规则，动作也定在那里：

```yaml
routing:
  decisions:
    - name: handle-content-risk
      priority: 300
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: safety
            name: unsafe-content
      modelRefs:
        - model: safety-capable-model
```

`safety-capable-model` 换成 `providers.models` 里一个后端能恰当处理内容风险的别名。信号命中也可能包含一个危机中求助的人；它不意味恶意，也不该导向一刀切的拒绝。该拒绝时才用按类别细分的决策。Safety 信号只出分；消费它的决策选应对策略。提示词攻击的决策和内容风险的决策放在一起排，相对优先级要显式定。

模型出错，信号未知。`rules.on_unknown: fail_request` 会在决策持续未知时返回 HTTP 503。打了分的不安全请求选中配好的处理路由。诊断在 `x-vsr-matched-safety`、dashboard 和回放记录里给出命中的规则名和分类结果。

本地上下文预算、外部端点和失败策略见[共享模型配置](../../../model-runtime/guides/safety.md)，按类别细分的策略见[完整 HTTP 示例](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/safety/content-safety.yaml)。

## 选 Vela Shield {#select-vela-shield}

内置 Safety 模块默认用 [Vela Safety](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M-Safety)。[Vela Shield](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M-Shield) 是另训的一套，标签同样是 `safe`/`unsafe`，现有规则和阈值原样适用。换模型后阈值要重验。

所有安全规则都想换 Shield，就设模块的模型：

```yaml
global:
  model_catalog:
    modules:
      safety:
        safety:
          model_id: models/Vela-1.0-Encoder-307M-Shield
```

只想一个配方里的一条规则用 Shield，就声明部署、把规则绑上去。别的配方仍用模块的模型：

```yaml
global:
  model_catalog:
    deployments:
      shield:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Shield
        revision: a981a99eeb05a2859b88b5cee9af4352897ec4ec
recipes:
  - name: care
    routing:
      model_bindings:
        safety.unsafe-content:
          deployment: shield
          contract: label_distribution.v1
      signals:
        safety:
          - name: unsafe-content
            threshold: 0.5
```

模块形式用内置注册表里钉住的 revision；部署用它自己声明的 `revision`。两种形式下载的都只有根分类器。Shield 仓库在 `heads/` 下还发布了辅助头、在 `lc/` 下发布了标签条件编码器；router 不加载、也不下载它们。

## 长输入扫描 {#long-input-scanning}

Safety 头默认整输入推理。一套单独校准的窗口策略，可以扫长请求里随处可见的本地风险：

```yaml
global:
  model_catalog:
    modules:
      safety:
        safety:
          model_id: models/my-safety-model
          max_sequence_length: 32768
          window:
            size: 512
            overlap: 255
```

`max_sequence_length` 限的是分词后整个请求，特殊 token 也算。更长的输入直接失败；router 绝不悄悄只留第一个窗口。`window.size` 含特殊 token，`overlap` 数的是内容 token。对一个加两个特殊 token 的分词器，这个例子每次前进 255 个内容 token。每个窗口都还原原始特殊 token，位置从零起。内容为空、窗口预算非法，都失败。

具名的本地绑定，部署的 `input.max_tokens` 取代模块的整文档预算。加载的检查点和图支持 32K 前向时，文档预算 65,536、`window.size` 32,768 是合法的。只有单窗口必须装得进一次前向；文档可以要多个窗口。token 数按分类器的分词器算，不是下游生成模型的。一个窗口失败，整场扫描就失败，不会只给查过的前缀返一个好分。

扫描用原始 token ID，覆盖每个内容 token，最后一个窗口允许短一截。它把选中的不安全概率在窗内求和，再取最大的窗口分，启用一次规则阈值。Hazard 同理，取各窗口中选中的类别最高概率。那个产物是按扫描评过的，就在 `hazard` 头下单独设 `window`。外部分类器保留自己的输入处理契约。已发布的 Vela Hazard 工作点保持随附的 2,048 token 窗口和 32K 文档策略；把文档预算配大，不会让那个工作点够格换成另一种扫描。

阈值要用按你部署的准确模型、窗口大小、重叠和[档位](../../../model-runtime/profiles.md) 评出来的。窗口扫描能捞回整输入分类器漏掉的本地风险，但它读不懂同一窗口之外的远处拒绝或保护意图。引用材料和其他长程上下文，在应用层验。需要整输入语义时，别写 `window`。
