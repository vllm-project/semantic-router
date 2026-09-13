---
translation:
  source_commit: "bce357c513f391824e8320267d03977794c20f76"
  source_file: "docs/tutorials/signal/learned/embedding.md"
  outdated: false
---

# 嵌入信号 {#embedding-signal}

## 概览 {#overview}

`embedding` 按与代表性示例的语义相似度匹配请求。在 `routing.signals.embeddings` 下定义嵌入规则。

它依赖 `global.model_catalog.embeddings` 中配置的嵌入模型。

这些资产可以在本地运行，或通过外部兼容 OpenAI 的文本嵌入端点运行。共享提供方配置见[运行时嵌入](../../../installation/runtime/embeddings)；信号候选、阈值与决策条件保持不变。

## 主要优势 {#key-advantages}

- 比纯关键词规则更好处理改写。
- 团队可用示例短语调路由，不必重训分类器。
- 适合支持意图、产品流程与语义 FAQ 路由。
- 是从纯词法信号平滑升级的一步。

## 解决什么问题？ {#what-problem-does-it-solve}

关键词路由会漏掉措辞不同但语义相近的提示。完整领域分类在路由依赖窄意图时也可能过粗。

`embedding` 在嵌入空间中将新提示与示例候选匹配。

## 何时使用 {#when-to-use}

在以下情况使用 `embedding`：

- 措辞变但意图稳定
- 希望语义路由而不引入完整自定义分类器
- 示例比领域标签更易维护
- 支持或工作流意图需要比关键词更好的召回

## 配置 {#configuration}

```yaml
routing:
  signals:
    embeddings:
      - name: technical_support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - how to configure the system
          - installation guide
          - troubleshooting steps
          - error message explanation
          - setup instructions
      - name: account_management
        threshold: 0.72
        aggregation_method: max
        candidates:
          - password reset
          - account settings
          - profile update
          - subscription management
          - billing information
```

阈值与候选列表一起调；这比堆砌大量低质示例更重要。

用嵌入模型设置配置排序回退行为：

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          enable_soft_matching: true
          top_k: 1
          min_score_threshold: 0.5
          prototype_scoring:
            enabled: true
            cluster_similarity_threshold: 0.9
            max_prototypes: 8
            best_weight: 0.75
            top_m: 2
            margin_threshold: 0.05
```

`prototype_scoring` 把每条嵌入规则的候选库压缩成更小的代表性原型集，再从这些原型给规则打分，而不是永远依赖一份扁平候选列表。

Router 先给每条嵌入规则打分，再把 `top_k` 作为发出上限。默认是 `1`，因此只返回最强嵌入信号。设 `top_k: 0` 可返回所有达到阈值的规则。

## 设计并校验候选集 {#design-and-validate-candidate-sets}

把规则的 `candidates` 当作小型语义分类器，而不是关键词列表：

- 描述你想识别的输入种类。对文本，使用该意图的多样示例，而不是同一句子的几个版本。对图像规则，描述可见结构，例如 `photograph of a passport page`；不要依赖需要 OCR 的字面词。
- 从几个不同角度覆盖该类别。近重复候选几乎不增加召回，还可能让规则看起来比实际校准得更好。
- 在评估集中包含常规、良性示例。当适合胜者式发出时，竞争的良性规则也能给普通输入更好的语义匹配。用实际的 `top_k` 与阈值设置测试；良性规则不是安全黑名单。
- 任一强示例都应匹配时，使用 `aggregation_method: max`。仅当希望候选集整体广泛一致时才使用 `mean`。
- 对照已部署模型的已标注正负流量校准 `threshold`。阈值在模型、维度、模态或流量分布之间不能可靠迁移。

变更候选、模型或阈值时，重新运行已标注评估。把这三项输入一起记录，以免配置更新静默复用不相容的阈值。

## 多模态查询（`query_modality`） {#multimodal-queries-query_modality}

每条嵌入规则接受可选的 `query_modality` 字段，声明规则查询从传入请求载荷的哪种模态计算。候选在任何情况下都是文本；规则在同一共享多模态空间中，将文本锚点集与声明模态的查询嵌入做余弦匹配。

接受的值：

- `"text"`（默认，向后兼容）：从请求文本嵌入查询。没有 `query_modality` 字段的现有规则行为与以前完全相同。
- `"image"`：从 OpenAI 风格聊天消息中允许列表内的内联 `data:image/...;base64,...` 附件嵌入查询。
- 目前不支持音频查询嵌入；请使用 `text` 或 `image`。

`"image"` 要求 `global.model_catalog.embeddings.semantic.embedding_config.model_type: multimodal`，使候选与查询共享一个嵌入空间。Router 会拒绝与仅文本嵌入模型配对的图像模态规则。

### 示例：把敏感影像路由到本地 {#worked-example-route-sensitive-imagery-on-prem}

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        multimodal_model_path: models/multi-modal-embed-small
        embedding_config:
          model_type: multimodal

routing:
  signals:
    embeddings:
      # query_modality 默认为 text。
      - name: technical_support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - how to configure the system
          - installation guide
          - troubleshooting steps

      # 在共享多模态嵌入空间中，将内联图像附件与文本锚点匹配。
      - name: medical_imagery_phi
        query_modality: image
        threshold: 0.55
        aggregation_method: max
        candidates:
          - chest X-ray with patient identifier strip
          - dermatology lesion close-up photograph
          - electronic health record application screenshot showing patient demographics
          - ultrasound scan with patient name overlay
```

决策随后可以像对待其他信号一样，按新信号路由：

```yaml
routing:
  decisions:
    - name: route_medical_imagery_on_prem
      description: Keep medical imagery on the in-cluster vision model.
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: medical_imagery_phi
      modelRefs:
        - model: in-cluster-vlm
```

### 图像路由示例 {#image-routing-example}

可选的
[`image-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml)
示例包含 `identifier_document_imagery`、`code_or_terminal_imagery`，以及良性的 `ambient_office_imagery` 规则。用你部署中的示例替换其候选，并重新校准阈值。示例值不是可移植默认。

### 与 `modality` 信号类型的区别 {#distinction-from-the-modality-signal-type}

`query_modality`（本节）声明嵌入规则的**输入模态**——查询从哪种载荷模态计算。单独的 [`modality`](modality) 信号类型声明**输出模态**（`AR`、`DIFFUSION`、`BOTH`），用于路由图像生成请求。两个概念共用一个名字，但解决不同问题，并位于不同配置面上。

## 依赖与限制 {#dependencies-and-limitations}

- 文本或图像内容由已配置的嵌入运行时处理。远程提供方目前只支持文本，并接收被嵌入的文本。不支持音频查询嵌入。
- 相似度分数与阈值在嵌入模型、维度或模态之间不可移植。这些变更时请重新校准。
- 图像匹配是语义匹配，不是 OCR 或 PII 抽取；字面文本或受监管实体重要时，请使用专用检测器。
- 完整示例：
 [`support.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/support.yaml)
 与
 [`image-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml)。
