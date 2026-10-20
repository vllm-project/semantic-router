---
translation:
  source_commit: "2e968c6bb62123c2cfbc239cc727f1e4d1d92b03"
  source_file: "docs/tutorials/signal/heuristic/input-modality.md"
  outdated: false
---

# 输入模态信号 {#input-modality-signal}

## 概览 {#overview}

`input_modality` 确定性匹配已解析请求中存在哪些输入种类——`text`、`image`、`audio` 或 `video`。检测纯属结构：Router 检查用户消息中的 content-part 类型，从不运行分类器、嵌入模型或任何其他机器学习推理。

输入模态独立于预期*输出*模态（`modality` 信号区分 `AR` 与 `DIFFUSION` 生成），也独立于为语义匹配嵌入的载荷（嵌入的 `query_modality`）。一个请求可同时包含多种输入模态，每条已配置规则独立匹配。

## 主要优势 {#key-advantages}

- 以零推理成本，把任何包含图像（或音频/视频）的请求路由到有能力的模型
- 暴露请求的完整模态集合，供 AND/OR 决策规则、投影与追踪组合
- 适用于 Chat Completions、Anthropic Messages、Response API 以及 classify/eval HTTP API（各协议覆盖见检测规则）

## 解决什么问题？ {#what-problem-does-it-solve}

常见拓扑是：任何包含图像的请求走向视觉能力模型，纯文本请求继续走正常语义路由。在该信号出现前，图像是否存在只能通过 `conversation` 信号的 `image_content` 源到达，既难发现又只覆盖图像。`input_modality` 直接命名这一概念，并覆盖全部四种模态。

注意 `NOT image_input` 并不等于「仅文本」：没有图像的请求仍可能是仅音频或空请求。将 `text` 模态规则与媒体规则的否定组合，才能精确表达仅文本。

## 何时使用 {#when-to-use}

当路由决策取决于存在哪些输入种类——而不是媒体内容是什么——时，使用 `input_modality`。要按图像的语义内容路由，请改用带 `query_modality: image` 的 embedding 信号。

## 配置 {#configuration}

```yaml
routing:
  signals:
    input_modality:
      - name: image_input
        description: Request contains at least one image content part.
        modality: image
      - name: audio_input
        description: Request contains at least one audio content part.
        modality: audio

  decisions:
    - name: vision-request
      description: Send image-bearing requests to the vision pool.
      priority: 1000
      rules:
        operator: AND
        conditions:
          - type: input_modality
            name: image_input
      modelRefs:
        - model: vision-model
```

`modality` 必须是 `text`、`image`、`audio` 或 `video` 之一。规则名必须唯一且已去空白。

## 检测规则 {#detection-rules}

计数只覆盖用户消息，因此系统提示与 assistant 历史永远不会满足规则。每个入口协议都会先解码一次到 Router 的中性请求，计数来自其中的内容种类，因此无论请求由哪种协议携带，规则匹配方式相同：

- **Chat Completions**：纯字符串内容或 `text` 部分计为文本，`image_url` 部分计为图像，`input_audio` 部分计为音频。
- **Anthropic Messages**：`text` 与 `image` 块；该协议没有音频或视频块类型。
- **Response API**：`input_text` 计为文本，`input_image` 计为图像。Response API 解码器目前不接受音频内容类型。
- **Classify/eval API**：`messages[].content` 上的相同部分类型，外加音频的 `input_audio` / `audio_url`，以及视频的 `video_url` / `input_video`。

配置接受 `video`，walker 也能识别中性视频内容，但目前没有入口协议解码器接受视频内容部分，因此在数据平面上 `video` 规则尚不能匹配。它可以通过 classify/eval API 匹配。

## 可观测性 {#observability}

匹配规则暴露在 `x-vsr-matched-input-modality` 响应头、classify/eval 响应的 `input_modality` 字段、router-回放记录，以及标准的每信号 Prometheus 指标（`llm_signal_match_total{signal_type="input_modality"}`）中。

## 相关信号 {#related-signals}

- [`modality`](../learned/modality.md) — 预期输出模态（`AR`、`DIFFUSION`、`BOTH`）。
- 带 `query_modality: image` 的 [`embedding`](../learned/embedding.md) — 图像内容的语义分类。
- [`conversation`](conversation.md) — 请求形态事实；其 `image_content` 源仍受支持，并共享同一底层图像计数。
