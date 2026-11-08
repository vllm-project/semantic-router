---
is_mtpe: true
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/decision/output.md"
  outdated: false
---

# 输出

## 概述

决策的 `output` 属性声明其回答携带的媒体。没有 `output` 的决策以文本回答。
声明 `output: {kind: image, action: generate}` 的决策以生成的图像回答：当请求只包含文本，
例如“画一座黄昏时的灯塔”，Router 会在选择模型之前为请求加上托管的 `image_generation`
操作，因此请求只会到达能够生成该图像的模型。

输出是决策属性，不是信号、算法或插件。它取代了过去由模型的 `modality` 关键字（`ar`、
`diffusion`、`omni`）和 `image_gen` 插件隐含的输出；仍然使用这两者的配置无法加载，
`vllm-sr config migrate` 会改写其中可以机械改写的部分。

## 它解决什么问题？

把“画一只猫”的请求路由到图像模型还不够：纯文本请求中没有图像 API 可以编码的内容，因此只指定图像模型的路由
会选中一个随后失败的后端。输出契约把回答的媒体纳入决策，在加载时检查它引用的每个模型都能产生该媒体，
并为请求加上产生该媒体的操作。

## 何时使用

- 决策以生成的图像回答：`output: {kind: image}`。
- 同一个模型在同一个响应中既以文本回答又生成图像：`output: {kind: [text, image]}`。

文本回答无需声明，每个决策默认都以文本回答。根据请求自身图像的内容进行路由属于信号，而不是输出契约
（参见[图像输入](#图像输入)）。

## 配置

```yaml
providers:
  models:
    - name: painter
      api_format: images
      capabilities: [image_generation]
      modalities: {input: [text], output: [image]}
      backend_refs:
        - {name: omni-server, provider: vllm, endpoint: vllm-omni:8000}
    - name: omni
      api_format: responses
      capabilities: [chat, image_generation]
      modalities: {input: [text, image], output: [text, image]}
      backend_refs:
        - {name: omni-model, provider: vllm, endpoint: omni:8000}
recipes:
  - name: default
    signals:
      modality:
        - {name: wants_image, labels: [image]}
        - {name: wants_text_image, labels: [text_image]}
    decisions:
      - name: image
        rules: {operator: AND, conditions: [{type: modality, name: wants_image}]}
        output: {kind: image, action: generate}
        modelRefs: [{model: painter}]
      - name: text-and-image
        rules: {operator: AND, conditions: [{type: modality, name: wants_text_image}]}
        output: {kind: [text, image], action: generate}
        modelRefs: [{model: omni}]
```

等价的 DSL：

```dsl
ROUTE image {
  WHEN modality("wants_image")
  MODEL "painter"
  OUTPUT {
    kind: "image"
    action: "generate"
  }
}
```

`kind` 取 `text`（默认）、`image`、`audio` 或 `video`，也可以是由一个模型同时写出的多个类型组成的列表。
`action` 取 `generate`（默认）或 `edit`，后者编辑请求提供的图像或视频。

## 加载检查

决策引用的每个模型都必须在其 `providers.models[]` 条目中声明输出所需的事实：

| 输出 | 模型声明 |
|---|---|
| text（默认） | `modalities.output` 中的 text；除非模型的 modalities 另有说明，每个模型都满足 |
| `image`，`generate` | `capabilities` 中的 `image_generation` |
| `image`，`edit` | `image_generation` 和 `image_input` |
| `audio` | `audio_output` |
| `video` | `video_output`，编辑时还需要 `video_input` |
| 列表 | 同一个模型声明每种类型所需的全部事实 |

缺少其中任何一项的模型会以 `output_capability_missing` 加载失败，错误会指出模型和需要声明的事实。
本版本执行文本、图像生成，以及由一个模型写出的文本加生成图像；图像编辑、音频和视频会以
`output_not_executable` 失败，并指出所需的编解码器，直到该编解码器发布为止。生成的图像还需要一个能携带它的
API：图像 API（`api_format: images`）上的模型返回图像，Responses API（`api_format: responses`）上的模型
返回图像和文本。执行器类算法（例如 `escalate`）只以文本回答，因此不能提供图像。

## 运行时效果

- 托管的 `image_generation` 操作会加入请求的能力需求，因此候选过滤会跳过其 API 或声明的能力无法承载该操作的模型，
  `candidate_requirements` 也会看到它。
- `image` 决策强制执行该操作；`[text, image]` 决策只提供该操作，由模型决定是否调用。
- Responses API 客户端会收到带有图像的 `image_generation_call` 项，以及模型写出文本时的消息项。
- [Images API](../../api/router#images-api) 客户端（`POST /v1/images/generations`）以该 API 的格式收到图像。
  其请求按提示词路由，并且必须到达输出为图像的决策；否则会在调用任何模型之前收到 `image_route_required`。
- Chat Completions 或 Anthropic Messages 客户端无法接收图像：Router 会在调用模型之前以 `unsupported_capability`
  拒绝其请求。`tool_choice` 为 `none` 的请求同样会被拒绝。
- 在调试响应中，`x-vsr-trace` 的 `modality` 成员给出所选决策的输出类型。

## 图像输入

根据请求中图像的内容进行路由，使用 `input.image: true` 的 `systemone` 问题，由能够读取图像的决策模型回答。
问题字段以及能够回答图像问题的模型，请参阅 [System One](../signal/learned/systemone)。
