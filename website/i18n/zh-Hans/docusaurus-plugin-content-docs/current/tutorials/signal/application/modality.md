---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/modality.md"
  outdated: false
---

# 模态信号 {#modality-signal}

## 概述 {#overview}

`modality` 读取请求要求的输出：文本、图片、文本加图片、语音、音频或视频。它提出 `modality` [预设](../../../system-one/reference#preset-catalog)问题：“What kind of output does this request ask for?”，有八个选项：`text`、`image`、`text_image`、`image_edit`、`speech`、`audio`、`video` 和 `video_edit`。

## 主要优势 {#key-advantages}

- 把图片生成路由与纯文本路由分开。
- 让输出意图在 `recipes[].decisions` 中可见。
- 读取书面请求，所以纯文本的“画一只猫”也能到达图片路由。

## 解决什么问题？ {#what-problem-does-it-solve}

文本对话、图片生成和混合请求常常共用一个入口，却需要不同的后端。`modality` 给出请求要求的输出，让决策可以把每个请求发给能产出它的模型。

## 何时使用 {#when-to-use}

在以下情况使用 `modality`：

- 一个入口同时服务文本和图片生成后端；
- 某些路由只应接受要求特定输出的请求；
- 编辑请求应交给能读取它所编辑媒体的模型。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      modality:
        - name: wants_image
          labels: [image]
        - name: wants_text_image
          labels: [text_image]
    decisions:
      - name: image
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: modality, name: wants_image}
        output: {kind: image, action: generate}
        modelRefs:
          - model: image-model
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: modality, name: ...}` 条件使用 |
| `labels` | 规则匹配的输出；没有 `labels` 的规则本身不匹配任何内容，由它的条件用 `label` 指定一个 |
| `predicate` | 选中输出的最低概率；默认为模型的校准阈值（vela3-fast 上为 0.63），否则为 0.5 |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

该信号只负责路由。以生成型输出作答的决策要在它的 `output` 契约中声明这种输出（如上面的 `image`），而且它指定的每个模型都必须声明该输出需要的能力（这里是 `image-model` 上的 `image_generation`）；[输出](../../decision/output)介绍了这个契约。之后，Responses API 客户端会收到生成的图片。Images API 客户端（`POST /v1/images/generations`）按提示词路由，必须到达这样的决策，否则会在调用任何模型之前以 `image_route_required` 被拒绝。编辑类标签 `image_edit` 和 `video_edit` 要与 [`input_modality`](../heuristic/input-modality) 条件结合，以确认请求携带了要编辑的媒体；本版本在加载时以 `output_not_executable` 拒绝编辑类输出。

### 回答问题的模型 {#the-model-that-answers}

除非 `modality` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有模态规则；见[选择部署](../../../system-one/deployments)。vela3-fast 和 Vela 3.0 的 d3 规格在其中三个选项 `text`、`image` 和 `text_image` 上训练过（训练形式中为 `AR`、`DIFFUSION` 和 `BOTH`，再映射回来）；需要其他选项的问题以零样本方式提出。它读取当前用户轮次，并截断到模型的输入长度。

为早期版本编写的配置把规则命名为 `AR`、`DIFFUSION` 和 `BOTH`，由名称决定输出。`vllm-sr config migrate` 会给这类规则补上 `labels`。完整示例见 [`config/fragments/signal/modality/multimodal.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/modality/multimodal.yaml)。
