---
translation:
  source_commit: "7c874be2"
  source_file: "docs/tutorials/plugin/image-gen.md"
  outdated: true
---

# 图像生成（Image Generation）

## 概览

`image_gen` 是一个路由局部插件，用于将匹配的路由移交给图像生成后端。

## 主要优势

- 将多模态或图像生成行为限制在对应路由内。
- 在配置中清晰公开后端详情。
- 让一个路由器同时承载文本和图像路由，而不会混淆两种行为。

## 解决什么问题？

有些路由不应遵循标准 Chat Completions 流程。`image_gen` 为需要图像生成的路由明确声明这一移交过程。

## 何时使用

- 匹配的路由应调用图像生成后端
- 路由需要后端专用的生成设置
- 纯文本路由不应受到影响

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: image_gen
    configuration:
      enabled: true
      backend: vllm_omni
      backend_config:
        base_url: http://image-router:8005
        model: Qwen/Qwen-Image
        num_inference_steps: 28
        cfg_scale: 4.5
```

选定的后端会接收图像提示词和生成参数。应使用经过身份验证的可信端点，并在该插件之前应用请求侧安全策略。完整示例见：
[`config/fragments/plugin/image-gen/basic.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/image-gen/basic.yaml)。
