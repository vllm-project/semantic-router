---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/image-gen.md"
  outdated: false
---

# Image Generation

## 概览

`image_gen` 是路由局部插件：将已匹配路由交给图像生成后端。

对应 `config/plugin/image-gen/basic.yaml`。

## 主要优势

- 多模态或图像生成行为局部在路由。
- 后端细节在配置中清晰暴露。
- 单一路由器可同时承载文本与图像路由而不混淆行为。

## 解决什么问题？

部分路由不应走标准 chat-completions 流程。`image_gen` 为需要图像生成的路由显式表达交接。

## 何时使用

- 已匹配路由应调用图像生成后端
- 路由需要后端专用生成设置
- 纯文本路由应保持不受影响

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: image_gen
  configuration:
    enabled: true
    backend: vllm_omni
    backend_config:
      base_url: http://image-router:8005
      model: Qwen/Qwen-Image
      num_inference_steps: 28
      cfg_scale: 4.5
```
