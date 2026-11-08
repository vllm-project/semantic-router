---
title: 图像和音频
description: 用 Vela Omni 嵌入模型按请求带的图像和音频路由。
translation:
  source_commit: "439c22531380dfd51abf2ce33ce6dc4db87b1fb7"
  source_file: "docs/model-runtime/guides/multimodal.md"
  outdated: true
is_mtpe: true
---

# 图像和音频 {#images-and-audio}

Vela 1.0 Omni 把文本、图像和音频嵌进同一个向量空间，所以图像既能和文字描述（一张护照页的照片）比，也能和示例图像比。router 用它做多模态嵌入（`global.model_catalog.embeddings.multimodal`）。

| 模型 | 大小 | 向量 | 文本上限 | 什么时候用 |
| --- | --- | --- | --- | --- |
| `vllm-sr/Vela-1.0-Omni-Nano` | 164M | 384 | 512 token | CPU 上快速路由 |
| `vllm-sr/Vela-1.0-Omni-Mini` | 1.36B | 768 | 32,768 token | 更准、文本更长 |

音频按原始采样率收，最长 30 秒。

## 打开它 {#turn-it-on}

声明多模态嵌入部署，并在 `embeddings.multimodal` 中指定它；不指定，多模态嵌入就不开：

```yaml
global:
  model_catalog:
    deployments:
      omni-nano:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Omni-Nano
        revision: 2ff2d66385dbdd661a560ec3e8bcb45a0527d92e
        device: cpu
        input: {overflow: truncate}
    embeddings:
      multimodal: omni-nano
```

Omni 由 router 在 runtime 里跑在 CPU 上。要用 Mini，就改为声明一个 `vllm-sr/Vela-1.0-Omni-Mini` 的 `model_runtime` 部署。

## 验一下 {#check-it}

runtime 会从 Hugging Face 仓库按钉住的 revision 下载 Omni，读到的每个文件都校验，跟每个内置模型一样；Nano 0.67 GB，Mini 4.3 GB。想自己调 Omni，就从仓库检出一个环境装上带 `multimodal` extra 的 runtime——它读图像（Pillow）——然后起服务：

```bash
pip install "./src/model-runtime[multimodal]"
vllm-srun serve vllm-sr/Vela-1.0-Omni-Nano --device cpu --port 8100
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": [{"type": "text", "text": "a photograph of a passport page"}]}'
```

## ONNX Runtime（可选） {#onnx-runtime}

runtime 也能在可选的 `onnxruntime` 引擎上把 Omni 当 ONNX 图来服务。router 的镜像不含 ONNX Runtime，所以这是给你自己的安装和镜像用的。从仓库根把包构建一次，装上 `onnx` extra，然后服务这个包目录：

```bash
docker buildx build -f tools/models/vela_omni/Dockerfile \
  --build-arg VELA_OMNI_VARIANTS=nano --output type=local,dest=./omni .
pip install "./src/model-runtime[multimodal,onnx]"
vllm-srun serve "$PWD/omni/vela-1.0-omni-nano" --engine onnxruntime --device cpu --port 8100
```

在 router 配置里，部署把包目录名写进 `artifact`，并设 `engine: onnxruntime`；构建 router 镜像时加 `--build-arg MODEL_RUNTIME_EXTRAS=multimodal,onnx`，并把包拷进去。

图像走 `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`，音频走 `{"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}`。`GET /v1/models` 在 `embedding.modalities` 下列出模型收哪些模态。
