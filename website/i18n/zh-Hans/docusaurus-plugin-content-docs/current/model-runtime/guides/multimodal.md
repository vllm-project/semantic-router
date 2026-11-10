---
title: 图像和音频
description: 在模型运行时上运行 Vela Omni 嵌入模型，它能嵌入文本、图像和音频。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/model-runtime/guides/multimodal.md"
  outdated: false
---

# 图像和音频 {#images-and-audio}

Vela 1.0 Omni 把文本、图像和音频嵌入同一个向量空间，因此图像既能和文字描述（“一张护照页的照片”）比较，也能和示例图像比较。它由模型运行时提供服务。router 只为召回嵌入文本：它可以把 Omni 部署指定为某个存储的 `embedding.deployment`；要按图像的内容做决策，则向接受图像的部署提出一个读取图像（`input.image`）的 [systemone](tutorials/signal/systemone.md) 问题。

| 模型 | 大小 | 向量 | 文本上限 | 适用场景 |
| --- | --- | --- | --- | --- |
| `vllm-sr/Vela-1.0-Omni-Nano` | 164M | 384 | 512 token | 在 CPU 上快速嵌入 |
| `vllm-sr/Vela-1.0-Omni-Mini` | 1.36B | 768 | 32,768 token | 更高的准确度、更长的文本 |

音频按原始采样率接收，最长 30 秒。

## 启用 {#turn-it-on}

声明一个 Omni 部署，并在用它嵌入文本的存储中指定它，这里是使用 Nano 384 维向量的向量库：

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
  stores:
    vector_store:
      enabled: true
      backend_type: memory
      embedding: {deployment: omni-nano, dimension: 384}
```

向量库需要时，router 才在运行时中把 Omni 跑在 CPU 上。要使用 Mini，改为声明一个 `vllm-sr/Vela-1.0-Omni-Mini` 的 `model_runtime` 部署。

## 检查 {#check-it}

运行时按固定的 revision 从 Hugging Face 仓库下载 Omni，并像对每个内置模型一样校验它读取的每个文件；Nano 为 0.67 GB，Mini 为 4.3 GB。要自己调用 Omni，从仓库检出中安装带 `multimodal` extra 的运行时（它用 Pillow 读取图像），然后启动服务：

```bash
pip install "./src/model-runtime[multimodal]"
vllm-srun serve vllm-sr/Vela-1.0-Omni-Nano --device cpu --port 8100
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": [{"type": "text", "text": "a photograph of a passport page"}]}'
```

图像以 `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` 传入，音频以 `{"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}` 传入。`GET /v1/models` 在 `embedding.modalities` 下列出模型接受的模态。

## ONNX Runtime（可选） {#onnx-runtime-optional}

运行时也可以在可选的 `onnxruntime` 引擎上以 ONNX 图的形式运行 Omni。Router 镜像不包含 ONNX Runtime，所以这适用于你自己的安装和镜像。从仓库根目录构建一次模型包，安装 `onnx` extra，然后运行该包目录：

```bash
docker buildx build -f tools/models/vela_omni/Dockerfile \
  --build-arg VELA_OMNI_VARIANTS=nano --output type=local,dest=./omni .
pip install "./src/model-runtime[multimodal,onnx]"
vllm-srun serve "$PWD/omni/vela-1.0-omni-nano" --engine onnxruntime --device cpu --port 8100
```

在 router 配置中，部署把包目录写作它的 `artifact`，并设置 `engine: onnxruntime`；构建 router 镜像时加上 `--build-arg MODEL_RUNTIME_EXTRAS=multimodal,onnx`，并把模型包复制进镜像。
