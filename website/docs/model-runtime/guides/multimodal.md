---
title: Images and audio
description: Serve the Vela Omni embedding models, which embed text, images and audio, on the model runtime.
sidebar_position: 2
---

# Images and audio

Vela 1.0 Omni embeds text, images and audio into one vector space, so an
image can be compared with a text description ("a photograph of a passport
page") or with example images. The model runtime serves it. The router embeds
text only, for recall: it can name an Omni deployment as a store's
`embedding.deployment`, and a decision about what an image shows asks a
[systemone](tutorials/signal/systemone.md) question that reads it
(`input.image`) of a deployment that accepts images.

| Model | Size | Vector | Text limit | Use when |
| --- | --- | --- | --- | --- |
| `vllm-sr/Vela-1.0-Omni-Nano` | 164M | 384 | 512 tokens | Fast embeddings on CPU |
| `vllm-sr/Vela-1.0-Omni-Mini` | 1.36B | 768 | 32,768 tokens | Higher accuracy, longer text |

Audio of up to 30 seconds is accepted at its original sampling rate.

## Turn it on

Declare an Omni deployment and name it where a store embeds text with it,
here the vector store at Nano's 384 dimensions:

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

The router runs Omni on the CPU in the runtime once the vector store needs
it. To use Mini, declare a `model_runtime` deployment of
`vllm-sr/Vela-1.0-Omni-Mini` instead.

## Check it

The runtime downloads Omni from its Hugging Face repository at the pinned
revision and checks every file it reads, as it does for every built-in model;
Nano is 0.67 GB and Mini 4.3 GB. To call Omni yourself, install the runtime
from a repository checkout with its `multimodal` extra, which reads images
(Pillow), and serve it:

```bash
pip install "./src/model-runtime[multimodal]"
vllm-srun serve vllm-sr/Vela-1.0-Omni-Nano --device cpu --port 8100
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": [{"type": "text", "text": "a photograph of a passport page"}]}'
```

Images go in as `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
and audio as `{"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}`.
`GET /v1/models` lists the modalities a model accepts under `embedding.modalities`.

## ONNX Runtime (optional)

The runtime can also serve Omni as ONNX graphs on its optional `onnxruntime`
engine. Router images don't include ONNX Runtime, so this is for your own
installs and images. Build the bundle once from the repository root, install
the `onnx` extra, and serve the bundle directory:

```bash
docker buildx build -f tools/models/vela_omni/Dockerfile \
  --build-arg VELA_OMNI_VARIANTS=nano --output type=local,dest=./omni .
pip install "./src/model-runtime[multimodal,onnx]"
vllm-srun serve "$PWD/omni/vela-1.0-omni-nano" --engine onnxruntime --device cpu --port 8100
```

In a router configuration, a deployment names the bundle directory as its
`artifact` and sets `engine: onnxruntime`; build the router image with
`--build-arg MODEL_RUNTIME_EXTRAS=multimodal,onnx` and copy the bundle into it.
