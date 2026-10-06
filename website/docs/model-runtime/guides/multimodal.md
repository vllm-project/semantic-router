---
title: Images and audio
description: Route requests on the images and audio they carry with the Vela Omni embedding models.
---

# Images and audio

Vela 1.0 Omni embeds text, images and audio into one vector space, so an
image can be compared with a text description ("a photograph of a passport
page") or with example images. The router uses it for
[embedding signals](tutorials/signal/learned/embedding.md) whose
`query_modality` is `image` or `audio`.

| Model | Size | Vector | Text limit | Use when |
| --- | --- | --- | --- | --- |
| `vllm-sr/Vela-1.0-Omni-Nano` | 164M | 384 | 512 tokens | Fast routing on CPU |
| `vllm-sr/Vela-1.0-Omni-Mini` | 1.36B | 768 | 32,768 tokens | Higher accuracy, longer text |

Audio of up to 30 seconds is accepted at its original sampling rate.

## Turn it on

Select the multimodal embedding model and add a signal that reads images:

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        multimodal_model_path: models/vela-1.0-omni-nano
        embedding_config:
          model_type: multimodal
routing:
  signals:
    embeddings:
      - name: identity_documents
        threshold: 0.29
        aggregation_method: max
        query_modality: image
        candidates:
          - photograph of a passport page
          - photograph of a driver's license or national ID card
  decisions:
    - name: private-images
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: identity_documents
      modelRefs:
        - model: private-vision-model
```

The router runs Omni on the CPU in the runtime. The thresholds of the
[image routing pack](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml)
are calibrated for Omni Nano; check them again on your own images when you
switch to Mini.

## Check it

Router images include the Omni Nano bundle, so the router needs nothing else.
The Hugging Face repositories hold the source model, not the files the runtime
serves, so to call Omni yourself, build its bundle once from the repository
root and point the runtime at it:

```bash
docker buildx build -f tools/models/vela_omni/Dockerfile \
  --build-arg VELA_OMNI_VARIANTS=nano --output type=local,dest=./omni .
export VLLM_SR_RUNTIME_PREPARED_DIR="$PWD/omni"
vllm-sr serve vllm-sr/Vela-1.0-Omni-Nano --device cpu --port 8100
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": [{"type": "text", "text": "a photograph of a passport page"}]}'
```

Images go in as `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
and audio as `{"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}`.
`GET /v1/models` lists the modalities a model accepts under `embedding.modalities`.
