---
title: OpenVINO models
description: Serve exported text encoders with the shared Router Model catalog.
---

# OpenVINO Router Models

Use an OpenVINO-enabled build to serve an exported text embedding or sequence
classifier through the shared Router Model catalog. Recipes inherit the global
binding; an explicit recipe binding selects an isolated override.

```yaml
global:
  model_catalog:
    deployments:
      text-encoder:
        artifact: models/my-openvino-export
        provider: openvino
        device: CPU
        precision: native
        input:
          max_tokens: 512
          overflow: reject
    bindings:
      embedding:
        deployment: text-encoder
        contract: embedding.v1
        adapter: bert
```

The artifact directory needs `config.json`, `tokenizer.json`, and the exported
`openvino_model.xml`/`.bin` and `openvino_tokenizer.xml`/`.bin` pairs. The IR files
may also live in an `openvino/` subdirectory. Use a local derived artifact for
locally converted graphs; do not label it as an unchanged Hugging Face revision.
A classifier additionally needs a contiguous `id2label` map in `config.json`.

Declare the export's actual `max_position_embeddings` and `pad_token_id`.
A finite `tokenizer_config.json` `model_max_length` can further restrict capacity.
The deployment budget cannot exceed those limits. The provider reads suffix
special-token IDs from tokenizer metadata and retains them when truncating.

**Export the tokenizer IR without internal truncation.** Initialization checks
the compiled tokenizer graph and rejects truncation clamps, slicing, fixed-length
padding, and operators whose length behavior is not supported. A
`tokenizer.json` setting alone is insufficient. With `openvino_tokenizers`,
disable the source tokenizer's truncation and set `model_max_length=None`
before conversion. Verify counts on inputs beyond the export's declared model
capacity. A graph rejected by this check needs a compatible export; there is no
silent fallback to an unverified count.

The provider supports `reject` and `truncate`, full text vectors, and categorical
sequence classification. It does not provide layer selection, vector cropping,
windowed inference, image/audio encoders, or token classification. Requested
unsupported views fail preparation or inference. Precision follows the exported
IR; the adapter does not add vector normalization. OpenVINO device selectors such
as `CPU`, `GPU.0`, or `AUTO` are passed to OpenVINO explicitly.

A declared binding loads only when a configured consumer needs it. Compatible
consumers and reload generations share the same owned handle; different graphs
or execution policies retain separate resources.
