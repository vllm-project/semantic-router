---
title: In-Process Inference
description: Run Router classifiers and embeddings with owned local model instances.
---

# In-process inference

Use a local deployment when the Router should execute the model in its own
process. Each task receives a prepared handle with a known input and output
contract. Candle and ORT can coexist; choosing an engine does not replace the
other binding for the whole process.

## Run a generic sequence classifier

Provide a complete compatible sequence checkpoint at `models/email-classifier`.
This example assumes its label order is `BENIGN`, `PHISHING`; use the actual
labels of your trained model. A PEFT adapter directory alone is not a complete
checkpoint. The local image must contain the Candle binding, and any enabled
maintained default models must also be provisioned.

Save this canonical configuration as `config.yaml`. Replace the downstream
endpoint with your reachable answer model:

```yaml
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: 127.0.0.1:8000
          protocol: http
routing:
  model_bindings:
    classifier.email-risk:
      deployment: email-risk-cpu
      contract: label_distribution.v1
      adapter: auto
  signals:
    classifiers:
      - name: email-risk
        type: local
        labels: [BENIGN, PHISHING]
  decisions:
    - name: inspect-email
      priority: 100
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: classifier
            name: email-risk
            label: PHISHING
            predicate:
              gte: 0.8
      modelRefs:
        - model: answer-model
global:
  model_catalog:
    deployments:
      email-risk-cpu:
        artifact: models/email-classifier
        provider: candle
        device: cpu
        precision: native
        input:
          max_tokens: 512
          overflow: reject
```

```bash
vllm-sr config validate --config config.yaml
make vllm-sr-dev
vllm-sr serve --config config.yaml --image-pull-policy never
curl -sS http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Review this email requesting a password reset."}]}'
```

Validation checks the declaration; startup checks the real checkpoint, labels,
and engine. Both matching and nonmatching requests use the configured answer
model in this minimal example. The decision shows how to consume a model score;
add your intended model selection or plugin action separately.

## Select ORT for a compatible graph

For an exported mmBERT sequence graph, replace only the deployment and adapter
in that example:

```yaml
global:
  model_catalog:
    deployments:
      email-risk-cpu:
        artifact: models/email-classifier-onnx
        provider: ort
        device: cpu
        precision: native
        input:
          max_tokens: 512
          overflow: reject
routing:
  model_bindings:
    classifier.email-risk:
      deployment: email-risk-cpu
      contract: label_distribution.v1
      adapter: mmbert
      head: onnx/model.onnx
```

The artifact must contain the graph's actual tokenizer/config and any external
tensor files. For the owned AMD path, use `device: migraphx:0` in an image with
the matching runtime libraries. `precision: fp16` is an explicit MIGraphX
conversion request, not the default. See [Engines and hardware](engines-and-hardware.md).

## Share an encoder only when it is actually shareable

Candle's modern BERT family can load one headless backbone and bind compatible
sequence or token heads. The head directory supplies its real head weights,
config, and tokenizer metadata. The runtime validates compatibility and shares
the physical encoder only when resource identity agrees. Different recipe
handles have independent lifetimes and task interpretation.

ORT heads are complete graphs. Two graphs are not treated as a shared encoder
merely because they came from the same training run. Independent artifacts,
devices, precision, or incompatible task options remain independent resources.

Combined classification executes real task forwards for each input. It does
not promise a joint forward or a vectorized cross-task batch. The legacy
traditional “unified” placeholder is not a supported inference capability.

## LoRA checkpoints

The supported Router LoRA composition uses complete merged BERT or modern
BERT-family checkpoints for its sequence/token tasks. BERT uses its merged
`bert_lora` adapter path. Unknown architectures and unmerged adapter-delta
artifacts fail capability checks; a directory name containing “merged” does
not establish its file format.

The low-level Qwen multi-LoRA implementation reports that loaded adapter
deltas are not applied by its forward path. It is not registered as a Router
local model-binding task. Do not use that API or a PEFT-only directory as
proof that a fine-tuned Router classifier is executing. See the
[training guide](../../training/mmbert-safety-classifier.md) for checkpoint
preparation and validation.

## ML selectors and NLP

KNN, K-means, SVM, and MLP consume embedding features under the existing
`global.router.model_selection.ml` configuration and decision algorithm.
They do not use classifier deployment bindings. A trained selector must use
the same feature space as the runtime embedding provider.

The MLP selector loads a pretrained JSON artifact from
`global.router.model_selection.ml.mlp.pretrained_path`. Router construction
currently uses CPU even though a low-level GPU constructor and a `device`
configuration field exist. Changing that field does not enable GPU execution.
For a trained 768-dimensional selector, its global settings are:

```yaml
global:
  router:
    model_selection:
      ml:
        models_path: models/model-selection
        embedding_dim: 768
        mlp:
          pretrained_path: models/model-selection/mlp_model.json
```

Select `algorithm: {type: mlp}` on the consuming decision and supply eligible
model references and a matching embedding provider. The
[MLP guide](../../tutorials/algorithm/selection/mlp.md) covers training and
candidate selection.

BM25 and N-gram keyword matching use owned `nlp-binding` handles on CPU.
Their relevance scores retain their own units. Prompt compression combines
classical TextRank, position, TF-IDF, and novelty scoring in Go; its estimated
token budget does not expand a native classifier's real tokenizer limit.
Keyword heuristics and compression retain their existing signals/modules; they
are not automatically converted into transformer deployments. See
[Keyword signals](../../tutorials/signal/heuristic/keyword.md) and
[Model selection](../../tutorials/algorithm/overview.md).
