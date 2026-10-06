---
title: Quickstart
description: Install the model runtime, serve a model, send it a request, and use it from the router.
---

# Quickstart

In about ten minutes you install the model runtime, serve a model on your
CPU, ask it a question, and then let the router use the same model for
routing.

You need Linux or macOS, Python 3.10 or newer, and about 3 GB of free disk
for the model download. No GPU is needed.

## 1. Install

The runtime is the `vllm-srun` Python package, released together with the
`vllm-sr` CLI and with the same version. Install PyTorch first, from the
[index that matches your hardware](https://pytorch.org/get-started/locally/)
(CPU, CUDA or ROCm), then the CLI with its `runtime` extra. This guide uses
the CPU build, in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "vllm-sr[runtime]"
```

On ROCm, answers byte-identical to the released packages are guaranteed in the
router images, which carry the release's own PyTorch build. A `pip` install
with the official PyTorch wheel serves the same models, but its answers can
differ slightly ([Choose a model](./choose-a-model.md#hardware)). A repository
checkout is needed only to develop the runtime (`make model-runtime-install`).

Check that the runtime sees its built-in models:

```bash
vllm-srun models
```

## 2. Serve a model

Start Decision 2.0 Kai, the smallest decision model. A decision model answers
questions you write in plain language.

```bash
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
```

The first start downloads the model (about 1.5 GB) into your Hugging Face
cache and checks every file against its pinned hash. The model is ready when
its health check passes. In a second terminal:

```bash
curl -s localhost:8100/health
```

It answers `{"status": "ready", ...}` once the model has loaded and passed its
self-check. Until then it answers HTTP 503 with the current stage.

## 3. Send a request

Ask two questions about one request at once: which kind of work it is
(a **choice**), and whether it needs multi-step reasoning (a **yes/no**
question, called **noul**).

```bash
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of work is this?",
             "criteria": {"code": "Writing or fixing code", "math": "Mathematics", "chat": "Anything else"}},
    "reasoning": {"type": "noul", "instructions": "Does answering this need multi-step reasoning?"}
  }
}'
```

The answer for `kind` names the chosen option and the probability of every
option. The answer for `reasoning` is the probability that the answer is yes:

```json title="Response"
{
  "model": "Decision-2.0-Kai-0.6B",
  "answers": {
    "kind": {"type": "choice", "choice": "code", "probabilities": {"code": 0.504, "math": 0.133, "chat": 0.362}, "confidence": 0.106},
    "reasoning": {"type": "noul", "noul": 0.519}
  },
  "usage": {"input_tokens": 168, "output_tokens": 0}
}
```

Add `"options": {"return_meta": true}` to the request to also get `meta`: the
revision, profile and device that answered and how long it took. On 16 CPU
cores this request takes about 0.2 seconds. `GET /v1/models` shows what is loaded, where it runs and whether it
passed its self-check.

The same command serves classifiers. Stop the server with Ctrl-C and serve the
Vela Domain classifier instead:

```bash
vllm-sr serve vllm-sr/Vela-1.0-Encoder-307M-Domain --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["What is the derivative of x squared?"]}'
```

The result lists the most likely domain (`label`) and the probability of each
of the 14 domains.

## 4. Use it from the router

The router runs models for you. Name the model as a **deployment** with
`provider: model_runtime`, then ask it questions in a `decision` signal. Save
this as `config.yaml`, replacing `vllm:8000` with an OpenAI-compatible backend
that answers your users:

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
          endpoint: vllm:8000
          protocol: http
routing:
  modelCards:
    - name: answer-model
  signals:
    decision:
      - name: needs_reasoning
        deployment: decision-kai
        question:
          type: noul
          instructions: Does answering this request need multi-step reasoning?
        predicate:
          gte: 0.7
  decisions:
    - name: think-first
      priority: 100
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: decision
            name: needs_reasoning
      modelRefs:
        - model: answer-model
          use_reasoning: true
    - name: default-route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: answer-model
          use_reasoning: false
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
```

Validate the file and start the router:

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

The router starts its own runtime for `decision-kai`. The first time, it
downloads its own copy of the model into the `models/` directory next to your
`config.yaml` and keeps it there for later starts. Send a request through the
router and look at which route it took:

```bash
curl -s -D - -o /dev/null localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' -H 'x-vsr-debug: true' \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Plan a three-step proof that there are infinitely many primes."}]}' \
  | grep -i '^x-vsr-'
```

`x-vsr-selected-decision` names the route, and
`x-vsr-matched-decision-model` lists the decision signals that matched. While
the model is still loading, the signal is unknown and `on_unknown: no_match`
sends requests to `default-route`.

To reuse the server you started in step 2 instead of a second copy of the
model, replace `artifact` and `device` with its address, for example
`endpoint: http://host.docker.internal:8100`, and start that server with
`--host 0.0.0.0` so the router container can reach it.

## Next steps

- [Choose a model, size and hardware](model-runtime/choose-a-model.md)
- Turn on a built-in feature: [classify requests](./guides/classify.md),
  [detect PII](./guides/pii.md), [stop prompt attacks](./guides/safety.md),
  [use embeddings](./guides/embeddings.md)
- [Run it with the router](./deploy.md): GPUs, Kubernetes, sharing a runtime
- [Troubleshooting](model-runtime/troubleshooting.md)
