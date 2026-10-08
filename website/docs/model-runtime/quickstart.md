---
title: Quickstart
description: Install the model runtime, serve a model, send it a request, and use it from the router.
---

# Quickstart

In about ten minutes you install the `vllm-sr` CLI, serve a model on your
CPU, ask it a question, and then let the router use the same model for
routing.

You need Linux, macOS or WSL2 with Docker or Podman, Python 3.10 or newer, and
a few gigabytes of free disk for the router image and the model download. No
GPU is needed. Engine mode (`vllm-sr serve MODEL`) is newer than the 0.4.0
release; see the [release channel note](../installation/installation.md).

## 1. Install

Install the CLI into a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install vllm-sr
```

The model runtime, the `vllm-srun` Python package, ships only inside the
router images. `vllm-sr serve MODEL` runs it in a container from the
`vllm-sr` image, which the CLI pulls on first use, so nothing else is
installed on your machine. On a GPU host, `--platform amd` or
`--platform nvidia` selects the `vllm-sr-rocm` or `vllm-sr-cuda` image and
passes the GPUs through; on macOS the runtime runs on the CPU, because
containers there get no GPU. The images carry the release's own PyTorch build,
so their answers are the ones the models were released with
([Choose a model](./choose-a-model.md#hardware)).

## 2. Serve a model

Start Decision 2.0 Kai, the smallest decision model. A decision model answers
questions you write in plain language.

```bash
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
```

On an AMD GPU, the same model runs on the first GPU with:

```bash
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --platform amd --device rocm:0 --port 8100
```

The `vllm-sr-rocm` image is a 6.5 GB download on first use. `rocm:N` picks
another GPU of the host.

The runtime runs in the foreground; Ctrl-C stops it. The first start downloads
the model (about 1.5 GB) into the engine-mode cache, `~/.cache/vllm-sr/models`,
and checks every file against its pinned hash; later starts reuse it. Set
`VLLM_SR_ENGINE_CACHE_DIR` to keep the cache elsewhere. The model is ready when
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
cores this request takes about 0.2 seconds. `POST /v1/systemone` is the same
endpoint under its System One name. `GET /v1/models` shows what is loaded,
where it runs and whether it passed its self-check.

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
`config.yaml` and keeps it there for later starts. `vllm-sr serve` returns once
the model has loaded, and prints its state while it waits. Send a request
through the router and look at which route it took:

```bash
curl -s -D - -o /dev/null localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' -H 'x-vsr-debug: true' \
  -d '{"model": "vllm-sr/auto", "messages": [{"role": "user", "content": "Plan a three-step proof that there are infinitely many primes."}]}' \
  | grep -i '^x-vsr-'
```

`x-vsr-selected-decision` names the route, and
`x-vsr-matched-decision-model` lists the decision signals that matched. If the
runtime restarts later, the signal is unknown until the model is back and
`on_unknown: no_match` sends requests to `default-route` meanwhile.

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
