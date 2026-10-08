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
GPU is needed. Engine mode (`vllm-sr serve ARTIFACT --engine`) is newer than the 0.4.0
release; see the [release channel note](../installation/installation.md).

## 1. Install

Install the CLI into a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install vllm-sr
```

The model runtime, the `vllm-srun` Python package, ships only inside the
router images. `vllm-sr serve ARTIFACT --engine` starts the instance frontend from the
`vllm-sr` image, which the CLI pulls on first use, so nothing else is
installed on your machine. On a GPU host, `--platform rocm` or
`--platform cuda` selects the `vllm-sr-rocm` or `vllm-sr-cuda` image and
passes the GPUs through; on macOS the runtime runs on the CPU, because
containers there get no GPU. The images carry the release's own PyTorch build,
so their answers are the ones the models were released with
([Choose a model](./choose-a-model.md#hardware)).

## 2. Serve a model

Start Decision 2.0 Kai, the smallest decision model. A decision model answers
questions you write in plain language.

```bash
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform cpu
```

On an AMD GPU, the same model runs on the first GPU with:

```bash
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform rocm --device-ids 0
```

The `vllm-sr-rocm` image is a 6.5 GB download on first use. `rocm:N` picks
another GPU of the host in canonical YAML; `--device-ids N` selects a host GPU at startup.

The CLI starts the persistent frontend, Dashboard and a managed model worker.
The first start downloads the model into the instance model cache. Later starts
reuse it. Startup waits for readiness; use `vllm-sr status` to inspect the stack
and `vllm-sr stop` to stop it.

The initial Engine configuration publishes the selected model on the default
listener. An existing config keeps its explicit `listeners[].systemone.models`
allowlist and API keys; restarting in another mode or selecting a model never
broadens it. Every start without `--engine` uses Router mode; MODEL alone does
not select Engine mode. Bare `vllm-sr serve` preserves the configured default
judgment deployment, or initializes Vela 2.0 0.3B in a new configuration.
In a second terminal, inspect the public native model list:

```bash
curl -s localhost:8899/v1/systemone/models
```

## 3. Send a request

Ask two questions about one request at once: which kind of work it is
(a **choice**), and whether it needs multi-step reasoning (a **yes/no**
question, called **noul**).

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/Decision-2.0-Kai-0.6B",
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
  "model": "vllm-sr/Decision-2.0-Kai-0.6B",
  "answers": {
    "kind": {"type": "choice", "choice": "code", "probabilities": {"code": 0.504, "math": 0.133, "chat": 0.362}, "confidence": 0.106},
    "reasoning": {"type": "noul", "noul": 0.519}
  },
  "usage": {"input_tokens": 168, "output_tokens": 0}
}
```

Add `"options": {"return_meta": true}` to the request to also get `meta`: the
revision, profile and device that answered and how long it took. On 16 CPU
cores this request takes about 0.2 seconds in the recorded worker benchmark.
`POST /v1/decisions` is an alias of `/v1/systemone`; both use the same explicit
public model ID. Native discovery uses `/v1/systemone/models`, while `/v1/models`
remains Chat discovery. The worker's classify, embeddings, rerank and bundle
APIs are separate; see the [task guides](./guides/classify.md).

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
        deployment: primary
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
      primary:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
```

Validate the file and start the router:

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

Use the same logical deployment key (`primary` in this example) when you want Router tasks and direct System One requests
to share the same model pool. The frontend keeps managed workers available
when routing is toggled; `global.router.enabled` changes only routing.
To activate this new authored config over the existing Engine state, run
`vllm-sr serve --config config.yaml --replace-active-config`.

Send a request
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

Router and Engine mode share one managed instance. To attach an independently
operated worker instead, use an explicit `endpoint` and `served_name`; the
[deployment guide](./deploy.md) describes its separate worker API and lifecycle.

## Next steps

- [Choose a model, size and hardware](model-runtime/choose-a-model.md)
- Turn on a built-in feature: [classify requests](./guides/classify.md),
  [detect PII](./guides/pii.md), [stop prompt attacks](./guides/safety.md),
  [use embeddings](./guides/embeddings.md)
- [Run it with the router](./deploy.md): GPUs, Kubernetes, sharing a runtime
- [Troubleshooting](model-runtime/troubleshooting.md)
