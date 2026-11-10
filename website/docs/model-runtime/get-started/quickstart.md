---
title: Quickstart
sidebar_position: 2
description: Install the model runtime, serve a model, send it a request, and use it from the router.
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'

# Quickstart

In about ten minutes you install the `vllm-sr` CLI, serve a decision model,
ask it a question, and then let the Router use the same model for routing.

**You need** Linux, macOS or WSL2 with Docker or Podman, Python 3.10 or newer,
12 GiB of memory, and about 9 GB of free disk for the Router image and the
model download. No GPU is needed.

:::note
Engine mode (`vllm-sr serve ARTIFACT --engine`) is newer than the 0.4.0
release; see the [release channel note](../../installation/installation.md).
:::

## 1. Install

Install the development channel, which includes Engine mode, into the
installer's isolated virtual environment. `--no-launch` leaves startup for the
next step:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --no-launch
```

The model runtime, the `vllm-srun` Python package, ships only inside the Router
images, so nothing else is installed on your machine. The CLI pulls the image
on first use. The images carry the release's own PyTorch build, so their
answers are the ones the models were released with
([Choose a model](./choose-a-model.md#hardware)).

## 2. Serve a model

Start Vela 3.0 vela3-fast, the Router's default decision model, which runs on
a CPU in FP32. A decision model answers questions you write in plain language.

<Tabs groupId="platform">
<TabItem value="cpu" label="CPU" default>

```bash
vllm-sr serve vllm-sr/vela3-fast --engine --platform cpu
```

On macOS the runtime runs on the CPU, because containers there get no GPU.

</TabItem>
<TabItem value="rocm" label="AMD GPU">

```bash
vllm-sr serve vllm-sr/vela3-fast --engine --platform rocm --device-ids 0
```

The `vllm-sr-rocm` image is a 6.5 GB download on first use. `--device-ids N`
selects a host GPU at startup; `rocm:N` picks one in canonical YAML.

vela3-fast's reference answers are recorded for CPUs only, so on a GPU its
self-check reports
[`unverified`](../deploy/troubleshooting.md#a-ready-models-self-check-says-unverified).

</TabItem>
<TabItem value="cuda" label="NVIDIA GPU">

```bash
vllm-sr serve vllm-sr/vela3-fast --engine --platform cuda --device-ids 0
```

`--platform cuda` selects the `vllm-sr-cuda` image and passes the GPUs
through.

vela3-fast's reference answers are recorded for CPUs only, so on a GPU its
self-check reports
[`unverified`](../deploy/troubleshooting.md#a-ready-models-self-check-says-unverified).

</TabItem>
</Tabs>

In Engine mode the CLI starts only the persistent frontend and a managed model
worker, in one Router container:

- The first start downloads the model; later starts reuse it.
- Startup prints a short summary ending in a `Ready:` line with the System One
  URL (`--verbose` prints every detail).
- `vllm-sr status` inspects the stack and `vllm-sr stop` stops it.
- `--dashboard` adds the Dashboard, `--observability` adds Jaeger, Prometheus
  and Grafana, and `--no-minimal` starts the whole stack.

In a second terminal, list the public native models:

```bash
curl -s localhost:8899/v1/systemone/models
```

:::tip Larger and multimodal models
Decision 3.0 models also answer questions about images and videos. Start the
smallest, d3-edge, on an AMD GPU with
`vllm-sr serve vllm-sr/d3-edge -e --platform rocm`, and send images and videos
as described in
[Router and Engine modes](https://github.com/vllm-project/semantic-router/blob/main/src/vllm-sr/INSTANCE_MODES.md#ask-about-images-and-videos).
`vllm-sr/d3-turbo` reads each input once for all of its questions and also
finds things in images and videos
([Box and track questions](https://github.com/vllm-project/semantic-router/blob/main/src/vllm-sr/INSTANCE_MODES.md#find-things-in-images-and-videos)),
and `vllm-sr/d3-ultra` reads a text request's question list in front of its
state. [Supported models](../supported-models/overview.md) lists every model.
:::

<details>
<summary>Where Engine mode keeps its state, and which models it publishes</summary>

- Without `--config`, and with no `config.yaml` in the working directory,
  Engine mode keeps each model's configuration, state and download in
  `~/.local/state/vllm-sr/engine/<model>`.
- The initial Engine configuration publishes the selected model on the default
  listener. An existing config keeps its explicit
  `listeners[].systemone.models` allowlist and API keys; restarting in another
  mode or selecting a model never broadens it.
- Every start without `--engine` uses Router mode; MODEL alone does not select
  Engine mode. Bare `vllm-sr serve` preserves the configured default judgment
  deployment, or initializes Vela 3.0 vela3-fast in a new configuration.

</details>

## 3. Send a request

Ask two questions about one request at once: which kind of work it is (a
**choice**), and whether it needs multi-step reasoning (a **yes/no** question,
called **noul**).

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/vela3-fast",
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
  "model": "vllm-sr/vela3-fast",
  "answers": {
    "kind": {"type": "choice", "choice": "code", "probabilities": {"code": 0.98, "math": 0.005, "chat": 0.015}, "confidence": 0.97},
    "reasoning": {"type": "noul", "noul": 0.457}
  },
  "usage": {"input_tokens": 68, "output_tokens": 0}
}
```

On 16 cores of an AMD EPYC 9575F this request takes about 0.1 seconds.

- Add `"options": {"return_meta": true}` to also get `meta`: the revision,
  precision (`dtype`) and device that answered, whether batching can move its
  answers (`numerics`), and how long it took.
- `POST /v1/decisions` takes the same requests and the same explicit public
  model ID. It answers a failed question in place, where `/v1/systemone`
  answers the request as a whole, as System One does.
- Native discovery uses `/v1/systemone/models`. `/v1/models` remains Chat
  discovery, and in Engine mode it also lists the System One models as
  `models: [{name, description, release_date}]`.

The worker's other APIs are separate; see the
[HTTP API reference](../reference/http-api.md). Compressed bodies, multipart
image uploads, assets, Box, track and long context are in
[System One compatibility](../reference/http-api.md#system-one-compatibility).

## 4. Use it from the Router

The Router runs models for you. Name the model as a **deployment** with
`provider: model_runtime`, then ask it questions in a `systemone` signal. Save
this as `config.yaml`, replacing `host.docker.internal:8000` with an
OpenAI-compatible backend that answers your users:

```yaml title="config.yaml"
version: v0.5
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    systemone:
      models: [vllm-sr/vela3-fast]
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: host.docker.internal:8000
          protocol: http
recipes:
  - name: default
    signals:
      systemone:
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
            - type: systemone
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
        artifact: vllm-sr/vela3-fast
        device: cpu
```

Validate the file, then restart in Router mode using this configuration:

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml --replace-active-config
```

`--replace-active-config` applies the file you just wrote over any
configuration saved for this directory; later restarts can omit it to retain
Dashboard edits. The `primary` deployment answers both the routing question and
direct System One requests, and the listener explicitly publishes its native
model name.

Send a request through the Router and look at which route it took:

```bash
curl -s -D - -o /dev/null localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' -H 'x-vsr-debug: true' \
  -d '{"model": "vllm-sr/auto", "messages": [{"role": "user", "content": "Plan a three-step proof that there are infinitely many primes."}]}' \
  | grep -i '^x-vsr-'
```

`x-vsr-selected-decision` names the route, and the `signals` member of
`x-vsr-trace` lists the `systemone` signals that matched.

:::note
If the runtime restarts later, the signal is unknown until the model is back,
and `on_unknown: no_match` sends requests to `default-route` meanwhile.
:::

Router and Engine mode share one managed instance. To attach an independently
operated worker instead, use an explicit `endpoint` and `model`; see
[Connect the Router](../deploy/connect-the-router.md#attach-to-a-runtime-you-run).

## Next steps

- [Choose a model, size and hardware](./choose-a-model.md)
- Turn on a built-in feature:
  [the application signals](../../tutorials/signal/overview.md#application-signals),
  [your own questions](../../system-one/use/custom-questions.md),
  [embeddings for recall](../guides/embeddings.md)
- [Connect the Router](../deploy/connect-the-router.md) and
  [place and scale workers](../deploy/placement-and-scaling.md): GPUs,
  Kubernetes, sharing a runtime
- [Troubleshooting](../deploy/troubleshooting.md)
