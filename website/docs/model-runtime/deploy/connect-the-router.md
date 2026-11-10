---
title: Connect the Router to the model runtime
sidebar_label: Connect the Router
sidebar_position: 1
description: Declare model_runtime deployments, let the Router start and supervise their workers or attach to a runtime you run, bind signals to them, and publish a native API.
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'

# Connect the Router to the model runtime

Every feature that needs a model gets it from a **deployment**: a named model
under `global.model_catalog.deployments`. A `model_runtime` deployment either
runs in a worker the Router starts and supervises, or attaches to a runtime you
run yourself. Bindings and public model names identify the deployment,
wherever its workers run.

<Tabs groupId="runtime-owner">
<TabItem value="managed" label="Managed by the Router" default>

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      decision-edge:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        device: auto
```

Without `endpoint`, the Router starts a worker for the deployment and restarts
it if it exits.

</TabItem>
<TabItem value="attached" label="Attached to a runtime you run">

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      decision-shared:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        endpoint: http://decision-runtime.example.com:8100
        model: d3-edge
```

With `endpoint`, the Router attaches to a runtime it does not start, restart or
stop; see [Attach to a runtime you run](#attach-to-a-runtime-you-run).

</TabItem>
</Tabs>

:::tip You may not need any of this
Built-in features already know which model they need. Configure a deployment
yourself only to put a model on a GPU, pin a different model, place or scale
its replicas, or attach a shared worker.
:::

## Engine and Router modes

The frontend owns API listeners, authentication, configuration and model
resources. Router and Engine are two capabilities of the same frontend:

- **Engine** serves native System One questions without Chat backends or a
  user-authored routing configuration.
- **Router** additionally runs signals, decisions, algorithms and Chat routing.

```bash
vllm-sr serve vllm-sr/d3-edge -e
vllm-sr serve --config config.yaml
```

- Startup sets `global.router.enabled`: every `serve` invocation without `-e`
  starts Router mode; `-e` starts Engine mode and preserves saved routing
  without initializing its signals or Chat backends.
- Dashboard displays this startup mode and manages model deployments
  independently. It does not switch modes.
- The native API is available in either mode. Listener keys and native model
  grants still apply.
- On the docker target, Engine mode starts only the Router container unless
  `--dashboard`, `--observability` or `--no-minimal` asks for more.

## Built-in features need no configuration

When you turn on a built-in feature, such as a `domain` signal or the semantic
cache, the Router already knows which model it needs. Judgment tasks share the
decision model; embeddings use their own deployment
([Choose a model for a signal](../../installation/choose-a-model-for-a-signal.md)).

Which workers start:

- Deployments that an enabled signal or feature resolves to, or that a
  published native model grant names, are loaded.
- Engine mode also loads the decision model. In Router mode, the decision model
  is loaded when something resolves to it.
- Other declarations do not start workers.

To choose CPU or GPU placement, configure the deployment's `device` or
`replicas`, as described below and in
[Place and scale workers](./placement-and-scaling.md).

## Describe a deployment

Write a deployment to choose the model, its engine arguments and worker
placement:

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      vela-domain:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: cpu
        input:
          max_tokens: 512
          overflow: reject
      decision-edge:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        device: auto
```

| Field | Meaning |
| --- | --- |
| `provider` | Always `model_runtime` for models the runtime serves. |
| `artifact` | A Hugging Face repository, or an absolute path to a local copy. |
| `revision` | The exact 40-character commit to load. Built-in models are already pinned; other repositories need one. |
| `device` | `auto` (default), `cpu`, `cuda:N`, `rocm:N`, `xpu:N`, `npu:N`, `mps`, or an accelerator a plugin adds. |
| `max_num_seqs`, `max_num_batched_tokens`, `batch_window_ms`, `dtype`, `enable_shared_context`, `max_model_len` | Engine arguments: how the worker batches requests, the precision it computes in and a Decision model's input limit. Leave them out for the runtime's defaults. `max_model_len` applies to a managed worker without replicas. See [Engine arguments](./engine-args.md). |
| `input` | For task models: the longest input in tokens (`max_tokens`) and what to do with longer input (`overflow`: `reject`, `truncate` or `window`). |
| `replicas` | Explicit managed `device` or attached `endpoint`/`model` placements for this model. Omit for one worker. |
| `endpoint` | Attach to a runtime you run instead of starting one. |
| `model` | The model's ID on an attached runtime, as its `GET /v1/models` lists it (default: the deployment name). |

An attached runtime also takes `auth`, `headers`, `timeout_ms` and `tls`; the
[model catalog reference](../../installation/model-catalog.md#connection-block)
lists every field.

### Bind a signal to a deployment

Connect a signal to the deployment with a **binding**, keyed by its signal
type:

```yaml title="config.yaml"
global:
  model_catalog:
    bindings:
      domain: vela-domain
```

- A binding under `global.model_catalog.bindings` applies everywhere. A recipe
  can override it under its own `recipes[].model_bindings`.
- Rules that ask their own question (`decision`, `classifiers`, `safety`,
  `complexity`) name their deployment on the rule instead; see
  [Choose a model for a signal](../../installation/choose-a-model-for-a-signal.md).
- The signal decides which kind of answer it reads: a semantic judgment, label
  probabilities, independent label scores, text spans, vectors or relevance
  scores. A binding names only the deployment.

The Router checks bindings against the model's description: heads, labels,
dimensions, question types and input limits. A known mismatch is rejected.

:::note
For custom questions to an offline attached runtime, metadata validation is
deferred until a ready model card is observed; requests cannot run an
unchecked task. See [readiness](./readiness-and-monitoring.md#when-a-model-is-not-ready).
:::

## Attach to a runtime you run

Start a runtime anywhere the Router can reach, then point a deployment at it:

```bash title="On the runtime host"
vllm-srun serve vllm-sr/d3-edge --device rocm:0 --host 0.0.0.0 --port 8100
```

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      gpu-edge:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        endpoint: http://gpu-runtime.example.com:8100
        model: d3-edge
```

- `model` is the ID the runtime lists in `GET /v1/models`. A runtime lists a
  published model under its repository name without the organization, here
  `d3-edge`, unless `--served-model-name` gives it another one.
  Without `model` the Router asks for the deployment name, and a name the
  runtime does not serve leaves the deployment unavailable.
- `endpoint` takes `http://host:port`, `https://host:port` or a Unix socket,
  `unix:///path/to/runtime.sock`.
- The Router does not start, restart or stop an attached runtime; it checks its
  health and treats it as unavailable while it is down.
- Run an attached runtime from the same release as the Router. The Router
  accepts only a runtime that serves exactly its contract version and reports
  any other as `incompatible`, with both versions in the reason.

:::warning
A runtime started this way listens on `127.0.0.1` unless you pass `--host`. It
has no authentication of its own: expose it only on a private network.
:::

On a large host, give such a runtime `--threads`, or run it in a cpuset, when
it serves a model on the optional ONNX Runtime engine (an ONNX package, or an
Omni bundle): each graph of that model runs its own pool of up to `--threads`
CPU threads, and without the option up to every CPU the process may run on (an
Omni bundle has four graphs). Managed CPU workers receive `--threads` from the
[CPU budget](./placement-and-scaling.md#cpu-threads).

To share GPU models between several Routers in a cluster, see
[Run on Kubernetes](./kubernetes.md#share-a-runtime-between-routers).

## Publish a native API

With a `primary` deployment of `vllm-sr/vela3-fast`, add an explicit grant to
a standalone listener so clients can ask it questions directly:

```yaml title="config.yaml"
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    systemone:
      models: [vllm-sr/vela3-fast]
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
```

- Use `GET /v1/systemone/models` to discover published native models, then send
  `POST /v1/systemone` or `POST /v1/decisions` with that `model` ID.
- A deployment uses its Hub artifact ID unless you set `public_name`; local
  artifacts need an explicit public name.
- Add listener `api_keys` when clients must authenticate.
- The native grant is separate from the Chat `listeners[].models` allowlist and
  works in either startup mode. An existing grant stays unchanged when a model
  is replaced or scaled; update it when you intend to publish another model.

See the [quickstart](../get-started/quickstart.md#3-send-a-request) for a
complete native request. Publication does not guarantee readiness; inspect it
with `vllm-sr instance --config config.yaml models`.
