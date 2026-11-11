---
title: Command reference
sidebar_label: Commands
sidebar_position: 1
description: The options of vllm-sr serve and vllm-srun serve, the other vllm-srun commands, and the models file.
---

# Command reference

`vllm-sr serve` starts the Router's persistent frontend and its managed
runtime workers; `vllm-srun serve` starts one worker directly. This page lists
every option of both, and the models file a worker can read.

## `vllm-sr serve` {#vllm-sr-serve}

`vllm-sr serve` starts the persistent frontend and its managed runtime workers.
Use Engine mode for native System One serving without Chat backends:

```bash
vllm-sr serve vllm-sr/d3-edge -e
```

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL` | configured default, or Vela 3.0 vela3-fast | Override the default judgment deployment's artifact. Other deployments, backend LLMs and public grants stay as authored. |
| `--engine`, `-e` | off | Start Engine mode. Every invocation without this flag starts Router mode, including a restart of a saved Engine configuration. |
| `--config FILE` | `config.yaml` | Canonical listeners, deployments, bindings and routing. Docker preserves the active configuration unless explicitly replaced. |
| `--platform` | `auto` | Execution backend: `auto`, `cpu`, `cuda` or `rocm`. Automatic detection inspects the container host or the selected Kubernetes cluster. |
| `--data-parallel-size`, `-dp` | existing placement, or one worker | Number of independent managed replicas of the default deployment. Writes the canonical `replicas` list. |
| `--device-ids` | existing placement, or available GPUs | Docker host GPU indices. With `-dp N`, supply one ID to share that GPU or N IDs to use one per replica. This does not change GPU visibility. |
| `--max-num-seqs`, `--max-num-batched-tokens`, `--batch-window-ms`, `--dtype`, `--enable-shared-context` | configured value, or the runtime default | The default deployment's [engine arguments](../deploy/engine-args.md), independent of replica count and placement. |
| `--revision` | configured pin, built-in release pin, or repository default | Optional advanced branch, tag or full commit. Resolved once before startup into the immutable commit stored in the active deployment and shared by all replicas. |
| `--image` | platform image | Use a specific local or published frontend image. |
| `--image-pull-policy` | `always` | `always`, `ifnotpresent` or `never`. |
| `--container-runtime` | detected | `docker` or `podman`. |

Omitting a model option preserves its configured value. Selecting a different
artifact drops the previous artifact's revision, then resolves the new one.
A local absolute package path has no Hub revision. Canonical YAML `revision`
continues to require an immutable commit; branch and tag resolution is a CLI
startup operation, not a per-request lookup.

A new multi-GPU placement uses the first N available GPUs. If fewer are
available, specify one `--device-ids` value to request colocated workers
explicitly. Resizing an authored placement repeats its device order. An
existing visibility mask is honored: for example, host GPU 7 maps to runtime
ordinal 0 under `ROCR_VISIBLE_DEVICES=7`. A mask that names GPUs by UUID
(`GPU-...`) also selects the platform with `--platform auto`; `--device-ids`
and new multi-GPU placements still need a mask of host indices. Kubernetes
uses pod allocation ordinals in canonical config and does not accept physical
`--device-ids`.

Listener ports and API grants belong in `listeners`; the initial native API
listener uses port `8899`. Selecting a model never broadens an existing
listener's public model grants. Dashboard reports the startup mode; changing
mode requires another `serve` invocation. Use the positional `MODEL` argument
and `--device-ids` in place of the former `--decision-model` and `--device`
serve flags. Worker `--models`, `--host` and `--port` options belong to
`vllm-srun`; configure managed instance models and listeners in YAML.

## `vllm-srun serve` {#vllm-srun-serve}

`vllm-srun serve` starts a worker directly with the full runtime protocol. It
is the command used inside the images and needs an installed model runtime
(`make model-runtime-install` for a source checkout). Use it for workers that
a frontend attaches to, or for direct runtime development:

| Option | Default | Meaning |
| --- | --- | --- |
| `MODEL ...` | | Hub repositories, built-in model names or local package directories. Several models share one process. `MODEL@REVISION` pins a revision. |
| `--models FILE` | | A models file instead of `MODEL` arguments. |
| `--revision SHA` | | The 40-character commit to load, for one `MODEL`. |
| `--device` | `auto` | `auto`, or an accelerator with an optional index: `cpu`, `cuda[:N]`, `rocm[:N]`, `xpu[:N]`, `npu[:N]`, `mps`, or one a plugin adds. |
| `--host` | `127.0.0.1` | Address to listen on. |
| `--port` | `8100` | Port to listen on. |
| `--uds PATH` | | Listen on a Unix socket instead of TCP. |
| `--engine` | `auto` | Engine plugin: `auto` (the first that runs the model, native first), `native` (PyTorch) or `onnxruntime` (the `onnx` extra). |
| `--family` | detected | Force a model family plugin. |
| `--served-model-name` | the model name | The model ID the API reports, for one `MODEL`. |
| `--threads` | all cores | CPU threads for the process. Each graph of an ONNX Runtime model runs its own pool of up to this many threads, so set it on a large host. |
| `--memory-budget GIB` | | Refuse to load a model estimated above this size. |
| `--max-queue` | `256` | Queued requests per model before the runtime answers 429. |
| `--max-queued-tokens` | `4194304` | Queued tokens per model before 429. |
| `--max-num-seqs` | `8192` | Most rows one forward holds, across requests: one per question of a decision request, one per input or window otherwise. `1` runs every row alone. The default never binds at the default token budget. See [Engine arguments](../deploy/engine-args.md). |
| `--max-num-batched-tokens` | `65536` | Padded tokens one forward may hold: its rows times its longest row, counted so also for models that pack rows. |
| `--batch-window-ms` | `2.0` | How long the scheduler waits after the first request to collect more, when `--max-num-seqs` is above 1. |
| `--dtype` | `auto` | `auto` (the precision the model was published in), `float32`, `bfloat16` or `float16` (GPUs only). |
| `--enable-shared-context`, `--no-enable-shared-context` | the family's | Compute the shared state of a multi-question decision request once. On by default for Decision 2.0 only. |
| `--max-model-len` | the package's limit | Most input tokens one question of a Decision model may have, at most the package's limit; a longer question fails with `max_length_exceeded`. Decision models only. See [Engine arguments](../deploy/engine-args.md#input-limit). |
| `--max-request-bytes` | `33554432` | Largest request body. |
| `--systemone-schema` | `vllm-sr/5.0` | The System One schema `/v1/systemone` and `/v1/systemone/batch` speak; `vllm-sr/4.0` also accepts the earlier contract's request forms. |
| `--max-bundle-tasks` | `64` | Most tasks in one `/v1/bundle` request. |
| `--load-attempts` | `5` | Loads of a model before it stays `failed`; a damaged package or a failed self-check is not retried. |
| `--load-retry-seconds` | `5` | Wait before the first reload of a model that failed to load, doubling up to 300 s. |
| `--forward-timeout-seconds` | `300` | A forward running longer makes its model `degraded` and `GET /health/live` answer 503 until it returns, so a liveness probe restarts a hung worker. `0` turns this watchdog off. |
| `--shutdown-grace-seconds` | `8` | On SIGTERM, how long the server waits for requests and forwards in flight. It then exits even while a forward runs, with status 4, and logs the forwards it abandoned. |
| `--result-cache-entries` | `16384` | Recent results each model keeps by content; `0` turns the cache off. |
| `--cache-dir` | `HF_HUB_CACHE` | Hugging Face cache directory. |
| `--offline` | off | Use only files already in the cache. |
| `--base-path DIR` | | A local copy of the pinned base model an adapter package needs, instead of a download. Its files are checked against the package's hashes. |
| `--accept-licence ID` | | Accept a restricted licence (repeatable). |
| `--log-level` | `info` | `debug`, `info`, `warning` or `error`. |
| `--autotune-cache DIR` | `$VLLM_SRUN_AUTOTUNE_CACHE` | Keep compiled GPU kernels, and the kernel tuning of models without recorded choices, between runs. Built-in models run their recorded choices on MI300X and MI325X instead of tuning. |

## Other commands

| Command | Prints |
| --- | --- |
| `vllm-srun models` | The built-in models with their pinned revisions |
| `vllm-srun plugins` | The installed families, engines and accelerators |
| `vllm-srun devices` | As JSON, the devices this host offers and the one `--device auto` tries first |

## Models file

A models file lists the models of one process, each with its own options:

```yaml title="models.yaml"
models:
  - model: vllm-sr/Vela-1.0-Encoder-307M-Domain
    name: vela-domain
    device: cpu
  - model: vllm-sr/d3-edge
    name: decision-edge
    device: auto
    enable_shared_context: false
```

Each entry takes `model` (required), `revision`, `name`, `device`, `engine`,
`family`, `memory_budget_gib`, `options` (family options; a Vela 2.0 model
takes `max_scan_tokens`, its [scan budget](./http-api.md#long-inputs)) and the
[engine arguments](../deploy/engine-args.md) `max_num_seqs`,
`max_num_batched_tokens`, `batch_window_ms`, `dtype`,
`enable_shared_context` and `max_model_len`; an entry that leaves one out takes the process's
command-line value. The Router writes this file for the processes it manages.
