---
title: Router configuration reference
sidebar_label: Router configuration
sidebar_position: 3
description: The Router fields that configure model runtime deployments, and the environment variables of Router-managed runtimes.
---

# Router configuration reference

These are the Router fields that describe a `model_runtime` deployment and how
its features read it, and the environment variables that change how the Router
runs its managed workers.

## Deployment and binding fields {#router-configuration}

| Field | Where | Meaning |
| --- | --- | --- |
| `provider: model_runtime` | deployment | The model runs in the model runtime. |
| `artifact`, `revision` | deployment | Hub repository and commit, or an absolute local path. |
| `device` | deployment | See [Run it with the router](../deploy/connect-the-router.md#describe-a-deployment). |
| `max_num_seqs`, `max_num_batched_tokens`, `batch_window_ms`, `dtype`, `enable_shared_context`, `max_model_len` | deployment | The deployment's [engine arguments](../deploy/engine-args.md); an unset one keeps the runtime's default. `max_model_len` applies to a managed worker without replicas. |
| `input.max_tokens`, `input.overflow` | deployment | Input limit of task models. A deployment that answers questions takes only `overflow: window` with `max_tokens`: the scan budget its questions read a long part whole to (see [Long inputs](./http-api.md#long-inputs)). That includes the default decision model, a deployment published for direct System One inference, and the deployment of a local System One alias, whether or not a listener publishes the alias. A routed System One entrypoint is not one deployment. A bound deployment that is none of these keeps a task-model limit. |
| `signal_timeout_ms` | `global.model_catalog` | Deadline of a request's model-runtime signals. By default the request's deadline less a tenth of the time left (at least a second), or 45 s for a served request, which has none the router sees. A signal still running resolves through its policy: a jailbreak or PII rule as no answer, which matches unless the rule sets `on_unknown: no_match`, and any other signal as a failed signal, through `on_error` or `rules.on_unknown`. |
| `on_unscanned` | `recipes[].signals.systemone[]`, `jailbreak[]`, `pii[]` | With `coverage: full`, what an answer that could not read the whole input means: `match`, `no_match`, `unknown` (default) or `best_effort`. `jailbreak` and `pii` rules read a long input in windows and default to `best_effort`, which decides on the windows read. |
| `routing_view`, `scan_window` | deployment | `routing_view: {max_tokens, head_share}` bounds what a routing question reads of a long input to its head and tail (default 8,192 tokens, a quarter head); `scan_window: {tokens, overlap, max_parallel, max_tokens, reread_floor}` sizes the windows a guard question reads it in (8,192, overlapping by 512), each a call of its own, `max_parallel` (4) at a time, growing up to `max_tokens` between the head and the tail when 256 do not cover the input or a scan that fell behind reads the rest in time that way; a larger window that comes close to a match (`reread_floor`, 0.9) is read again at `tokens`. Both are sized in bytes, four a token, and never above the model's validated length less 1,024 counted a byte a token; see [Long inputs](../../system-one/deploy/performance.md#long-inputs). |
| `replicas` | deployment | Placements of independent workers for one logical model. Each has `device` or `endpoint`/`model`; omit for one worker. Do not combine with deployment-level placement. |
| `outlier_detection` | deployment | `consecutive_timeouts` (default 3; 0 turns it off), `base_ejection_ms` (30000), `max_ejection_ms` (ten times the base) and `max_ejection_percent` (50). A replica whose exchanges run out of their deadline that many times in a row leaves dispatch for the base, twice as long each time in a row; see [Eject a replica that keeps timing out](../deploy/placement-and-scaling.md#eject-a-replica-that-keeps-timing-out). |
| `enabled` | `global.router` | Default `true`. Disable routing signals while retaining the frontend, native API and decision model. |
| `endpoint`, `model` | deployment | Attach to a runtime you run; `model` is the served model's ID. The [connection block](../../installation/model-catalog.md#connection-block) adds `auth`, `headers`, `timeout_ms` and `tls`. |
| `bindings.<type>`, `recipes[].model_bindings.<type>` | `global.model_catalog`, recipe | The deployment a signal type uses: a name, or `{deployment, head}` to select a head. |
| `deployment` | rule | The deployment a `decision`, `classifiers`, `safety` (and `hazard`) or `complexity` rule asks. |

The binding keys are the signal types `domain`, `jailbreak`, `pii`,
`fact_check`, `user_feedback`, `modality`, `hallucination`, `preference` and
`reask`. Embeddings name their deployment in
`global.model_catalog.embeddings.default`. [Choose a model for a signal](../../installation/choose-a-model-for-a-signal.md)
shows how each resolves; the task guides list the answer each signal reads.

## Router-managed runtimes

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_SRUN_COMMAND` | `vllm-srun` | Command the router runs for a managed process. |
| `VLLM_SRUN_DIR` | a private temporary directory | Where the router puts the processes' Unix sockets and models files. |
| `VLLM_SRUN_CACHE_DIR` | `/app/models/model-runtime` in router images | Hugging Face cache of managed runtimes. |
| `VLLM_SRUN_CPU_THREADS` | Half the CPU budget, rounded down; minimum 1, maximum 16 | Threads per managed CPU worker. A positive integer overrides the default, capped by the router's full CPU budget; invalid or nonpositive values use the default. It does not control worker count or attached runtimes. |
| `VLLM_SRUN_READY_TIMEOUT` | `10m` | How long the router waits for a deployment to become ready when it starts or reloads, as a duration such as `30m`. A first start may download and verify large models. |
| `VLLM_SRUN_RESULT_CACHE` | `4096` | Recent classify and decision results for a single-worker deployment; `0` disables it. Multi-worker pools bypass this frontend cache; worker caches remain independent. |
| `VLLM_SRUN_AUTOTUNE_CACHE` | | The `--autotune-cache` directory of a runtime. |
| `HF_TOKEN` | | Token for gated or private repositories. |

The CPU budget is the router's `GOMAXPROCS`, normally derived from CPU affinity
and the container's CPU quota. `vllm-sr serve` forwards `VLLM_SRUN_CPU_THREADS`
to the router. Worker thread counts stay independent of the active routing signals;
see [CPU threads](../deploy/placement-and-scaling.md#cpu-threads) for tuning guidance.

The router talks to managed runtimes only over Unix sockets in a directory only
its own user can open (mode 0700). It restarts a process that exits, waiting 1
second at first and up to 60 seconds after repeated failures, and stops every
process (SIGTERM, then SIGKILL after 10 seconds, logged as
`runtime_process_killed`) when it exits. It also restarts a process in
which every model failed to load, the same way, so a model that failed for a
passing reason comes back without a configuration reload. A process that still
serves any model keeps running.
