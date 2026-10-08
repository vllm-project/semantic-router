---
title: Troubleshooting and FAQ
sidebar_label: Troubleshooting and FAQ
description: Fix the common problems of the model runtime and answers to frequent questions.
---

# Troubleshooting and FAQ

Start with what the runtime says about itself. For a runtime you started:

```bash
curl -s localhost:8100/health
curl -s localhost:8100/v1/models
```

For the router's managed runtimes, look at the router's metrics and logs:

```bash
curl -s localhost:9190/metrics | grep '^vsr_model_runtime'
```

`vsr_model_runtime_ready{deployment="..."} 1` means the deployment answers.
The router log names each managed runtime process and why it stopped.

## Startup waits for the models

The router starts serving only once every model it manages for the
configuration has loaded. Until then `/ready` returns `503` and
`vllm-sr serve` keeps waiting, printing what the router waits for:

```text
Waiting for Router-managed model deployments, 0 of 1 ready: decision-kai (vllm-sr/Decision-2.0-Kai-0.6B) loading
```

`/startup-status` reports `phase: loading_model_deployments` and lists each
deployment in `model_deployments` with its state:

```bash
curl -s localhost:8080/startup-status
```

A first start downloads the models, so it takes longer than the next ones. The
wait ends after `VLLM_SRUN_READY_TIMEOUT` (10 minutes by default) with
`did not become ready within 10m0s`; start again and the download resumes from
the cache, or raise the timeout in the router's environment. A model that fails
to load ends the wait at once with its reason (see below).

## A signal never matches

The model is probably not ready yet, or its answers arrive too late.

1. Check `vsr_model_runtime_ready` for the deployment. While it is `0`, every
   signal that uses it is unknown and does not match.
2. Check `vsr_model_runtime_unknown_answers_total` by `reason`. `timeout`
   means the answer came after the signal's `timeout_ms`; raise it, or use a
   smaller model or a GPU. `unavailable` means the runtime was not ready or
   not reachable. `overloaded` means its queue was full.
3. Send the same text with debugging on and read the
   `x-vsr-matched-*` headers:

   ```bash
   curl -s -D - -o /dev/null localhost:8899/v1/chat/completions \
     -H 'content-type: application/json' -H 'x-vsr-debug: true' \
     -d '{"model": "vllm-sr/auto", "messages": [{"role": "user", "content": "your text"}]}'
   ```

4. Preview the routing of the same text without generating an answer. Its
   `signal_errors` names the signals that were unknown and why, for example
   `decision_timeout`:

   ```bash
   curl -s 'localhost:8080/api/v1/routing/preview?trace=true' \
     -H 'content-type: application/json' \
     -d '{"model": "vllm-sr/auto", "text": "your text"}' \
     | jq '{decision: .decision_result.decision_name, matched: .decision_result.matched_signals, signal_errors}'
   ```

## The runtime stays in `loading` or `warming`

- **First start:** the model is downloading. Large models take minutes. The
  router log and `GET /health` show the stage.
- **No network:** the runtime downloads from the Hugging Face Hub. Offline,
  copy the model into the cache first, or point `artifact` at a local copy.
  For example, fetch Vela Omni Nano at its pinned revision into the cache of
  the runtimes a router image manages (its model volume), then start with
  `HF_HUB_OFFLINE=1`:

  ```bash
  hf download vllm-sr/Vela-1.0-Omni-Nano \
    --revision 2ff2d66385dbdd661a560ec3e8bcb45a0527d92e \
    --cache-dir /app/models/model-runtime
  ```

  Mini's revision is `801bae3ad28df6891408f0e0441c676b30e132e3`. Router images
  no longer carry an Omni bundle, so a router that routes images needs this
  once in an air-gapped cluster.
- **`warming` for a long time on CPU:** the self-check runs a few requests
  through the model. Large decision models on a CPU are slow; use a GPU or
  `vllm-sr/Decision-2.0-Kai-0.6B`.
- **`loading` with "retrying after ..." in the reason:** the model failed to
  load for a reason that can pass, such as a busy GPU, too little free memory
  or an interrupted download. The runtime tries again up to five times, 5
  seconds apart at first and doubling, while the other models in its process
  keep serving (`--load-attempts`, `--load-retry-seconds`). A damaged package or
  a failed self-check is reported as `failed` at once.

## The runtime reports `failed`

`GET /v1/models` shows the reason for each model. When the router runs the
model, its log carries the same reason, for example
`model runtime is not ready: model @domain_classifier failed to load: ...`.
When every model of a runtime process the router runs has failed, the router
restarts that process (1 second at first, up to 60 seconds apart), so a passing
cause such as a busy GPU or a full disk clears on its own. A model that fails
beside models that still serve is retried by the runtime itself, so its
process keeps running. A model the router manages that still fails after three
tries stops the router from starting, and on a configuration reload the new
configuration is rejected while the previous one keeps serving. The common
reasons:

| Reason says | Do this |
| --- | --- |
| a file hash does not match | The download is damaged or the repository changed. Delete that model from the cache and start again. |
| a revision is required | A repository that is not built in needs `revision` with a 40-character commit. |
| access denied, gated or private | Log in with `hf auth login` or set `HF_TOKEN` for the account that has access. |
| does not fit, out of memory | Use a smaller model, a GPU with more memory, or place replicas on separate GPUs. |
| device not available | The named GPU does not exist or the installed PyTorch has no support for it. Use `device: auto`, or install the right PyTorch build. |
| built without LAPACK | The model needs LAPACK on the CPU, and this PyTorch (the ROCm image's) has none. Put the model on a GPU (`device: rocm:0`), or serve CPU models from the CPU image. The runtime does not retry it. |
| no family recognizes the package | The model's architecture is not supported. See [Choose a model](model-runtime/choose-a-model.md#your-own-models). |
| a licence must be accepted | The model's licence restricts use. Pass `--accept-licence <id>` after you checked that you may use it. |

When one model of a process fails, the other models in that process keep
serving.

## A ready model's self-check says `unverified`

`GET /v1/models` shows each model's self-check under `golden`. `matched`
means its answers equal the released answers for that kind of device.
`unverified` means the model serves, but the runtime can't vouch for that:

- **No `reason`, and `golden.reference` is empty:** there are no released
  answers for this device, for example for a model that is not built in or a
  GPU without a validated record (CUDA). The runtime checked only that the
  answers are well formed and the same on every run.
- **`reason` starts with `kernel choices not applied`:** on AMD Instinct
  MI300X and MI325X GPUs, Decision 2.0 and Vela 2.0 run their kernels with
  the configurations their release recorded with FLA 0.5.2, so every process
  gives the same answers. When those can't run here, the model still loads,
  the log says `... runs without its recorded kernel choices: ...`, and its
  answers may differ from the released ones.

| `reason` says | Do this |
| --- | --- |
| `FLA is not installed` | Install `fla-core==0.5.2`, or use the router's ROCm image, which has it. |
| `they were recorded with FLA 0.5.2, not ...` | Install `fla-core==0.5.2`. |
| `failed to import` | Repair the FLA installation; the message names the error. |
| `has no autotuned kernel ...` | Install `fla-core==0.5.2`. |

`--autotune-cache` keeps compiled kernels across restarts; it does not
replace the recorded configurations.

## The router refuses the configuration

| Message mentions | Do this |
| --- | --- |
| `removed model execution fields`, `candle`, `ort`, `openvino`, `precision`, `variant`, `use_mmbert_32k`, `use_nli`, `polarity_guard` | Run `vllm-sr config migrate`. See [Migrate](model-runtime/migrate.md). |
| a binding does not match the model | The model has different labels, heads, dimensions or a smaller input limit than the feature needs. Bind a model made for that feature, or change `input`. |
| `artifact must be a Hub repository ID or an absolute package path` | Use `owner/name` for a Hub model or an absolute path for a local copy. |
| `revision must be a 40-hex commit` | Use the full commit hash, not a branch or tag. |

## A long input is rejected

Task deployments reject input longer than `input.max_tokens` instead of
cutting it silently. Raise `max_tokens` up to the model's limit, or choose
another policy:

```yaml
global:
  model_catalog:
    deployments:
      vela-pii:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-PII
        input:
          max_tokens: 32768
          overflow: window
```

`truncate` keeps the beginning of the text. `window` reads the whole text in
overlapping windows and combines their results, which is what PII and safety
scans should use so nothing is missed.

`window` reads at most `max_tokens`: a longer input fails with
`scan_budget_exceeded`. Through Vela 2.0, routing questions read a long
request's first tokens (8,192 on a CPU for Vela 2.0 0.3B) and safety questions
read it whole up to the model's scan budget (four inputs on a CPU). To let the
safety questions read more, give the deployment a scan budget:

```yaml
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        input:
          max_tokens: 131072
          overflow: window
```

For prompt guard and safety signals, if a native model rejects the length of
one input, the router retries overlapping windows that cover the entire text,
with bounded concurrency. It keeps the highest window risk and requires
complete input coverage from each window. An unfinished scan stays unresolved;
the scan budget and deadline still apply. Direct `/v1/decisions` calls retain
the model's input limits.

A jailbreak or PII rule matches content its model did not read in full: an
input over the model's `max_tokens` under `reject`, over its cap, truncated, or
not scanned within the signals' deadline. The match reports the type
`unscanned` and the reason (`input_limit`, `scan_budget` or `deadline`)
whatever `on_error` says, so padding a prompt cannot carry an attack or
personal data past the check. Set `on_unscanned: allow` on the module to let
such content follow `on_error` instead. The other signals report the reason
and follow their own `on_error`
([Reference](model-runtime/reference.md#long-inputs)).

## A request waits on a slow model

A model-runtime signal that has not answered by the signals' deadline resolves
through its policy, so one slow model does not fail the whole request: a
routing signal through `on_error`, a safety signal as unscanned. The deadline
is the request's less a tenth of the time left, or 45 s for a served request;
set `global.model_catalog.signal_timeout_ms` to shorten it. On a CPU, Vela 2.0
0.3B reads about 1,000 tokens a second on four cores, so a long request may
not finish its safety scan within it.

## The router cannot reach an attached runtime

- The runtime listens on `127.0.0.1` unless you start it with `--host 0.0.0.0`.
- From inside a container, `localhost` is the container itself. Use the
  host's address, a Docker network name or a Kubernetes Service.
- `curl <endpoint>/health` from the router's machine must answer.

## Requests are slower than expected

- For the router's runtimes, compare the router's
  `vsr_model_runtime_server_seconds` by `phase` with
  `vsr_model_runtime_transport_seconds` for the slow deployment (see the
  [reference](./reference.md#metrics)). Most time in `forward` means the model
  itself is slow on this device. Time in `queue` means the model is saturated:
  add a GPU, use a smaller model or another process. A large transport share
  points at the host (CPU contention, a remote endpoint).
- For a runtime you started, look at `vllm_srun_request_duration_seconds` and
  `vllm_srun_queue_duration_seconds` on its `/metrics`, or at the
  `Server-Timing` header of its responses.
- On CPU, models of one process share the CPU threads. Start the runtime with
  `--threads` set to the cores you can give it.
- CPU models slow down sharply when other work holds some of their cores,
  because every thread waits for the slowest one. A process that uses a ROCm
  GPU can keep one CPU core busy even while idle, and so can an LLM server on
  the same host: give the CPU models cores of their own.
- Decision models on a GPU can use `shared_context` or `batching`; see
  [Profiles](./profiles.md).

## FAQ

**Do I need a GPU?** No. Every built-in task model and the smaller decision
models run on a CPU.

**Does the runtime send my data anywhere?** No. It downloads model files from
the Hugging Face Hub and never sends request text out. It does not log request
text, and its metrics contain no request content.

**Can I use a model that is not built in?** Yes, for supported architectures:
give a Hub repository with a `revision`, or an absolute local path. The
runtime computes and reports the identity of models it does not know.

**Is it safe to load models from the Hub?** The runtime never runs code that
ships inside a model repository, and it checks every file of a built-in model
against its pinned hash before loading.

**Can several routers share one runtime?** Yes. Start it once and give every
router the same `endpoint`.

**Can one runtime serve several models?** Yes. Pass several models to
`vllm-sr serve`, or let the router group its deployments into processes.

**What happens to my cached and stored vectors when I change the embedding
model?** They stay apart from new ones and are not reused. See
[Re-embed when the embedding model changes](model-runtime/migrate.md#re-embed-when-the-embedding-model-changes).

**Where are models stored?** In the Hugging Face cache (`HF_HUB_CACHE`) for a
runtime you start, and under the router's model directory for managed
runtimes; see the [reference](./reference.md#router-managed-runtimes).
