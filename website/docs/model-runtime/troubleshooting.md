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
     -d '{"model": "auto", "messages": [{"role": "user", "content": "your text"}]}'
   ```

## The runtime stays in `loading` or `warming`

- **First start:** the model is downloading. Large models take minutes. The
  router log and `GET /health` show the stage.
- **No network:** the runtime downloads from the Hugging Face Hub. Offline,
  copy the model into the cache first, or point `artifact` at a local copy.
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
process keeps running. A task model that still fails after three tries stops
the router from starting. The common reasons:

| Reason says | Do this |
| --- | --- |
| a file hash does not match | The download is damaged or the repository changed. Delete that model from the cache and start again. |
| a revision is required | A repository that is not built in needs `revision` with a 40-character commit. |
| access denied, gated or private | Log in with `hf auth login` or set `HF_TOKEN` for the account that has access. Vela 2.0 is a private preview. |
| does not fit, out of memory | Use a smaller model, a GPU with more memory, or give the model its own `process`. |
| device not available | The named GPU does not exist or the installed PyTorch has no support for it. Use `device: auto`, or install the right PyTorch build. |
| no family recognizes the package | The model's architecture is not supported. See [Choose a model](model-runtime/choose-a-model.md#your-own-models). |
| a licence must be accepted | The model's licence restricts use. Pass `--accept-licence <id>` after you checked that you may use it. |

When one model of a process fails, the other models in that process keep
serving.

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

## The router cannot reach an attached runtime

- The runtime listens on `127.0.0.1` unless you start it with `--host 0.0.0.0`.
- From inside a container, `localhost` is the container itself. Use the
  host's address, a Docker network name or a Kubernetes Service.
- `curl <endpoint>/health` from the router's machine must answer.

## Requests are slower than expected

- Look at `vllm_sr_runtime_request_duration_seconds` and
  `vllm_sr_runtime_queue_duration_seconds` on the runtime's `/metrics`. Time
  spent queueing means the model is saturated: add a GPU, use a smaller model
  or another process.
- On CPU, models of one process share the CPU threads. Start the runtime with
  `--threads` set to the cores you can give it.
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
