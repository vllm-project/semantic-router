# Release note: the built-in model runtime

The first release that includes
[#4512](https://github.com/vllm-project/semantic-router/pull/4512) and
[#4618](https://github.com/vllm-project/semantic-router/issues/4618) runs every
model the router uses in the built-in model runtime, `vllm-srun`, and removes
the in-process native bindings (candle, ONNX Runtime and OpenVINO). Most router
configurations need no change: a feature that never named a back end gets the
same Vela model as before, now served by the runtime. Read the breaking changes
before you upgrade an operator deployment, a training pipeline, stored vectors
or a runtime you run yourself.

## The runtime ships in the router images

The runtime is the `vllm-srun` package and command. Every router image ships
it, and it is not published to PyPI: `vllm-sr` stays the only PyPI package.
Engine mode (`vllm-sr serve ARTIFACT --engine`) runs the persistent
instance frontend and its managed model workers with routing disabled. The
CLI and Docker or Podman are all it needs. To try this development version,
install without automatically starting a stack, then start Engine mode:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --no-launch
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform cpu
```

`--platform rocm` or `--platform cuda` runs the `vllm-sr-rocm` or
`vllm-sr-cuda` image with the GPUs passed through. The images carry the
release's pinned PyTorch build. Numerical equivalence depends on the model,
profile and hardware; validate it before changing precision or execution settings. See the
[model runtime Quickstart](../model-runtime/quickstart).

Builds of `main` between
[#4481](https://github.com/vllm-project/semantic-router/pull/4481) and #4618
named the runtime `vllm-sr-runtime`. If you used one, rename what you set:

| Before | After |
| --- | --- |
| `vllm-sr-runtime` (package and command) | `vllm-srun` |
| `vllm_sr_runtime` (module) | `vllm_srun` |
| `vllm_sr_runtime.families`, `.engines`, `.accelerators`, `.profiles` (plugin entry-point groups) | `vllm_srun.families`, `.engines`, `.accelerators`, `.profiles` |
| `VLLM_SR_RUNTIME_COMMAND`, `_DIR`, `_CACHE_DIR`, `_RESULT_CACHE`, `_AUTOTUNE_CACHE`, `_READY_TIMEOUT` | `VLLM_SRUN_COMMAND`, `_DIR`, `_CACHE_DIR`, `_RESULT_CACHE`, `_AUTOTUNE_CACHE`, `_READY_TIMEOUT` |
| `VLLM_SR_RUNTIME_CPU_PROCESSES` or `VLLM_SRUN_CPU_PROCESSES` | Removed. Set `VLLM_SRUN_CPU_THREADS` to choose threads per managed CPU worker; process counts do not translate directly to thread counts. |
| `VLLM_SR_RUNTIME_PREPARED_DIR` | removed: Vela Omni downloads like every model (below) |
| `vllm_sr_runtime_*` metrics | `vllm_srun_*` |

`VLLM_SR_RUNTIME_CONFIG_PATH`, `VLLM_SR_RUNTIME_STATUS_DIR` and
`VLLM_SR_RUNTIME_CONTAINERS` belong to the CLI and the dashboard, shipped in
v0.4.0, and keep their names.

Managed CPU workers now default to half the router's available CPU budget,
rounded down, with a minimum of one thread and a maximum of 16. The thread count
stays independent of the number of active models; attached runtimes keep their
own settings. See
[CPU threads](model-runtime/deploy.md) before choosing an override.

## Breaking changes

- **Native back ends are refused.** `provider: candle`, `ort` or `openvino`,
  and their execution fields (`precision`, `custom_ops_profile`,
  `compilation_cache_dir`, `variant`, `use_mmbert_32k` and the like), stop the
  router at startup. `vllm-sr config migrate` rewrites them; see
  [Migrate from the native bindings](../model-runtime/migrate).
- **Retired router keys are refused.** The router and the CLI refuse
  `gemma_model_path` and `bert_model_path`, even when empty, and point to
  `vllm-sr config migrate`, which drops them.
- **Retired embedding models: re-embed stored vectors.** EmbeddingGemma,
  MiniLM, `mmbert-embed-32k-2d-matryoshka` and `multi-modal-embed-small` /
  `-large` are replaced by Vela Embedding and Omni. Vectors they stored in the
  semantic cache, memory, vector stores or RAG must be
  [re-embedded](../model-runtime/migrate#re-embed-when-the-embedding-model-changes).
  Older classifier names map to the Vela 1.0 model for the same task, with the
  same labels.
- **The operator CRD no longer has `gemma_model_path`.** kubectl's default
  strict validation refuses a manifest that still sets it, and a stored
  `SemanticRouter` resource loses the field on upgrade. Remove it before you
  upgrade, and re-embed what EmbeddingGemma stored.
- **Retired features.** The NLI model (the hallucination explainer and the
  response cache's `polarity_guard`), OpenVINO, the ONNX Runtime MIGraphX and
  CK flash-attention paths, and RISC-V builds.
- **Router images carry no ONNX Runtime and no Omni bundle**
  ([#4619](https://github.com/vllm-project/semantic-router/issues/4619)). Vela
  1.0 Omni runs on the native engine from its published repository, which the
  runtime downloads (Nano 0.67 GB, Mini 4.3 GB) and verifies like every model.
  An air-gapped cluster that routes images fetches it into the model volume
  first ([Troubleshooting](../model-runtime/troubleshooting#the-runtime-stays-in-loading-or-warming)).
  `vllm-sr config migrate` moves deployments off the retired
  `/opt/router-model-artifacts/vela-1.0-omni-*` paths. ONNX Runtime is the
  optional `onnx` extra of `vllm-srun`, for ONNX packages and prepared Omni
  bundles (`engine: onnxruntime`).
- **Runtime contract 2.0.0.** The router treats an attached runtime that
  doesn't serve contract 2.x, such as the 1.0.0 runtime of #4481, as
  incompatible. Upgrade attached runtimes together with the router.
- **Long inputs cost what their model reads**
  ([#4654](https://github.com/vllm-project/semantic-router/issues/4654)). The
  runtime tokenizes an input only as far as its token budget decides the
  answer, with the same tokens as reading it whole, and rejects one certainly
  over a `reject` budget without tokenizing it. An input read in windows
  (classify `overflow: window`, a Vela 2.0 state part) is read up to a scan
  budget in tokens and fails with `scan_budget_exceeded` past it; Vela 2.0's
  is four inputs on a CPU and 32 on a GPU, and a question with
  `overflow: truncate` reads only a long part's first tokens, one forward on a
  CPU. Contract 2.1.0 adds `scan_budget_exceeded`, the usage flag
  `tokens_lower_bound`, the model limits `max_scan_tokens` and
  `truncate_tokens`, the decisions option `max_tokens` and the question field
  `overflow` ([Long inputs](model-runtime/reference.md#long-inputs)).
- **Routing reads a long request's beginning, safety reads it whole.** Through
  Vela 2.0, routing questions truncate. Safety questions (prompt guard,
  safety, PII, hallucination) read a long request whole up to the scan budget,
  or a question deployment's `input: {overflow: window, max_tokens}`. Compatible questions can share a bundle; input states, task stages and
  window policies can still require separate forwards.
- **Model signals have a deadline.** A model-runtime signal still running at
  `global.model_catalog.signal_timeout_ms` (by default the request's deadline
  less a tenth, or 45 s for a served request) resolves through its availability policy. A decision using
  `rules.on_unknown: fail_request` can reject the request. The deadline does
  not guarantee that an active model forward stops immediately.
- **Jailbreak and PII rules match content their model did not read**: over the
  model's input or cap, truncated, or not scanned by the deadline. The match
  has the type `unscanned` and holds whatever `on_error` says, on requests and
  on responses; `on_unscanned: allow` on the module turns it off.
- **One decisions call per request stage**
  ([#4741](https://github.com/vllm-project/semantic-router/issues/4741)). Every
  question a request stage asks a decision-model deployment goes in one call,
  sent once every signal that asks that deployment has asked, never on a
  timer, so the answers no longer depend on how fast each signal started.
  Questions about other texts, such as the earlier messages a history-aware
  jailbreak or PII rule reads, go in the same call as further states, each
  read as a request of its own. Once the call is sent, every question in it
  waits for it as long as the latest of them, so a decision question's
  `timeout_ms` no longer drops an answer that shares the call. Contract 2.2.0
  adds the decisions field `states`; an attached runtime on 2.1 gets one call
  per text in the same bundle.
- **Training contract `semantic-router.training/v2`.** One
  `runtime/model-runtime@v1` capability replaces `runtime/candle@v1` and
  `runtime/onnxruntime@v1`. BERT and RoBERTa (`architecture/hf-bert@v1`), the
  ONNX format and its exporter are dropped. The API moves to
  `/api/training/v2`, and v1 documents are refused.
