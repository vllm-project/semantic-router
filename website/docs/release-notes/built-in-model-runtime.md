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

## Install the runtime with the CLI

The runtime is the `vllm-srun` package and command, released together with
`vllm-sr` and with the same version. Engine mode (`vllm-sr serve MODEL`) no
longer needs a repository checkout: install PyTorch from the
[index that matches your hardware](https://pytorch.org/get-started/locally/),
then the CLI's `runtime` extra.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "vllm-sr[runtime]"
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
```

The `vllm-sr` wheel itself stays CLI-only, and every router image already
ships the runtime. On ROCm, answers byte-identical to the released model
packages are guaranteed in the router images, which carry the release's own
PyTorch build, not in a `pip` install with the official PyTorch wheel. See the
[model runtime Quickstart](model-runtime/quickstart.md).

Builds of `main` between
[#4481](https://github.com/vllm-project/semantic-router/pull/4481) and #4618
named the runtime `vllm-sr-runtime`. If you used one, rename what you set:

| Before | After |
| --- | --- |
| `vllm-sr-runtime` (package and command) | `vllm-srun` |
| `vllm_sr_runtime` (module) | `vllm_srun` |
| `vllm_sr_runtime.families`, `.engines`, `.accelerators`, `.profiles` (plugin entry-point groups) | `vllm_srun.families`, `.engines`, `.accelerators`, `.profiles` |
| `VLLM_SR_RUNTIME_COMMAND`, `_DIR`, `_CACHE_DIR`, `_PREPARED_DIR`, `_RESULT_CACHE`, `_CPU_PROCESSES`, `_AUTOTUNE_CACHE`, `_READY_TIMEOUT` | `VLLM_SRUN_COMMAND`, `_DIR`, `_CACHE_DIR`, `_PREPARED_DIR`, `_RESULT_CACHE`, `_CPU_PROCESSES`, `_AUTOTUNE_CACHE`, `_READY_TIMEOUT` |
| `vllm_sr_runtime_*` metrics | `vllm_srun_*` |

`VLLM_SR_RUNTIME_CONFIG_PATH`, `VLLM_SR_RUNTIME_STATUS_DIR` and
`VLLM_SR_RUNTIME_CONTAINERS` belong to the CLI and the dashboard, shipped in
v0.4.0, and keep their names.

## Breaking changes

- **Native back ends are refused.** `provider: candle`, `ort` or `openvino`,
  and their execution fields (`precision`, `custom_ops_profile`,
  `compilation_cache_dir`, `variant`, `use_mmbert_32k` and the like), stop the
  router at startup. `vllm-sr config migrate` rewrites them; see
  [Migrate from the native bindings](model-runtime/migrate.md).
- **Retired router keys are refused.** The router and the CLI refuse
  `gemma_model_path` and `bert_model_path`, even when empty, and point to
  `vllm-sr config migrate`, which drops them.
- **Retired embedding models: re-embed stored vectors.** EmbeddingGemma,
  MiniLM, `mmbert-embed-32k-2d-matryoshka` and `multi-modal-embed-small` /
  `-large` are replaced by Vela Embedding and Omni. Vectors they stored in the
  semantic cache, memory, vector stores or RAG must be
  [re-embedded](model-runtime/migrate.md#re-embed-when-the-embedding-model-changes).
  Older classifier names map to the Vela 1.0 model for the same task, with the
  same labels.
- **The operator CRD no longer has `gemma_model_path`.** kubectl's default
  strict validation refuses a manifest that still sets it, and a stored
  `SemanticRouter` resource loses the field on upgrade. Remove it before you
  upgrade, and re-embed what EmbeddingGemma stored.
- **Retired features.** The NLI model (the hallucination explainer and the
  response cache's `polarity_guard`), OpenVINO, the ONNX Runtime MIGraphX and
  CK flash-attention paths, and RISC-V builds.
- **Runtime contract 2.0.0.** The router treats an attached runtime that
  doesn't serve contract 2.x, such as the 1.0.0 runtime of #4481, as
  incompatible. Upgrade attached runtimes together with the router.
- **Training contract `semantic-router.training/v2`.** One
  `runtime/model-runtime@v1` capability replaces `runtime/candle@v1` and
  `runtime/onnxruntime@v1`. BERT and RoBERTa (`architecture/hf-bert@v1`), the
  ONNX format and its exporter are dropped. The API moves to
  `/api/training/v2`, and v1 documents are refused.
