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
Engine mode (`vllm-sr serve MODEL`) runs the runtime in a container from the
router image, so the CLI and Docker or Podman are all it needs:

```bash
pip install vllm-sr
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
```

`--platform amd` or `--platform nvidia` runs the `vllm-sr-rocm` or
`vllm-sr-cuda` image with the GPUs passed through. The images carry the
release's own PyTorch build, so their answers are byte-identical to the
released model packages. See the
[model runtime Quickstart](model-runtime/quickstart.md).

Builds of `main` between
[#4481](https://github.com/vllm-project/semantic-router/pull/4481) and #4618
named the runtime `vllm-sr-runtime`. If you used one, rename what you set:

| Before | After |
| --- | --- |
| `vllm-sr-runtime` (package and command) | `vllm-srun` |
| `vllm_sr_runtime` (module) | `vllm_srun` |
| `vllm_sr_runtime.families`, `.engines`, `.accelerators`, `.profiles` (plugin entry-point groups) | `vllm_srun.families`, `.engines`, `.accelerators`, `.profiles` |
| `VLLM_SR_RUNTIME_COMMAND`, `_DIR`, `_CACHE_DIR`, `_RESULT_CACHE`, `_CPU_PROCESSES`, `_AUTOTUNE_CACHE`, `_READY_TIMEOUT` | `VLLM_SRUN_COMMAND`, `_DIR`, `_CACHE_DIR`, `_RESULT_CACHE`, `_CPU_PROCESSES`, `_AUTOTUNE_CACHE`, `_READY_TIMEOUT` |
| `VLLM_SR_RUNTIME_PREPARED_DIR` | removed: Vela Omni downloads like every model (below) |
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
- **Router images carry no ONNX Runtime and no Omni bundle**
  ([#4619](https://github.com/vllm-project/semantic-router/issues/4619)). Vela
  1.0 Omni runs on the native engine from its published repository, which the
  runtime downloads (Nano 0.67 GB, Mini 4.3 GB) and verifies like every model.
  An air-gapped cluster that routes images fetches it into the model volume
  first ([Troubleshooting](model-runtime/troubleshooting.md#the-runtime-stays-in-loading-or-warming)).
  `vllm-sr config migrate` moves deployments off the retired
  `/opt/router-model-artifacts/vela-1.0-omni-*` paths. ONNX Runtime is the
  optional `onnx` extra of `vllm-srun`, for ONNX packages and prepared Omni
  bundles (`engine: onnxruntime`).
- **Runtime contract 2.0.0.** The router treats an attached runtime that
  doesn't serve contract 2.x, such as the 1.0.0 runtime of #4481, as
  incompatible. Upgrade attached runtimes together with the router.
- **Training contract `semantic-router.training/v2`.** One
  `runtime/model-runtime@v1` capability replaces `runtime/candle@v1` and
  `runtime/onnxruntime@v1`. BERT and RoBERTa (`architecture/hf-bert@v1`), the
  ONNX format and its exporter are dropped. The API moves to
  `/api/training/v2`, and v1 documents are refused.
