# vLLM plugin-registry prototype: Decision 2.0 on stock vLLM (design + results, 2026-10-01)

Serving track (coordinator notes 2026-09-30 19:40, 19:50, 20:00, ~23:40). Branch `xunzhuo/decision-2-vllm-plugin`;
package `src/vllm-sr-plugins/`; sessions `src/training/decision2/v2/serving/`; state
[`plugin-state.md`](plugin-state.md). All measurements: node A GPU1 (one MI325X, shared lease), image
`decision20-train-fast:host2`, vLLM `0.29.1rc1.dev187+gaf1c01499.rocm723`, PyTorch `2.12.0+git6bbd260`, Transformers
5.17.0. Nothing was uploaded; per-prompt answers stay on the node.

RESULTS_PLACEHOLDER

## Design

### One package, registered through vLLM's plugin entry points

`pip install src/vllm-sr-plugins` into the environment that has vLLM, then `VLLM_PLUGINS=vllm_sr_decision2,vllm_sr_decisions`.
No vLLM file is modified or forked.

| Entry point | Name | What it registers |
| --- | --- | --- |
| `vllm.general_plugins` | `vllm_sr_decision2` | `Decision2Qwen3_5ForScoring` in `ModelRegistry`, plus vLLM's own Qwen3.5 text-config hook for it (FP32 gated-delta state as the checkpoint's `mamba_ssm_dtype` says; 1-D positions). Opt-in `VLLM_SR_GDN_BF16_INPUTS=1`: a `CustomOp.register_oot` override of `chunk_gated_delta_rule` for float32 studies. |
| `vllm.endpoint_plugins` | `vllm_sr_decisions` | `POST /v1/decisions` and the alias `POST /v1/system_one` on vLLM's OpenAI-compatible server. |

- **Model class.** `Decision2Qwen3_5ForScoring` is vLLM's `Qwen3_5Model` (text-only hybrid: gated-delta + full
  attention, vLLM's own kernels) plus `CandidateHead` in FP32 (the training head's parameter names and arithmetic),
  with no LM head (`is_pooling_model = True`). It is served straight from a package directory: `backbone/*.safetensors`
  through vLLM's default loader and `decision_head.safetensors` as a secondary weight source. The engine loads
  753,446,208 parameters, the release receipt's count.
- **Candidate pooler.** Serves the built-in `token_classify` pooling task: the one task with per-request output
  lengths that both model runners accept (Model Runner V2, the default, rejects the `plugin` task). Each request carries
  its option-endpoint and query positions in `PoolingParams.extra_kwargs["decision2"]`; the pooler copies those hidden
  rows out of every prefill step the request takes part in (chunked prefill works), applies the FP32 head once the
  prompt is complete, and returns one raw logit per candidate. A request without valid positions gets a NaN, never an
  engine failure.
- **Endpoint.** At startup it opens the served package exactly like `Decision2.from_pretrained`: every file against
  `MODEL_MANIFEST.json`, the model identity against the scored identity, then the calibration and Score offsets bound
  to it. Prompt rendering, tokenization, over-budget handling, calibration, Score offsets and answer formatting run
  the **package's own vendored runtime** (imported under a private module name), so the served contract is the
  package's. Each valid question becomes one engine request; a request's questions run concurrently and vLLM batches
  them with all other traffic (continuous batching instead of per-item padding).
- **Pinned build.** vLLM `0.29.1rc1.dev187+gaf1c01499.rocm723` in `decision20-train-fast:host2`. Pooler, loader,
  custom-op and endpoint-plugin interfaces move between vLLM releases, so the plugin warns on any other build; a new
  build needs the unit tests (27, run inside the image) and the parity panels first.

### Bugs found on the way (all fixed before the measured runs)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Engine start fails | Model Runner V2 rejects the `plugin` pooling task | serve `token_classify` (`711330eac`) |
| Engine start fails | DEV2.0-0.8B's vendored runtime predates Score offsets | load offsets only when the manifest binds them (`678ac2060`) |
| First request HTTP 400 (`query.raw` missing) | `from __future__ import annotations` hid FastAPI's `Request` type | real annotations; a FastAPI route test through the service now runs on CPU (`91c9d9e8d`) |
| `--dtype float32` engine start fails | vLLM's chunked gated-delta kernel asserts on float32 | opt-in custom-op override (`ecb46f764`); see numerics |
