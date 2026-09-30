# vLLM plugin-registry prototype: Decision 2.0 on stock vLLM (design + results, 2026-10-01)

Serving track (coordinator notes 2026-09-30 19:40, 19:50, 20:00, ~23:40). Branch `xunzhuo/decision-2-vllm-plugin`;
package `src/vllm-sr-plugins/`; sessions `src/training/decision2/v2/serving/`; state
[`plugin-state.md`](plugin-state.md). All measurements: node A GPU1 (one MI325X, shared lease), image
`decision20-train-fast:host2`, vLLM `0.29.1rc1.dev187+gaf1c01499.rocm723`, PyTorch `2.12.0+git6bbd260`, Transformers
5.17.0. Nothing was uploaded; per-prompt answers stay on the node.

## Summary

- **Plugin v0 works on stock vLLM.** One pip-installable package (`vllm-sr-plugins`) registers
  `Decision2Qwen3_5ForScoring`, a candidate pooler and `POST /v1/decisions` (alias `/v1/system_one`, identical
  answers). DEV2.0-0.8B answered all 11,053 scored answers of four stored panels with 0 errors, 0 missing answers and 0
  input mismatches.
- **Parity.** A `bfloat16` engine changes 63 of 11,053 answers (0.57%; max probability drift .075, max Score
  expected-value drift .124). The shipped Transformers runtime with BF16 parameters changes 64 (0.58%) with the same
  drift distribution, so the gap is BF16 residual-stream precision, not vLLM's kernels. `float32` needs two workarounds
  (below) and changes 42 (0.38%; max drift .050). Every changed answer was a near-tie when scored (stored margin
  ≤ .026, median ≈ .002).
- **Latency and throughput (same GPU).** The plugin over HTTP: single-request p50 15.2 ms vs 22.7 ms for the shipped
  runtime in-process (1.5× faster), and 622 vs 44 items/s at concurrency 128 (14×, from continuous batching; the runtime
  has no cross-request batching).
- **BF16-resident quick win (shipped runtime).** Bit-identical answers (400 / 400 items; 0 changes on all four panels),
  p50 22.7 → 21.4 ms, peak memory 2.91 → 2.01 GiB at 0.8B.
- **Vela-1.0-Encoder-307M-Domain on vLLM's native ModernBERT classify** (no plugin): 0 argmax changes vs Transformers
  on 265 inputs up to 32,714 tokens (max probability drift .004 in float32, .070 in bfloat16); p50 ≈ 3 ms; about
  1,900–2,000 requests/s.
- **One runner type per vLLM instance.** Label-token deciders need the generate runner (Tier 1a); our scoring-head
  models need the pooling runner plus this plugin (Tier 1b). Demo: a stock Qwen3.5-0.8B label-token read with
  `allowed_token_ids` and processed logprobs matches Transformers (0 argmax changes on 16 reads, max drift .023).
  On a shared 2,760-token state, 86% of prompt tokens hit the prefix cache whether 8 questions are sent one by one,
  staggered or all at once (all at once: 8 answers in 51 ms; one by one: 162 ms). Tier 1b reads no cache yet (first
  next step).
- **Serving shape.** One model and one runner per engine. Two engines sharing one GPU (Decision + Vela) barely
  interfere at 1–8 concurrent requests each (throughput within 7%, p50 within 0.2 ms); at 32 each, Decision throughput
  drops 10% and Vela's 26%.
- **GPU:** 0.83 of the 3 GPU-h cap, node A GPU1 (details at the end).

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
  build needs the unit tests (28, all run inside the image) and the parity panels first.

### Bugs found on the way (all fixed before the measured runs)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Engine start fails | Model Runner V2 rejects the `plugin` pooling task | serve `token_classify` (`711330eac`) |
| Engine start fails | DEV2.0-0.8B's vendored runtime predates Score offsets | load offsets only when the manifest binds them (`678ac2060`) |
| First request HTTP 400 (`query.raw` missing) | `from __future__ import annotations` hid FastAPI's `Request` type | real annotations; a FastAPI route test through the service now runs on CPU (`91c9d9e8d`) |
| `--dtype float32` engine start fails | vLLM's chunked gated-delta kernel asserts on float32 | opt-in custom-op override (`ecb46f764`); see numerics |
| `--dtype float32` engine start fails again | default ROCm attention backend: prefix-prefill kernel needs 128 KiB LDS at float32 / head 256 | `--attention-backend TRITON_ATTN` (session flag) |
| A ModernBERT server aborts at start | vLLM loads an endpoint plugin whose required task the engine supports; `init_state` raised on a foreign architecture | keep foreign servers running, route answers 503 (`14fe9ac40`) |

## Results

All runs are on node A GPU1, with exact mirrors of the pushed commits named below. Parity uses the release-parity
definitions (`v2.release.examples`): an answer change is a different category (Choice key, Noul side of 0.5, Score
argmax level) or a different set of numbers; drift is the largest absolute difference of any reported number. The
stored predictions are the released DEV2.0-0.8B revision's (BF16 storage, `bede7938`; release parity-post: 0 changes,
drift ≤ 9e-16). Each panel request carries one item, like the release parity; its questions run concurrently.

### DEV2.0-0.8B parity vs stored scored predictions

Answer changes (max drift). "Transformers BF16 params" is a control: the shipped runtime with every backbone parameter
in BF16, so its residual stream and norms are BF16 like a bfloat16 vLLM engine.

| Panel | Answers | vLLM bfloat16 | vLLM float32 † | Transformers BF16 params | Shipped runtime, BF16-resident |
| --- | --- | --- | --- | --- | --- |
| typed FINAL | 2,000 | 20 (.061) | 14 (.031) | 17 (.044) | 0 (9e-16) |
| CSS15 | 6,547 | 33 (.031) | 22 (.027) | 36 (.030) | 0 (3e-16) |
| public 231 | 231 | 3 (.022) | 1 (.019) | 1 (.030) | 0 (4e-16) |
| mlx-diag (all) | 2,275 | 7 (.124) | 5 (.050) | 10 (.053) | 0 (4e-16) |
| **Total** | **11,053** | **63 (0.57%)** | **42 (0.38%)** | **64 (0.58%)** | **0** |
| public 231, concurrency 32 | 231 | 3 (.024) | 1 (.015) | — | — |

† `--dtype float32` with two workarounds, both measurement-only: the opt-in `VLLM_SR_GDN_BF16_INPUTS=1` override
(vLLM's chunked gated-delta kernel asserts on float32; the override feeds it the bfloat16 inputs that BF16 autocast
gives it in the shipped runtime) and `--attention-backend TRITON_ATTN` (the default ROCm attention backend's
prefix-prefill kernel needs 128 KiB of LDS at float32 / head size 256; MI325X has 64 KiB).

- **Drift distribution** (probabilities, all slots): median .0009–.0023 (bf16), .0007–.0018 (fp32); p99 .012–.019
  (bf16), .009–.019 (fp32). Changes by type, bf16: Choice 48, Score 11, Noul 4; fp32: Choice 33, Score 7, Noul 2.
- **Changed answers are near-ties** (`v2.serving.flips`): the stored margin (Noul |p − .5|, Choice / Score top-1 minus
  top-2) of every changed answer is ≤ .026 (per-panel medians .0009–.0039, except public 231 fp32: one change at
  .015); 5.9% of typed FINAL and 6.0% of CSS15 answers have a margin that small. bf16 and fp32 flip partly the same
  answers (typed FINAL 6, CSS15 11, mlx-diag 4, public 1 in both).
- **Batch composition matters little**: public 231 at concurrency 32 changes the same number of answers as at
  concurrency 1 (bf16 3, fp32 1).
- **Reading.** Serving DEV2.0 on bfloat16 vLLM costs about 0.6% answer changes on near-ties, the same as BF16 in the
  reference stack itself. Exact parity would need FP32 residuals with BF16 GEMMs (the autocast recipe) inside vLLM, for
  example a quantization-method plugin (see next steps). float32 cuts answer changes by a third and the largest drifts
  by up to half, but doubles latency.

### Latency and throughput (DEV2.0-0.8B, same GPU)

400 typed FINAL items (one question each, 346 input tokens on average). Plugin numbers include a local HTTP round trip
and JSON; runtime numbers are in-process `Decision2.system_one` calls with `torch.cuda.synchronize()` (no HTTP).

| Path | Precision | Single p50 / p95 (ms) | Items/s at concurrency 1 / 8 / 32 / 128 | Memory |
| --- | --- | --- | --- | --- |
| vLLM + plugin, `/v1/decisions` | bfloat16 | **15.2** / 15.6 | 66 / 208 / 449 / **622** | 1.4 GiB weights + KV pool (`--gpu-memory-utilization 0.2`) |
| vLLM + plugin, `/v1/decisions` | float32 † | 30.0 / 30.8 | 34 / 133 / 211 / 239 | 2.8 GiB weights + KV pool |
| Shipped runtime | FP32 master + BF16 autocast | 22.7 / 23.1 | 44 (no cross-request batching) | peak 2.91 GiB |
| Shipped runtime, BF16-resident Linear | same answers | 21.4 / 21.7 | 47 | peak 2.01 GiB |
| Transformers, BF16 params (control) | bfloat16 | 22.8 / 23.3 | 44 | peak 1.53 GiB |

- At concurrency 128 the bfloat16 plugin serves 215k input tokens/s; p50 latency grows to 203 ms (p95 250 ms).
- Engine start (weights, compile, CUDA-graph capture): 130–136 s cold, without a persisted compile cache.
- **BF16-resident Linear** converts all 186 backbone Linear weights (0 kept FP32): the per-call FP32 → BF16 re-cast
  goes away, answers stay bit-identical, and peak memory drops by 0.9 GiB. At 0.8B it saves 1.3 ms per call
  (−6%); the coordinator's estimate for 9B / 27B is larger (~10 / ~30 ms).

### Vela 1.0 classifier on vLLM's native ModernBERT classify

`llm-semantic-router/Vela-1.0-Encoder-307M-Domain` (the router's `domain_classifier`, 14 classes, CLS pooling,
32,768-token context) with stock vLLM (`--runner pooling`; no plugin in the engine), against Transformers FP32
(`sdpa`) on the router's 261 e2e domain cases plus four long concatenations (1,997 / 8,175 / 16,318 / 32,714 tokens).

| Engine dtype | Argmax changes (≤ 512 / ≤ 8,192 / > 8,192 tokens) | Max prob drift | Text input (server tokenizes) | Single p50 (ms) | Requests/s at concurrency 1 / 8 / 32 / 128 |
| --- | --- | --- | --- | --- | --- |
| float32 | 0 / 0 / 0 (n 259 / 4 / 2) | .0040 / .0010 / .0033 | 0 changes, .0028 | 2.9 | 331 / 1,097 / 1,933 / 1,982 |
| bfloat16 | 0 / 0 / 0 | .057 / .010 / .031 | 0 changes, .070 | 3.1 | 328 / 672 / 1,878 / 1,677 |

vLLM's `/tokenize` equals the Transformers tokenizer on 261 / 261 texts. Vela needs no plugin class: the native
architecture, CLS pooling and classifier head load from the checkpoint as published. For a 307M encoder on short inputs,
bfloat16 buys no speed and costs drift, so float32 is the better default.

### Tier 1a demonstration: label-token reads on the generate runner

Stock `Qwen/Qwen3.5-0.8B` (revision `2fc06364`), `--runner generate --language-model-only`, no plugin. 16 public
synthetic reads (4 support tickets × 3 Noul yes / no questions and 1 Choice A–D), rendered with the model's chat
template (thinking off); each request is `max_tokens=1`, `temperature=0`, `allowed_token_ids` = the label tokens,
`logprobs` = the number of labels, with the engine at `--logprobs-mode processed_logprobs`.

- The returned label probabilities sum to 1 (0.99999995–1.00000007): renormalized over the allowed labels.
- The argmax label is the generated token on 16 / 16 reads.
- Against Transformers FP32 (`Qwen3_5ForConditionalGeneration`, label logits at the answer position, softmax over the
  labels): 0 argmax changes, max probability drift .023.
- Latency: p50 15.8 ms single, 373 requests/s at concurrency 16 (p50 42 ms).
- Prefix caching is on by default for the hybrid in `align` mode (attention block 544 tokens). Eight yes / no questions
  over one 2,760-token state, each order on a fresh state and an idle engine:

| Send order | Prompt tokens from cache | 8 answers (wall) | First question | Follow-ups p50 |
| --- | --- | --- | --- | --- |
| one by one | 19,040 / 22,080 (86%) | 162 ms | 43.1 ms | 16.9 ms |
| first alone, then 7 at once | 19,040 / 22,080 (86%) | 79 ms | 38.3 ms | 38.4 ms |
| all 8 at once | 19,040 / 22,080 (86%) | 51 ms | 49.7 ms | 47.8 ms |

  Every follow-up reuses 5 cached blocks (2,720 tokens), even when all eight are sent together: the first prefill
  occupies the engine step and the others, arriving milliseconds later, are scheduled after it and read its state. So
  a System One call can send all its questions at once. Arrivals that land in the same engine step under load were not
  measured.

### Co-located engines on one GPU

A DEV2.0-0.8B bfloat16 engine (`--gpu-memory-utilization 0.2`, plugin) and a Vela Domain bfloat16 engine (0.1, stock)
on node A GPU1. Decision is driven with the 400 typed FINAL items, Vela with the 261 domain cases (× 4 in the joint
phase), both at the same worker count; alone, then both at once.

| Workers per engine | Decision alone → joint: requests/s; p50 / p95 ms | Vela alone → joint: requests/s; p50 / p95 ms |
| --- | --- | --- |
| 1 | 65 → 64; 15.3 / 15.6 → 15.4 / 15.9 | 338 → 326; 2.9 / 3.2 → 3.0 / 3.3 |
| 8 | 204 → 203; 39.1 / 40.0 → 39.1 / 41.4 | 1,179 → 1,096; 6.9 / 9.5 → 7.1 / 11.1 |
| 32 | 470 → 422; 61.3 / 86.7 → 67.7 / 133.4 | 2,059 → 1,519; 15.2 / 19.1 → 17.2 / 26.0 |
| 128 | 617 → 588; 194 / 318 → 204 / 241 | 1,801 → 2,242 *; 35.1 / 99.9 → 43.5 / 71.6 |

\* The joint Vela phase runs four times as many requests (1,044 vs 261), so its rates at high concurrency are not
comparable one to one. Router-scale traffic (a few to a few tens of concurrent decisions) co-locates well; heavy load on
both engines needs separate GPUs or a joint scheduler.

## Serving shape: one model per engine, one runner per engine

A vLLM engine (`vllm serve`, `LLM`, `AsyncLLM`) serves **one base model with one runner type**: `--runner generate`
or `--runner pooling`, fixed at start. Endpoint plugins attach to that engine's server, and a plugin loaded next to
another model now stays out of the way (the route answers 503) instead of aborting it.

- **Label-token deciders need the generate runner (Tier 1a).** The majority of third-party decision models read
  LM-head logits of label tokens at an answer position. Stock vLLM serves them with `allowed_token_ids` + `logprobs`
  (`--logprobs-mode processed_logprobs` returns the distribution renormalized over the labels, as shown above). They
  also get prefix caching on a shared state, including for Qwen3.5 hybrids ("align" mode).
- **Scoring-head models need the pooling runner plus this plugin (Tier 1b):** Decision 2.0 (option-endpoint head),
  AutoJev-style heads, Vela / ModernBERT sequence classifiers (native pooling, no plugin).
- **So a mixed deployment runs at least one engine per runner type, and one per model.** The options, with what each
  costs:

| Option | How | Fits | Costs / limits |
| --- | --- | --- | --- |
| Separate engines on one GPU | one `vllm serve` per model, `--gpu-memory-utilization` fractions (e.g. Decision 0.2 + Vela 0.1) | any mix of runners and models; what the co-location run measures | each engine has its own process, CUDA context, compile cache and CUDA graphs (≈ 2 min cold start at 0.8B); no joint scheduler, so the engines contend for the GPU |
| Multi-LoRA on one backbone | `--enable-lora`; per-request `lora_request` (LoRA works for pooling models) | several adapters of one base, e.g. the 27B adapter packages | Decision packages also carry a per-package head; the pooler needs per-adapter heads (vLLM #53555-style) — plugin work |
| Several heads on one backbone | the pooler applies N heads to the same hidden states | multi-task heads trained on one frozen or shared backbone | only for models trained that way; one request, several outputs |
| Replicas across GPUs | `--data-parallel-size N` or several engines behind a router | throughput | memory × N |
| In-process encoders (Tier 0) | ORT / candle next to the router | small encoders, lowest latency, CPU / other hardware | outside vLLM; ModernBERT on vLLM already runs at ≈ 3 ms |

Tier 1b's prefix caching: this vLLM build enables prefix caching for the hybrid pooling engine ("align" mode, 544-token
blocks), but `token_classify` requests default to `skip_reading_prefix_cache=True`, so the measured hit rate is 0% and
every question re-reads its state. Decision 2.0 renders the state first (`Context: <state> … Task type … Question …
Options … Decision:`), so all candidate and query rows follow the shared state and a request could read the cached state
blocks. It needs `skip_reading_prefix_cache=False` plus a guard for a prompt whose cached prefix reaches a candidate
row (an identical or near-identical earlier prompt): today the pooler answers NaN there → `invalid_model_output`; a
retry of that question with cache reads off would be exact. The endpoint's all-questions-at-once submission already
suits the cache (see the Tier 1a send-order table). That is the first Tier 1b next step.

## Limitations of v0

- Served profile: `qwen-full` with the shared candidate head (DEV2.0-0.8B / 2B / 4B / 9B class). Not yet: base-bound
  adapter packages (27B), `bf16z`-compressed storage, Qwen3 backbones (the DEV2.0-0.6B class), Kai-native encoders.
- bfloat16 serving changes ~0.6% of answers (near-ties) against the FP32-master scored predictions; the release gates
  were measured on the shipped runtime, so a vLLM-served card would need its own parity statement or re-scoring.
- No prefix-cache reads in Tier 1b yet (above). A request without positions (raw `/pooling`) gets NaN.
- Tested on one vLLM build and one GPU type (ROCm, MI325X). Pooler, loader, custom-op and endpoint-plugin interfaces are
  not stable across vLLM releases: compatibility CI per pinned build is required.
- `float32` needs the measurement-only workarounds above; it is a numerics tool, not a serving mode.

## Next steps

1. **Tier 1b prefix caching**: `skip_reading_prefix_cache=False` when every needed row follows the state, plus a guard
   (endpoint-level memo of identical encoded prompts, or an upstream per-request cap on cache reads); measure
   multi-question requests over one long state.
2. **Autocast-exact numerics in vLLM**: FP32 residual stream and norms with BF16 GEMMs, via an out-of-tree
   quantization-method plugin (BF16 weights, BF16 matmul, FP32 activations) on a float32 engine; target 0 changes.
3. **Qwen3 0.6B class** (DEV2.0-0.6B): a `Decision2Qwen3ForScoring` on vLLM's `Qwen3Model` (dense, no GDN) with the
   same pooler and endpoint.
4. **27B adapters**: serve the base-bound LoRA packages with vLLM LoRA on the pooling runner, one head per adapter.
5. **Kai / Lex three-path encoder port**: the encoder-family Decision models as a pooling model class with the same
   endpoint contract.
6. **Router integration**: an `http_decision` adapter and a `decision_answers.v1` contract in the router, calling
   `/v1/decisions` (one batched call per request, per-signal deadlines, fail-open); the engine supervised as a local
   sidecar over UDS (Tier 1).
7. **Compatibility CI**: the 28 plugin unit tests plus a public-231 parity smoke per pinned vLLM build.

### Upstream-contribution list (vLLM)

| Item | Why |
| --- | --- |
| float32 path in the chunked gated-delta op (or a documented cast) | `--dtype float32` fails for every Qwen3.5 / Qwen3-Next hybrid |
| ROCm prefix-prefill attention: tile sizes that fit 64 KiB LDS at float32 / head 256 (or automatic fallback) | float32 engines fail on MI300-class GPUs with the default backend |
| A generic "positions" pooling task, or Model Runner V2 accepting the `plugin` task | today a plugin pooler must borrow `token_classify` |
| `PoolingParams.extra_kwargs` as a documented per-request pooler input | the plugin's positions channel |
| Per-request cap on prefix-cache reads for pooling | safe shared-state reuse for positional heads |
| Per-adapter heads for LoRA pooling models | multi-adapter Decision serving |
| Endpoint-plugin docs: postponed annotations break FastAPI dependency injection; skip foreign architectures | two plugin-author pitfalls found here |

## Reproduction

```bash
# local: exact mirrors of one pushed commit (both subtrees)
bash src/training/decision2/v2/common/mirror_to_node.sh --path src/training/decision2 node-a <sha>
bash src/training/decision2/v2/common/mirror_to_node.sh --path src/vllm-sr-plugins node-a <sha>
# node A: one session container on one GPU under the shared lease (see run.sh for the flags)
bash <decision2 mirror>/src/training/decision2/v2/serving/run.sh --src <decision2 mirror> --plugin <plugin mirror> \
  --gpu 1 --run <name> --mount <package> --mount <panels> --mount <predictions> -- \
  plugin --plugin-src <plugin mirror>/src/vllm-sr-plugins --package <package> --dtype bfloat16 \
  --panel typed-final:<prompts>:<predictions>:1600 --bench <typed-final prompts>:400
```

| Run (node A `/data/dev2/runs/serving/`) | Mirror | What |
| --- | --- | --- |
| `p08-bf16-0930T1611` | `9c0abb957` | plugin bfloat16: 4 panels + concurrency-32 panel + bench |
| `rt08-0930T1618` | `9c0abb957` | shipped runtime FP32 master + BF16-resident (frozen autotune cache) |
| `rt08-bf16full-0930T1632` | `ecb46f764` | Transformers BF16-parameter control |
| `p08-fp32-tri-0930T1641` | `ecb46f764` | plugin float32 (GDN inputs BF16, `TRITON_ATTN`) |
| `vela-0930T1648` | `564d5f8e8` | Vela Domain, float32 + bfloat16, vs Transformers |
| `label-0930T1707` | `aac72d927` | Tier 1a label-token demo, shared state in three send orders |
| `coloc-0930T1654` | `564d5f8e8` | Decision + Vela engines on one GPU |
| `flips-bf16-fp32.json` | `a97cf14c0` | changed-answer margins (`v2.serving.flips`) |

## GPU-hours (node A GPU1, shared lease `owner.serving`)

**0.83 GPU-h of the 3 GPU-h cap**: 0.50 in the runs reported above, 0.24 in failed or superseded runs of this
continuation (float32 start failures × 2; Vela × 2 and co-location × 1: shared HF blob store not mounted, then the
endpoint plugin inside the Vela engine; label-token × 3: blob store, an over-long shared state, and one superseded by
the three-order run), 0.09 in the first worker's three failed smokes. Wall-clock per run is in each run's `launcher.json`; the lease file is marked released.
