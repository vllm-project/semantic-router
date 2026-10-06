# Embedders, rerankers and Omni: parity

The `task_heads` pooled and relevance heads (Vela 1.0 Embedding, Vela 1.0
Reranker, Qwen3-Embedding-0.6B) and the `multimodal_embedding` family (Vela
1.0 Omni Nano and Mini) against three references, with design section 17's
thresholds.

- **Date:** 2026-10-05 (the legacy comparison and the stack B goldens); the
  golden answers at the head on 2026-10-06; the Transformers, Omni-golden and
  reduced-copy rows are from 2026-10-04. The Omni rows on the native engine
  ([#4619](https://github.com/vllm-project/semantic-router/issues/4619)) are
  from 2026-10-06.
- **Packages:** `vllm-sr/Vela-1.0-Encoder-307M-Embedding@1e57cebf`,
  `vllm-sr/Vela-1.0-Encoder-307M-Reranker@a388e41c`,
  `Qwen/Qwen3-Embedding-0.6B@97b0c614`; `vllm-sr/Vela-1.0-Omni-Nano@2ff2d663`
  and `vllm-sr/Vela-1.0-Omni-Mini@801bae3a`, whose published weights the
  native engine serves. The Omni bundles prepared from them (bundle
  `c7cc9a5b26c0…` and `97d6b2b1045e…`) hold the official reference goldens and
  serve the ONNX Runtime rows.
- **CPU:** 16 vCPUs of an AMD EPYC 9575F (Zen 5) host in one cgroup cpuset,
  PyTorch 2.10 (oneDNN, MKL), ONNX Runtime 1.30, Transformers 5.18.
- **ROCm:** one AMD Instinct MI325X (gfx942), PyTorch 2.12 on ROCm 7.2, the
  native engine with its encoder graphs (the serving default). The golden
  check at the head runs the router's ROCm image (`a580be6b9`, vLLM's ROCm
  PyTorch 2.12.0+git6bbd260); the earlier one ran stack B, the official
  wheel (PyTorch 2.12.0+rocm7.2 from the rocm7.2 index, Triton 3.7.0).
- **Tools:** `tools/embed_parity.py` (Transformers, the Omni goldens and the
  reduced copies) and `tools/embed_legacy.py` (the legacy router path).

## Against the legacy router path (design section 17's reference)

The legacy side is the router's native facade (`pkg/modelruntime/native`) at
the legacy commit `61aa7eb2d`, built with its bindings, called as the router
called it: `Runtime.Embedding` + `EmbedWithOptions` (candle, adapters
`mmbert` and `qwen3`, contract `embedding.v1`, overflow truncate),
`Runtime.Relevance` + `ScorePairs` (candle, adapter `vela_reranker`, reject,
a fixed pair scorer) and the Omni adapter `vela_omni` (ONNX Runtime on the
prepared bundle). The runtime side serves the same packages through
`Runtime.call` with its result cache off, at `e0e0e2850`, in the same
16-vCPU cgroup cpuset as the performance run. Every input once per side;
Omni gets the corpus's 26 short texts, the bundle's golden images as encoded
bytes and its golden audio (channel-major PCM to the facade, a float WAV to
the runtime).

| Job | Inputs | Worst cosine | Max \|Δ\| | Rerank inversions outside ties | Passes |
| --- | --- | --- | --- | --- | --- |
| Embedding (768, layer 22) | 31 | 1.0000000 | 4.6e-07 |  | yes |
| Embedding (256, layer 22) | 31 | 1.0000000 | 8.0e-07 |  | yes |
| Embedding (768, layer 11) | 31 | 1.0000000 | 3.5e-07 |  | yes |
| Reranker (layer 22, 768) | 4 sets × 8 |  | 1.2e-05 (logit) | 0 | yes |
| Reranker (layer 6, 256) | 4 sets × 8 |  | 1.8e-05 (logit) | 0 | yes |
| Qwen3-Embedding | 31 | 1.0000000 | 6.1e-07 |  | yes |
| Omni Nano text | 26 | 1.0000000 | 2.1e-07 |  | yes |
| Omni Nano image | 3 | 1.0000000 | 1.8e-07 |  | yes |
| Omni Nano audio | 4 | 1.0000000 | 1.3e-06 |  | yes |
| Omni Mini text | 26 | 1.0000000 | 5.6e-07 |  | yes |
| Omni Mini image | 3 | 1.0000000 | 1.8e-07 |  | yes |
| Omni Mini audio | 5 | 1.0000000 | 2.6e-06 |  | yes |

Every job passes the CPU bar (cosine ≥ 0.99999, max |Δ| ≤ 1e-4, identical
rerank order outside ties) by about five orders of magnitude; no input
errors on one side only. Running a batch's Omni images one after another and
the per-graph pools change no value: every Omni input runs its graphs alone,
and the family's test checks that a batch answers each input as it answers
alone.

**Omni on the native engine (#4619),** the same legacy values against the
runtime at `2afe0f878` serving the published weights (node B vCPUs 64–79,
memory on their NUMA node, PyTorch 2.10.0):

| Job | Inputs | Worst cosine | Max \|Δ\| | Passes |
| --- | --- | --- | --- | --- |
| Omni Nano text | 26 | 0.9999999999996 | 2.4e-07 | yes |
| Omni Nano image | 3 | 0.9999999999987 | 3.6e-07 | yes |
| Omni Nano audio | 4 | 0.9999999999660 | 1.3e-06 | yes |
| Omni Mini text | 26 | 0.9999999999878 | 6.6e-07 | yes |
| Omni Mini image | 3 | 0.9999999999986 | 2.0e-07 | yes |
| Omni Mini audio | 5 | 0.9999999998042 | 2.6e-06 | yes |

The native engine runs each image and audio input alone, and texts in one
packed batch only where the load-time probe finds the model batch-invariant
(design section 8.5).

## Golden answers at the head

At `c05aa8b77` (this branch with staging `58cbe432e` merged),
`tools/golden_answers.py` recorded each model's answers into a copy of its
golden file (`registry/golden_answers_vela1.json`, `_omni.json`):

| Model | ROCm, router image `a580be6b9`, two fresh processes | CPU |
| --- | --- | --- |
| Vela Embedding | 1,536 / 1,536 values equal | matched, max \|Δ\| 3.0e-7 |
| Vela Reranker | 2 / 2 values equal | matched, max \|Δ\| 4.8e-7 |
| Qwen3-Embedding | 2,048 / 2,048 values equal | matched, max \|Δ\| 2.6e-7 |
| Omni Nano | CPU only | matched, max \|Δ\| 1.1e-7 |
| Omni Mini | CPU only | matched, max \|Δ\| 7.2e-8 |

**Omni on the native engine (#4619).** The `cpu` answers above were recorded on
the ONNX Runtime bundle. The native engine moves them by at most 4.8e-7 (Nano)
and 2.9e-7 (Mini), inside the CPU tolerance (readiness `matched`, 1,152 and
2,304 values), so `_omni.json` is re-recorded from the native engine, in the
CPU router image `1ff74ff22` on node D (PyTorch 2.10.0, 16 threads). The
`rocm` answers are new: the ROCm router image `1ff74ff22` on node D GPU6, one
process to record and a fresh one that repeats every value exactly (1,152 /
1,152 and 2,304 / 2,304). They lie within 4.8e-7 of the CPU answers. No other
golden file changes.

- **ROCm:** the router's ROCm image, `Dockerfile.extproc` at `a580be6b9` (`ACCELERATOR=rocm`), on node D GPU1;
  readiness `matched` in both processes, and every recorded value equals the
  file's `rocm` answers.
- **CPU:** PyTorch 2.10.0 CPU and ONNX Runtime 1.30.0, the router CPU image's
  pins; the three encoders on 16 threads of node D, Omni on 4 threads of node
  B (its prepared bundle is there). Readiness `matched` (the CPU tolerance is
  1e-3). The values move in their last bits with the host and the thread
  count, so they are not byte-identical to the record.

## ROCm goldens on stack B

The `task_heads` readiness check compares a model's golden request with the
answers recorded for its device class (`registry/golden_answers_vela1.json`,
`rocm`). On stack B at `e0e0e2850` (node B GPU2, two fresh processes, the
first with an empty Triton cache), every recorded value matched in both: Vela
Embedding 1,536 / 1,536, Vela Reranker 2 / 2 and Qwen3-Embedding 2,048 /
2,048, and the two processes' answers are byte-identical. `vela1`'s check of
all thirteen `task_heads` built-ins at `35ff3a8a5` (the same embedder code
apart from request-option parsing and the decoder graphs' capture mode)
matched byte for byte in three fresh processes. The stack change therefore
re-records nothing for these models.

## Against Transformers FP32, same token IDs

The reference is the package loaded by Transformers in FP32 on the CPU with
`output_hidden_states`; exits read its hidden states, intermediate exits raw
(the Embedding contract) or final-normed (the Reranker contract); pooling,
Matryoshka truncation and L2 as the package configures them. The runtime
serves the whole corpus as one request. The CPU rows ran the reference over
one right-padded batch at `ebecf9a3a`: the `exact` path on oneDNN's pre-packed
FP32 linears, and the Reranker's scorer as a two-output packed linear whose
second row is zeros (a one-output packed linear changes with the batch size,
which the load-time batch-invariance probe rejects). The ROCm rows ran each
reference input alone, at `5a73fc17e` (gfx942 fused rotary, encoder graphs).
The corpus is `tools/embed_corpus.py`: short queries, paragraphs and long
documents (to a few thousand tokens) in seven languages, code and JSON, and
rerank sets of one query with eight documents.

| Model | Device | Engine | Cases | Worst cosine | Max \|Δ\| | Rerank order |
| --- | --- | --- | --- | --- | --- | --- |
| Vela Embedding | CPU | native | 6 (exits 3, 6, 11, 22; dims 768, 128, 64) | 0.9999999999975 | 8.4e-7 | |
| Vela Embedding | CPU | onnxruntime | 6 | 0.9999999999986 | 6.4e-7 | |
| Vela Embedding | ROCm | native | 6 | 0.9999999999886 | 1.5e-6 | |
| Vela Reranker | CPU | native | 20 pair scorers (layers 3, 6, 11, 22 × dims 768–64) | | 1.8e-5 (logit) | identical |
| Vela Reranker | CPU | onnxruntime | 20 | | 2.0e-5 (logit) | identical |
| Vela Reranker | ROCm | native | 20 | | 1.8e-5 (logit) | identical |
| Qwen3-Embedding | CPU | native | 1 (last token, 1,024-d) | 0.9999999999923 | 7.1e-7 | |
| Qwen3-Embedding | ROCm | native | 1 | 0.9999999999926 | 6.2e-7 | |

ROCm, whose bar is cosine ≥ 0.9995, stays inside the CPU bar too.

## Omni against the official reference goldens

The goldens are the official reference's outputs recorded when the bundle
was prepared (`reference_parity.json`, `golden/`). Every stage is compared,
not only the final vector.

**On the native engine (#4619),** `tools/embed_parity.py omni --snapshot`
serves the published files in the router images built from `1ff74ff22` on
node D: the CPU image on 16 vCPUs (PyTorch 2.10.0), and the ROCm image on
GPU6 (one MI325X, PyTorch 2.12.0+git6bbd260). The bundles that hold the
goldens were exported again there with `tools/models/vela_omni/Dockerfile`;
their graphs and goldens are byte-identical to the earlier export.

| Stage | Nano CPU | Nano ROCm | Mini CPU | Mini ROCm |
| --- | --- | --- | --- | --- |
| Cases | 12 | 12 | 15 | 15 |
| Token IDs, pixels, decoded PCM | identical | identical | identical | identical |
| Resampled audio, max \|Δ\| | 1.2e-7 | 1.2e-7 | 1.2e-7 | 1.2e-7 |
| Whisper log-mel, max \|Δ\| | 7.5e-5 | 7.5e-5 | 7.5e-5 | 7.5e-5 |
| CLAP log-mel (dB), max \|Δ\| | 1.1e-2 | 1.1e-2 | 1.7e-2 | 1.7e-2 |
| CLAP window embeddings, max \|Δ\| | 1.5e-6 | 1.5e-6 | 4.4e-6 | 4.4e-6 |
| Final embeddings, worst cosine | 0.99999999998522 | 0.99999999998558 | 0.99999999984570 | 0.99999999984470 |
| Final embeddings, max \|Δ\| | 8.9e-7 | 8.1e-7 | 1.9e-6 | 1.8e-6 |

Both devices pass the CPU bar (cosine ≥ 0.99999, max |Δ| ≤ 1e-4), the GPU's
bar being cosine ≥ 0.9995. Tower by tower, on the real weights against
Transformers 4.57.6 and PyTorch 2.8 on the CPU, BERT, Qwen3, both Whisper
encoders and HTSAT are bit-identical and SigLIP is within 1.9e-6, a spread
MKL's one-row products already show on identical inputs at different buffer
alignments; `tests/test_omni_towers.py` checks each tower on random weights.

**On the ONNX Runtime bundle,** the runtime at `1c0c95497` (the later Omni
changes, audio preprocessing that builds each resampler once and skips Whisper
frames that hold only padding, are bit-identical by test):

| Stage | Nano (12 cases) | Mini (15 cases) |
| --- | --- | --- |
| Token IDs | identical | identical (incl. the query instruction) |
| Pixels (PNG, JPEG) | identical | identical |
| Decoded PCM (16-bit, 24-bit, float WAV) | identical | identical |
| Resampled audio, max \|Δ\| | 1.2e-7 | 1.2e-7 |
| Whisper log-mel, max \|Δ\| | 7.5e-5 | 7.5e-5 |
| CLAP log-mel (dB), max \|Δ\| | 1.1e-2 | 1.7e-2 |
| CLAP window embeddings, max \|Δ\| | 1.5e-6 | 4.5e-6 |
| Final embeddings | cosine ≥ 0.99999999998, max \|Δ\| 8.8e-7 | cosine ≥ 0.99999999985, max \|Δ\| 1.8e-6 |

The CLAP log-mel differences sit in the dB domain (values of tens of dB) and
vanish through the encoder: the CLAP embeddings agree to 4.5e-6.

## `max_speed` reduced copies against the exact path

`tools/embed_parity.py reduced` loads the model with a copy of its linear
layers and answers the parity corpus through both paths; the floor is
design section 5.4's (embeddings: cosine ≥ 0.999 per vector; reranking: 99 %
pairwise order agreement on document pairs whose exact logits differ by more
than 1e-3). Latency is the per-request p50 of both paths over the corpus in
the same process. Measured at `fe65a8d5c`, when `exact` still ran
`F.linear`.

| Model | Device | Copy | Worst cosine / order agreement | Max \|Δ\| | Exact → copy p50 ms | Floor |
| --- | --- | --- | --- | --- | --- | --- |
| Vela Embedding | CPU | `float32-packed` | 1.0000000 | 1e-6 | 35.8 → 16.6 | pass |
| Vela Embedding | CPU | `bfloat16` | 0.99989 | 5.4e-3 | 51.0 → 51.4 | pass |
| Vela Embedding | CPU | `int8` | 0.634 | 0.34 | 26.1 → 11.7 | fail |
| Vela Embedding | ROCm | `bfloat16` | 0.99989 | 7.3e-3 | 2.58 → 4.29 | pass |
| Vela Reranker | CPU | `float32-packed` | 100 % | 1.5e-5 (logit) | 122.4 → 59.7 | pass |
| Vela Reranker | CPU | `bfloat16` | 98.2 % | 0.16 (logit) | 133.5 → 63.8 | fail |
| Vela Reranker | CPU | `int8` | 80.2 % | 4.6 (logit) | 122.7 → 37.9 | fail |
| Vela Reranker | ROCm | `bfloat16` | 98.2 % (layer-3 exits) | 0.14 (logit) | 3.87 → 3.97 | fail |

Decision: no `BuiltinModel.reduced` entry for Vela Embedding or Reranker.
Dynamic int8 breaks both models (per-tensor activation scales against
ModernBERT's activation outliers), as it breaks the Decision 1.0 encoders.
BF16 keeps the Embedding above the floor but buys nothing: it is no faster
on the CPU, and on the GPU a single request is launch-bound and autocast
adds a cast per linear (2.6 → 4.3 ms). For the Reranker, BF16 is faster on
the CPU but its shallow exits fall below the floor on both devices.
oneDNN's pre-packed FP32 linears are the exact path's numbers within 1.5e-5
and twice as fast, so the `exact` path itself now runs them on x86 CPUs (the
`task_heads` kernel variants, as for the classify heads), which leaves a
`float32-packed` copy nothing to gain. Qwen3-Embedding runs the decoder path,
which loads no copy; Omni runs no reduced copy.
