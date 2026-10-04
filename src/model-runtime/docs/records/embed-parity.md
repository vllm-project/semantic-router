# Embedders, rerankers and Omni: parity

The `task_heads` pooled and relevance heads (Vela 1.0 Embedding, Vela 1.0
Reranker, Qwen3-Embedding-0.6B) and the `multimodal_embedding` family (Vela
1.0 Omni Nano and Mini) against three references, with design section 17's
thresholds.

- **Date:** 2026-10-04.
- **Packages:** `vllm-sr/Vela-1.0-Encoder-307M-Embedding@1e57cebf`,
  `vllm-sr/Vela-1.0-Encoder-307M-Reranker@a388e41c`,
  `Qwen/Qwen3-Embedding-0.6B@97b0c614`; Omni bundles prepared from
  `vllm-sr/Vela-1.0-Omni-Nano@2ff2d663` (bundle `c7cc9a5b26c0…`) and
  `vllm-sr/Vela-1.0-Omni-Mini@801bae3a` (bundle `97d6b2b1045e…`).
- **CPU:** 16 vCPUs of an AMD EPYC 9575F (Zen 5) host in one cgroup cpuset,
  PyTorch 2.10 (oneDNN, MKL), ONNX Runtime 1.30, Transformers 5.18.
- **ROCm:** one AMD Instinct MI325X (gfx942), PyTorch 2.12 on ROCm 7.2, the
  native engine with its encoder graphs (the serving default).
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
`Runtime.call` with its result cache off: the encoder rows at `7c2c6e21b`
(the CPU path of these models has not changed since), the Omni rows at
`bf4a889a9`. Every input once per side; Omni gets the corpus's 26 short
texts, the bundle's golden images as encoded bytes and its golden audio
(channel-major PCM to the facade, a float WAV to the runtime).

| Job | Inputs | Worst cosine | Max \|Δ\| | Rerank inversions outside ties |
| --- | --- | --- | --- | --- |
| Embedding (768, layer 22) | 31 | 1.0000000 | 4.6e-7 | |
| Embedding (256, layer 22) | 31 | 1.0000000 | 8.0e-7 | |
| Embedding (768, layer 11) | 31 | 1.0000000 | 3.5e-7 | |
| Reranker (layer 22, 768) | 4 sets × 8 | | 1.2e-5 (logit) | 0 |
| Reranker (layer 6, 256) | 4 sets × 8 | | 1.8e-5 (logit) | 0 |
| Qwen3-Embedding | 31 | 1.0000000 | 6.1e-7 | |
| Omni Nano text | 26 | 1.0000000 | 2.1e-7 | |
| Omni Nano image | 3 | 1.0000000 | 1.8e-7 | |
| Omni Nano audio | 4 | 1.0000000 | 1.3e-6 | |
| Omni Mini text | 26 | 1.0000000 | 5.6e-7 | |
| Omni Mini image | 3 | 1.0000000 | 1.8e-7 | |
| Omni Mini audio | 5 | 1.0000000 | 2.6e-6 | |

Every job passes the CPU bar (cosine ≥ 0.99999, max |Δ| ≤ 1e-4, identical
rerank order outside ties) by about five orders of magnitude; no input
errors on one side only.

## Against Transformers FP32, same token IDs

The reference is the package loaded by Transformers in FP32 on the CPU with
`output_hidden_states`; exits read its hidden states, intermediate exits raw
(the Embedding contract) or final-normed (the Reranker contract); pooling,
Matryoshka truncation and L2 as the package configures them. The runtime
serves the whole corpus as one request. The CPU rows ran the reference over
one right-padded batch at `ebecf9a3a`: the `exact` path on oneDNN's
pre-packed FP32 linears, and the Reranker's scorer as a two-output packed
linear whose second row is zeros (a one-output packed linear changes with
the batch size, which the load-time batch-invariance probe rejects). The ROCm rows ran each
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

## Omni against the bundle goldens (CPU)

The goldens are the official reference's outputs recorded when the bundle
was prepared (`reference_parity.json`, `golden/`). Every stage is compared,
not only the final vector (runtime at `1c0c95497`; the later Omni changes,
audio preprocessing that builds each resampler once and skips Whisper frames
that hold only padding, are bit-identical by test, and the legacy comparison
above checks the head end to end).

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
which loads no copy; Omni has no copy (its graphs are the bundle's).
