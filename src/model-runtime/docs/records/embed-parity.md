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
- **CPU:** 16 cores of an AMD EPYC 9575F (Zen 5) host, PyTorch 2.10 (MKL),
  ONNX Runtime 1.30, Transformers 5.18.
- **ROCm:** one AMD Instinct MI325X (gfx942), PyTorch 2.12 on ROCm 7.2, the
  native engine with its encoder graphs (the serving default).
- **Tools:** `tools/embed_parity.py` (Transformers and the Omni goldens) and
  `tools/embed_legacy.py` (the legacy router path).

## Against Transformers FP32, same token IDs

The reference runs each input alone (no padding) with
`output_hidden_states`; exits read its hidden states, intermediate exits raw
(the Embedding contract) or final-normed (the Reranker contract); pooling,
Matryoshka truncation and L2 as the package configures them. The corpus is
`tools/embed_corpus.py`: short queries, paragraphs and long documents (to a
few thousand tokens) in seven languages, code and JSON, and rerank sets of
one query with eight documents.

| Model | Device | Engine | Cases | Worst cosine | Max \|Δ\| | Rerank order |
| --- | --- | --- | --- | --- | --- | --- |
| Vela Embedding | CPU | native | 6 (exits 3, 6, 11, 22; dims 768, 128, 64) | 0.9999999999998 | 2.7e-7 | |
| Vela Embedding | CPU | onnxruntime | 6 | 0.9999999999982 | 5.9e-7 | |
| Vela Embedding | ROCm | native | 6 | 0.9999999999935 | 1.1e-6 | |
| Vela Reranker | CPU | native | 20 pair scorers (layers 3, 6, 11, 22 × dims 768–64) | | 3.8e-6 (logit) | identical |
| Vela Reranker | CPU | onnxruntime | 20 | | 2.0e-5 (logit) | identical |
| Vela Reranker | ROCm | native | 20 | | 1.8e-5 (logit) | identical |
| Qwen3-Embedding | CPU | native | 1 (last token, 1,024-d) | 1.0 | 0.0 | |
| Qwen3-Embedding | ROCm | native | 1 | 0.9999999999926 | 6.2e-7 | |

CPU passes the 0.99999 / 1e-4 bar by about five orders of magnitude; ROCm,
whose bar is cosine ≥ 0.9995, stays inside the CPU bar too.

## Against the legacy router path

The legacy side is the router's native facade (`pkg/modelruntime/native`) at
the legacy commit `61aa7eb2d`, built with its bindings, called as the router
called it: `Runtime.Embedding` + `EmbedWithOptions` (candle, adapters
`mmbert` and `qwen3`, contract `embedding.v1`, overflow truncate) and
`Runtime.Relevance` + `ScorePairs` (candle, adapter `vela_reranker`, reject,
a fixed pair scorer). The runtime side serves the same packages through
`Runtime.call` with its result cache off. Every input once per side.

| Job | Inputs | Worst cosine | Max \|Δ\| | Rerank inversions outside ties |
| --- | --- | --- | --- | --- |
| Embedding (768, layer 22) | 31 | 1.0000000 | 5.5e-7 | |
| Embedding (256, layer 22) | 31 | 1.0000000 | 5.6e-7 | |
| Embedding (768, layer 11) | 31 | 1.0000000 | 4.1e-7 | |
| Reranker (layer 22, 768) | 4 sets × 8 | | 1.3e-5 (logit) | 0 |
| Reranker (layer 6, 256) | 4 sets × 8 | | 2.3e-5 (logit) | 0 |
| Qwen3-Embedding | 31 | 1.0000000 | 6.7e-7 | |

## Omni against the bundle goldens (CPU)

The goldens are the official reference's outputs recorded when the bundle
was prepared (`reference_parity.json`, `golden/`). Every stage is compared,
not only the final vector.

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
