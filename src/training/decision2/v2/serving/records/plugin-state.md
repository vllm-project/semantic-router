# vLLM plugin prototype — state (keep current; newest first)

Assignment: coordinator notes 2026-09-30 19:40 (serving track), 19:50 / 20:00 (runtime design, `/v1/decisions`),
~23:40 (Tier 1a / 1b split) and the 23:55 continuation. Branch `xunzhuo/decision-2-vllm-plugin`, gist
`09-decision-2-serving.md`. GPU: node A GPU1 (shared lease `owner.serving`), ≤ 3 GPU-h in total. Nothing uploaded.

## Now

- 2026-10-01 ≈00:40 UTC+8 — **Continuation (6c2c1341) resumed after the ~20:40 silent stop.**
  - Found: three failed plugin sessions on node A GPU1 (0.093 GPU-h). The last (`p08-bf16-0930T124140`, mirror
    `678ac2060`) loaded the model and failed at the first request: `from __future__ import annotations` turned the
    route's `Request` annotation into a string, so FastAPI treated it as a query parameter (HTTP 400). The
    uncommitted fix (drop the future import) is valid and kept.
  - Code: `/v1/decisions` is now the route (entry point `vllm_sr_decisions`), `/v1/system_one` an alias; a FastAPI
    route test through the service would have caught the bug on CPU; sessions call `/v1/decisions`; panels take an
    optional concurrency; BF16-resident conversion counts non-exact Linear weights instead of failing; new
    `labeltoken` session (Tier 1a demo on the generate runner with stock Qwen3.5-0.8B).
  - Next: mirror, then on node A GPU1 in order: plugin bf16 (4 panels + bench), plugin fp32, runtime (FP32 master +
    BF16-resident, frozen autotune cache), vela, labeltoken, coloc if budget allows; record + gist 09 + merge.

## Inputs (node A)

| Input | Path | Check |
| --- | --- | --- |
| DEV2.0-0.8B (BF16 storage, revision `bede7938`) | `/data/dev2/runs/release/dev2-bf16-0.8B-20260929T224530Z/download/DEV2.0-0.8B` | manifest `a653dde2…` |
| Stored scored predictions | `/data/dev2/runs/release/inputs/dev2-0p8b-t1/derived/*.predictions.jsonl` | = release parity-post digests |
| Gold-free panel prompts | `/data/dev2/private/panels/goldfree/*.prompts.jsonl` | typed-final 1,600; css15 6,547; public231 231; mlx-diag 2,275 |
| Vela-1.0-Encoder-307M-Domain | `/data/dev2/hf-cache/models--llm-semantic-router--Vela-1.0-Encoder-307M-Domain` | router `domain_classifier` |
| Qwen3.5-0.8B (stock) | `/data/dev2/hf-cache/models--Qwen--Qwen3.5-0.8B` | Tier 1a demo |

## GPU-hours (node A GPU1)

| Run | Mirror | GPU-h | Outcome |
| --- | --- | --- | --- |
| smoke-bf16-122544 | `1408d273a` | 0.0164 | engine failed (plugin task rejected by Model Runner V2) |
| smoke-bf16-123241 | `711330eac` | 0.0392 | engine failed (runtime predates Score offsets) |
| p08-bf16-0930T124140 | `678ac2060` | 0.0372 | first request HTTP 400 (FastAPI annotation) |
| **Total** | | **0.093** | |

## Plan

| Step | Status |
| --- | --- |
| Plugin v0: model class, pooler, `/v1/decisions`, pinned build | code done; first served answers pending |
| DEV2.0-0.8B parity, bfloat16 and float32 | pending |
| Vela-1.0 Domain via native ModernBERT classify vs Transformers | pending |
| Serving-shape notes (+ co-located measurement if cheap) | pending |
| Latency / throughput: plugin vs shipped runtime (+ BF16-resident) | pending |
| Tier 1a label-token demo (generate runner, ≤ 0.2 GPU-h) | pending |
| Record, gist 09, merge into `xunzhuo/decision-2-training` | pending |
